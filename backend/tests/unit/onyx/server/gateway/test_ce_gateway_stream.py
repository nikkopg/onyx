import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import patch

import pytest

from onyx.error_handling.exceptions import OnyxError
from onyx.llm.constants import LlmProviderNames
from onyx.llm.interfaces import LLMUserIdentity
from onyx.llm.model_response import (
    ChatCompletionDeltaToolCall,
    Delta,
    ModelResponseStream,
    ResponseFunctionCall,
    StreamingChoice,
)
from onyx.llm.models import ReasoningEffort, Usage
from onyx.llm.multi_llm import LitellmLLM, LLMRateLimitError
from onyx.server.gateway.api import _resolve_model, _stream_chat_completion
from onyx.server.gateway.chat_completion_translation import RawLLMCallArgs
from onyx.server.manage.llm.models import LLMProviderView, ModelConfigurationView
from onyx.tracing.flows import LLMFlow


def _llm() -> LitellmLLM:
    return LitellmLLM(
        api_key=None,
        model_provider=LlmProviderNames.OLLAMA_CHAT,
        model_name="gemma4:31b-cloud",
        max_input_tokens=100000,
        api_base="http://localhost:11434",
    )


def _call_args() -> RawLLMCallArgs:
    return RawLLMCallArgs(
        prompt=[],
        tools=None,
        tool_choice=None,
        structured_response_format=None,
        max_tokens=None,
        reasoning_effort=ReasoningEffort.AUTO,
    )


def _chunk(delta: Delta, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-1",
        created="1700000000",
        choice=StreamingChoice(delta=delta, finish_reason=finish_reason),
    )


def _frames(chunks: list[ModelResponseStream] | Exception) -> list[Any]:
    def fake_stream(*_args: Any, **_kwargs: Any) -> Iterator[ModelResponseStream]:
        if isinstance(chunks, Exception):
            raise chunks
        yield from chunks

    llm = _llm()
    with patch.object(LitellmLLM, "stream_raw", side_effect=fake_stream):
        body = "".join(
            _stream_chat_completion(
                llm,
                _call_args(),
                "1/gemma4:31b-cloud",
                LLMFlow.CRAFT_LLM_GENERATION,
                LLMUserIdentity(user_id="u"),
            )
        )
    frames: list[Any] = []
    for block in body.split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: ")
        payload = block[len("data: ") :]
        frames.append(payload if payload == "[DONE]" else json.loads(payload))
    return frames


def test_stream_forwards_reasoning_text_and_tool_calls() -> None:
    last = _chunk(Delta(), finish_reason="tool_calls")
    last = last.model_copy(
        update={
            "usage": Usage(
                prompt_tokens=5,
                completion_tokens=7,
                total_tokens=12,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        }
    )
    frames = _frames(
        [
            _chunk(Delta(reasoning_content="Thinking.")),
            _chunk(Delta(content="Calling.")),
            _chunk(
                Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id="call_1",
                            index=0,
                            function=ResponseFunctionCall(
                                name="plane-1_cycle", arguments='{"a":1}'
                            ),
                        )
                    ]
                )
            ),
            last,
        ]
    )

    assert frames[-1] == "[DONE]"
    first_delta = frames[0]["choices"][0]["delta"]
    assert first_delta["role"] == "assistant"
    assert first_delta["reasoning_content"] == "Thinking."
    assert "role" not in frames[1]["choices"][0]["delta"]
    assert frames[1]["choices"][0]["delta"]["content"] == "Calling."
    tool_call = frames[2]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "plane-1_cycle"
    assert frames[3]["choices"][0]["finish_reason"] == "tool_calls"
    assert frames[3]["usage"]["prompt_tokens"] == 5
    assert all(frame["model"] == "1/gemma4:31b-cloud" for frame in frames[:-1])


def test_stop_after_tool_calls_is_reported_as_tool_calls() -> None:
    tool_delta = Delta(
        tool_calls=[
            ChatCompletionDeltaToolCall(
                id="call_1", function=ResponseFunctionCall(name="t", arguments="{}")
            )
        ]
    )
    frames = _frames([_chunk(tool_delta), _chunk(Delta(), finish_reason="stop")])
    assert frames[1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stop_without_tool_calls_stays_stop() -> None:
    frames = _frames([_chunk(Delta(content="hi"), finish_reason="stop")])
    assert frames[0]["choices"][0]["finish_reason"] == "stop"


def test_stream_error_does_not_leak_upstream_details() -> None:
    from litellm.exceptions import APIConnectionError

    error = APIConnectionError(
        message="Cannot connect to host internal-llm.corp:11434",
        llm_provider="ollama_chat",
        model="gemma4:31b-cloud",
    )
    frames = _frames(error)
    assert frames[0]["error"]["type"] == "provider_error"
    assert frames[0]["error"]["code"] == "CONNECTION_ERROR"
    assert "internal-llm.corp" not in frames[0]["error"]["message"]
    assert frames[-1] == "[DONE]"


def test_stream_error_before_first_chunk_still_ends_stream() -> None:
    with patch(
        "onyx.server.gateway.api.llm_generation_span",
        side_effect=RuntimeError("span setup failed"),
    ):
        frames = _frames([_chunk(Delta(content="never sent"))])
    assert "error" in frames[0]
    assert frames[-1] == "[DONE]"


def test_stream_rate_limit_error_type() -> None:
    frames = _frames(LLMRateLimitError("slow down"))
    assert frames[0]["error"]["type"] == "rate_limited"


def test_stream_can_be_consumed_from_different_threads() -> None:
    """StreamingResponse may pull each frame on a different worker thread; the
    generation span must not be entered and exited across those contexts."""
    chunks = [_chunk(Delta(content="a")), _chunk(Delta(content="b"), "stop")]

    def fake_stream(*_args: Any, **_kwargs: Any) -> Iterator[ModelResponseStream]:
        yield from chunks

    with patch.object(LitellmLLM, "stream_raw", side_effect=fake_stream):
        generator = _stream_chat_completion(
            _llm(),
            _call_args(),
            "1/gemma4:31b-cloud",
            LLMFlow.CRAFT_LLM_GENERATION,
            LLMUserIdentity(user_id="u"),
        )
        received: list[str] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            while True:
                frame = pool.submit(next, generator, None).result()
                if frame is None:
                    break
                received.append(frame)

    assert received[-1] == "data: [DONE]\n\n"
    assert len(received) == 3


def _model(name: str, *, visible: bool = True) -> ModelConfigurationView:
    return ModelConfigurationView(
        name=name,
        display_name=name,
        is_visible=visible,
        supports_image_input=False,
        supports_reasoning=False,
    )


def _provider(models: list[ModelConfigurationView]) -> LLMProviderView:
    return LLMProviderView(
        id=1,
        name="Ollama",
        provider=LlmProviderNames.OLLAMA_CHAT,
        api_key=None,
        model_configurations=models,
    )


def test_resolve_model_accepts_slashes_in_model_name() -> None:
    provider = _provider([_model("deepseek/deepseek-v4-pro")])
    resolved, model_name = _resolve_model("1/deepseek/deepseek-v4-pro", [provider])
    assert resolved is provider
    assert model_name == "deepseek/deepseek-v4-pro"


@pytest.mark.parametrize(
    "model_id",
    ["gemma4:31b-cloud", "x/gemma4:31b-cloud", "2/gemma4:31b-cloud", "1/hidden"],
)
def test_resolve_model_rejects_unknown_or_hidden(model_id: str) -> None:
    provider = _provider(
        [
            _model("gemma4:31b-cloud"),
            _model("hidden", visible=False),
        ]
    )
    with pytest.raises(OnyxError):
        _resolve_model(model_id, [provider])


def _route_mocks(provider_with_key: LLMProviderView | None) -> dict[str, Any]:
    return {
        "gateway_request_flow": LLMFlow.CRAFT_LLM_GENERATION,
        "check_token_rate_limits": None,
        "fetch_all_accessible_llm_providers": [_provider([_model("gemma4:31b-cloud")])],
        "fetch_accessible_llm_provider_by_id": provider_with_key,
    }


def _call_route(provider_with_key: LLMProviderView | None) -> tuple[Any, Any]:
    from unittest.mock import MagicMock

    from onyx.server.gateway import api
    from onyx.server.gateway.models import ChatCompletionRequest

    session = MagicMock()
    user = MagicMock()
    user.id = "u"
    patches = [
        patch.object(api, name, return_value=value)
        for name, value in _route_mocks(provider_with_key).items()
    ]
    for p in patches:
        p.start()
    try:
        with patch.object(api, "llm_from_provider", return_value=_llm()) as factory:
            response = api.create_chat_completion(
                ChatCompletionRequest(
                    model="1/gemma4:31b-cloud",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                    temperature=0.2,
                ),
                MagicMock(),
                user,
                session,
            )
            return response, (factory, session)
    finally:
        for p in patches:
            p.stop()


def test_route_uses_provider_with_api_key_and_releases_session() -> None:
    keyed = _provider([_model("gemma4:31b-cloud")]).model_copy(
        update={"api_key": "sk-real"}
    )
    _, (factory, session) = _call_route(keyed)
    assert factory.call_args.kwargs["llm_provider"].api_key == "sk-real"
    assert factory.call_args.kwargs["temperature"] == 0.2
    session.close.assert_called()


def test_route_rejects_provider_lost_between_lookups() -> None:
    with pytest.raises(OnyxError):
        _call_route(None)


def test_client_disconnect_closes_the_provider_stream() -> None:
    import threading

    closed = threading.Event()

    def endless_stream(*_args: Any, **_kwargs: Any) -> Iterator[ModelResponseStream]:
        try:
            while True:
                yield _chunk(Delta(content="x"))
        finally:
            closed.set()

    with patch.object(LitellmLLM, "stream_raw", side_effect=endless_stream):
        generator = _stream_chat_completion(
            _llm(),
            _call_args(),
            "1/gemma4:31b-cloud",
            LLMFlow.CRAFT_LLM_GENERATION,
            LLMUserIdentity(user_id="u"),
        )
        assert next(generator).startswith("data: ")
        generator.close()
        assert closed.wait(timeout=10)
