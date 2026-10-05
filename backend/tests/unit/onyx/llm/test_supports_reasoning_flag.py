from typing import Any
from unittest.mock import patch

from onyx.llm.constants import LlmProviderNames
from onyx.llm.model_request import UserMessage
from onyx.llm.models import ReasoningEffort
from onyx.llm.multi_llm import LitellmLLM

_COMPLETION = "onyx.llm.litellm_singleton.litellm.completion"
# Not in the LiteLLM registry, so only the admin flag can mark it as reasoning.
_UNLISTED_OLLAMA_MODEL = "gemma4:31b-cloud"


def _sent_kwargs(supports_reasoning: bool) -> dict[str, Any]:
    llm = LitellmLLM(
        api_key=None,
        model_provider=LlmProviderNames.OLLAMA_CHAT,
        model_name=_UNLISTED_OLLAMA_MODEL,
        max_input_tokens=100000,
        api_base="http://localhost:11434",
        supports_reasoning=supports_reasoning,
    )
    with patch(_COMPLETION) as completion:
        llm._completion(
            prompt=[UserMessage(content="hello")],
            tools=None,
            tool_choice=None,
            stream=False,
            parallel_tool_calls=False,
            reasoning_effort=ReasoningEffort.MEDIUM,
        )
    return dict(completion.call_args.kwargs)


def test_unlisted_model_sends_no_reasoning_without_flag() -> None:
    assert "reasoning_effort" not in _sent_kwargs(supports_reasoning=False)


def test_admin_flag_sends_reasoning_for_unlisted_model() -> None:
    assert _sent_kwargs(supports_reasoning=True).get("reasoning_effort") == "medium"
