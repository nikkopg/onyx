"""OpenAI-compatible LLM gateway for the Community Edition.

Serves ``POST /gateway/v1/chat/completions``, the only gateway route Craft's
OpenCode agent calls. Registered only when the Enterprise Edition is off,
since the Enterprise app serves its own gateway at the same prefix.
"""

import json
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.llm import (
    fetch_accessible_llm_provider_by_id,
    fetch_all_accessible_llm_providers,
)
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.factory import llm_from_provider
from onyx.llm.interfaces import LLMUserIdentity
from onyx.llm.multi_llm import LitellmLLM, LLMRateLimitError, LLMTimeoutError
from onyx.llm.utils import litellm_exception_to_error_msg
from onyx.server.features.build.craft_gateway import gateway_request_flow
from onyx.server.gateway.chat_completion_translation import (
    RawLLMCallArgs,
    translate_chat_completion_request,
)
from onyx.server.gateway.configs import (
    GATEWAY_LLM_TOTAL_TIMEOUT_SECONDS,
    GATEWAY_PATH_PREFIX,
)
from onyx.server.gateway.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from onyx.server.manage.llm.models import LLMProviderView
from onyx.server.query_and_chat.token_limit import check_token_rate_limits
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.create import trace
from onyx.tracing.framework.traces import Trace
from onyx.tracing.llm_utils import (
    llm_generation_span,
    record_llm_response,
    record_llm_span_output,
)
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_in_background

logger = setup_logger()

router = APIRouter(prefix=GATEWAY_PATH_PREFIX)


def _resolve_model(
    model_id: str, providers: list[LLMProviderView]
) -> tuple[LLMProviderView, str]:
    """``model_id`` is ``"<llm_provider_id>/<model_name>"`` (the id Craft's
    session config uses); the model name may itself contain slashes."""
    provider_id_raw, _, model_name = model_id.partition("/")
    if not provider_id_raw.isdigit() or not model_name:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Model must look like '<provider_id>/<model_name>', got '{model_id}'.",
        )
    provider_id = int(provider_id_raw)
    for provider in providers:
        if provider.id != provider_id:
            continue
        for model_configuration in provider.model_configurations:
            if (
                model_configuration.name == model_name
                and model_configuration.is_visible
            ):
                return provider, model_name
    raise OnyxError(
        OnyxErrorCode.NOT_FOUND,
        f"Model '{model_id}' is not available to this user.",
    )


def _client_error(error: Exception, llm: LitellmLLM) -> tuple[str, str]:
    """``(message, code)`` safe to show the caller. Uses the same mapping as
    chat, so connection details stay in the server log."""
    if isinstance(error, LLMTimeoutError) and error.__cause__ is None:
        return (
            f"The LLM call exceeded the gateway timeout of {GATEWAY_LLM_TOTAL_TIMEOUT_SECONDS}s.",
            "TIMEOUT",
        )
    message, code, _ = litellm_exception_to_error_msg(error, llm)
    return message, code


def _error_frame(error: Exception, llm: LitellmLLM) -> str:
    """OpenAI-style in-band error."""
    if isinstance(error, LLMRateLimitError):
        error_type = "rate_limit_error"
    elif isinstance(error, LLMTimeoutError):
        error_type = "timeout_error"
    else:
        error_type = "upstream_error"
    message, code = _client_error(error, llm)
    payload = {"error": {"message": message, "type": error_type, "code": code}}
    return f"data: {json.dumps(payload)}\n\n"


def _gateway_trace(flow: LLMFlow, model_id: str) -> Trace:
    return trace("llm_gateway", metadata={"flow": flow.value, "model": model_id})


def _produce_frames(
    llm: LitellmLLM,
    call_args: RawLLMCallArgs,
    model_id: str,
    flow: LLMFlow,
    user_identity: LLMUserIdentity,
    frames: "queue.Queue[str | None]",
    cancelled: threading.Event,
) -> None:
    """Runs in one thread for the whole stream: the trace and the generation
    span are contextvars and must open and close in the same context."""
    deadline = time.monotonic() + GATEWAY_LLM_TOTAL_TIMEOUT_SECONDS
    try:
        with (
            _gateway_trace(flow, model_id),
            llm_generation_span(
                llm=llm,
                flow=flow,
                input_messages=call_args.prompt,
                tools=call_args.tools,
            ) as span,
        ):
            content: list[str] = []
            reasoning: list[str] = []
            usage = None
            sent_tool_calls = False
            try:
                for index, chunk in enumerate(
                    llm.stream_raw(
                        prompt=call_args.prompt,
                        tools=call_args.tools,
                        tool_choice=call_args.tool_choice,
                        structured_response_format=call_args.structured_response_format,
                        max_tokens=call_args.max_tokens,
                        reasoning_effort=call_args.reasoning_effort,
                        user_identity=user_identity,
                    )
                ):
                    if cancelled.is_set():
                        return
                    if time.monotonic() > deadline:
                        raise LLMTimeoutError(
                            f"Gateway total timeout of {GATEWAY_LLM_TOTAL_TIMEOUT_SECONDS}s exceeded."
                        )
                    content.append(chunk.choice.delta.content or "")
                    reasoning.append(chunk.choice.delta.reasoning_content or "")
                    usage = chunk.usage or usage
                    sent_tool_calls = sent_tool_calls or bool(
                        chunk.choice.delta.tool_calls
                    )
                    wire = ChatCompletionChunk.from_stream_chunk(
                        chunk, model=model_id, include_role=index == 0
                    ).to_wire()
                    # Some providers (e.g. Ollama) end a tool-calling turn with
                    # "stop"; OpenAI clients expect "tool_calls" there.
                    for choice in wire["choices"]:
                        if sent_tool_calls and choice.get("finish_reason") == "stop":
                            choice["finish_reason"] = "tool_calls"
                    frames.put(f"data: {json.dumps(wire)}\n\n")
            finally:
                record_llm_span_output(
                    span,
                    output="".join(content),
                    usage=usage,
                    reasoning="".join(reasoning) or None,
                )
    except Exception as e:
        logger.warning(
            "CE gateway stream failed for model %s: %s: %s",
            model_id,
            type(e).__name__,
            e,
        )
        frames.put(_error_frame(e, llm))
    finally:
        frames.put("data: [DONE]\n\n")
        frames.put(None)


def _stream_chat_completion(
    llm: LitellmLLM,
    call_args: RawLLMCallArgs,
    model_id: str,
    flow: LLMFlow,
    user_identity: LLMUserIdentity,
) -> Iterator[str]:
    frames: queue.Queue[str | None] = queue.Queue()
    cancelled = threading.Event()
    run_in_background(
        _produce_frames,
        llm,
        call_args,
        model_id,
        flow,
        user_identity,
        frames,
        cancelled,
    )
    try:
        while (frame := frames.get()) is not None:
            yield frame
    finally:
        # Stops the producer at its next chunk when the client disconnects.
        cancelled.set()


@router.post("/v1/chat/completions")
def create_chat_completion(
    chat_request: ChatCompletionRequest,
    request: Request,
    user: User = Depends(require_permission(Permission.USE_LLM_GATEWAY)),
    db_session: Session = Depends(get_session),
) -> Response:
    flow = gateway_request_flow(request, user)
    if flow is None:
        raise OnyxError(
            OnyxErrorCode.INSUFFICIENT_PERMISSIONS,
            "This token cannot call the LLM gateway.",
        )
    check_token_rate_limits(user)

    call_args = translate_chat_completion_request(chat_request)
    # The connection must not stay checked out for a stream that can last
    # minutes, so all DB work happens before the LLM call.
    try:
        provider, model_name = _resolve_model(
            chat_request.model, fetch_all_accessible_llm_providers(db_session, user)
        )
        # The catalog above has no API keys; load the chosen provider with its key.
        provider_with_key = fetch_accessible_llm_provider_by_id(
            db_session, user, provider.id
        )
    finally:
        db_session.close()
    if provider_with_key is None:
        raise OnyxError(
            OnyxErrorCode.NOT_FOUND,
            f"Model '{chat_request.model}' is not available to this user.",
        )
    llm = llm_from_provider(
        model_name=model_name,
        llm_provider=provider_with_key,
        temperature=chat_request.temperature,
    )
    user_identity = LLMUserIdentity(user_id=str(user.id))

    if chat_request.stream:
        return StreamingResponse(
            _stream_chat_completion(
                llm, call_args, chat_request.model, flow, user_identity
            ),
            media_type="text/event-stream",
        )

    with (
        _gateway_trace(flow, chat_request.model),
        llm_generation_span(
            llm=llm, flow=flow, input_messages=call_args.prompt, tools=call_args.tools
        ) as span,
    ):
        try:
            response = llm.invoke_raw(
                prompt=call_args.prompt,
                tools=call_args.tools,
                tool_choice=call_args.tool_choice,
                structured_response_format=call_args.structured_response_format,
                max_tokens=call_args.max_tokens,
                reasoning_effort=call_args.reasoning_effort,
                user_identity=user_identity,
                total_timeout_s=GATEWAY_LLM_TOTAL_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(
                "CE gateway call failed for model %s: %s: %s",
                chat_request.model,
                type(e).__name__,
                e,
            )
            message, _ = _client_error(e, llm)
            if isinstance(e, LLMRateLimitError):
                raise OnyxError(OnyxErrorCode.RATE_LIMITED, message)
            if isinstance(e, LLMTimeoutError):
                raise OnyxError(OnyxErrorCode.GATEWAY_TIMEOUT, message)
            raise OnyxError(OnyxErrorCode.LLM_PROVIDER_ERROR, message)
        record_llm_response(span, response)

    wire: dict[str, Any] = ChatCompletionResponse.from_model_response(
        response, model=chat_request.model
    ).to_wire()
    return JSONResponse(content=wire)
