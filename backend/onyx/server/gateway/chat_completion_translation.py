"""Translate an OpenAI Chat Completions request into raw LLM call arguments.

Pure functions only; the route in ``onyx/server/gateway/api.py`` does auth,
model resolution and streaming.
"""

import json
from dataclasses import dataclass
from typing import Any

from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.model_request import (
    AssistantMessage,
    ChatCompletionMessage,
    RequestFunctionCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from onyx.llm.models import (
    ContentPart,
    ImageContentPart,
    ImageUrlDetail,
    NamedToolChoice,
    ReasoningEffort,
    TextContentPart,
    ToolChoice,
    ToolChoiceOptions,
)
from onyx.server.gateway.models import ChatCompletionRequest

# OpenAI effort names that have no ReasoningEffort member of the same name.
_REASONING_EFFORT_ALIASES: dict[str, ReasoningEffort] = {
    "none": ReasoningEffort.OFF,
    "minimal": ReasoningEffort.LOW,
}


@dataclass(frozen=True)
class RawLLMCallArgs:
    prompt: list[ChatCompletionMessage]
    tools: list[dict[str, Any]] | None
    tool_choice: ToolChoice | None
    structured_response_format: dict[str, Any] | None
    max_tokens: int | None
    reasoning_effort: ReasoningEffort


def _invalid(detail: str) -> OnyxError:
    return OnyxError(OnyxErrorCode.INVALID_INPUT, detail)


def _text_from_content(content: Any, role: str) -> str:
    """Flatten string or text-part content; non-text parts are rejected."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _invalid(f"Unsupported content for a {role} message.")
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            raise _invalid(f"A {role} message may only contain text parts.")
        texts.append(str(part.get("text", "")))
    return "".join(texts)


def _user_content(content: Any) -> str | list[ContentPart]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _invalid("Unsupported content for a user message.")
    parts: list[ContentPart] = []
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type == "text":
            parts.append(TextContentPart(text=str(part.get("text", ""))))
        elif part_type == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) or not url:
                raise _invalid("An image_url part needs a url.")
            detail = image_url.get("detail") if isinstance(image_url, dict) else None
            parts.append(
                ImageContentPart(image_url=ImageUrlDetail(url=url, detail=detail))
            )
        else:
            raise _invalid(f"Unsupported user content part type: {part_type}.")
    return parts


def _tool_calls(raw_tool_calls: Any) -> list[ToolCall] | None:
    if not raw_tool_calls:
        return None
    if not isinstance(raw_tool_calls, list):
        raise _invalid("tool_calls must be a list.")
    tool_calls: list[ToolCall] = []
    for raw in raw_tool_calls:
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict) or not raw.get("id"):
            raise _invalid("Each tool call needs an id and a function.")
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_calls.append(
            ToolCall(
                id=str(raw["id"]),
                function=RequestFunctionCall(
                    name=str(function.get("name", "")), arguments=arguments
                ),
            )
        )
    return tool_calls


def translate_messages(messages: list[dict[str, Any]]) -> list[ChatCompletionMessage]:
    """Prior assistant reasoning is dropped: providers on this path do not
    need it echoed back."""
    if not messages:
        raise _invalid("messages must not be empty.")
    prompt: list[ChatCompletionMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            prompt.append(SystemMessage(content=_text_from_content(content, role)))
        elif role == "user":
            prompt.append(UserMessage(content=_user_content(content)))
        elif role == "assistant":
            text = _text_from_content(content, role)
            prompt.append(
                AssistantMessage(
                    content=text or None,
                    tool_calls=_tool_calls(message.get("tool_calls")),
                )
            )
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                raise _invalid("A tool message needs a tool_call_id.")
            prompt.append(
                ToolMessage(
                    content=_text_from_content(content, role),
                    tool_call_id=str(tool_call_id),
                )
            )
        else:
            raise _invalid(f"Unsupported message role: {role}.")
    return prompt


def translate_tool_choice(tool_choice: Any) -> ToolChoice | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        try:
            return ToolChoiceOptions(tool_choice)
        except ValueError:
            raise _invalid(f"Unsupported tool_choice: {tool_choice}.")
    function = tool_choice.get("function") if isinstance(tool_choice, dict) else None
    if isinstance(function, dict) and function.get("name"):
        return NamedToolChoice(name=str(function["name"]))
    raise _invalid("Unsupported tool_choice.")


def translate_reasoning_effort(reasoning_effort: str | None) -> ReasoningEffort:
    if reasoning_effort is None:
        return ReasoningEffort.AUTO
    normalized = reasoning_effort.lower()
    if normalized in _REASONING_EFFORT_ALIASES:
        return _REASONING_EFFORT_ALIASES[normalized]
    try:
        return ReasoningEffort(normalized)
    except ValueError:
        raise _invalid(f"Unsupported reasoning_effort: {reasoning_effort}.")


def translate_chat_completion_request(
    request: ChatCompletionRequest,
) -> RawLLMCallArgs:
    return RawLLMCallArgs(
        prompt=translate_messages(request.messages),
        tools=request.tools or None,
        tool_choice=translate_tool_choice(request.tool_choice),
        structured_response_format=request.response_format,
        max_tokens=request.max_completion_tokens or request.max_tokens,
        reasoning_effort=translate_reasoning_effort(request.reasoning_effort),
    )
