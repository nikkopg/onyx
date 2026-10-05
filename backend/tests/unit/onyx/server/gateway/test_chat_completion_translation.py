import pytest

from onyx.error_handling.exceptions import OnyxError
from onyx.llm.model_request import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from onyx.llm.models import (
    ImageContentPart,
    NamedToolChoice,
    ReasoningEffort,
    TextContentPart,
    ToolChoiceOptions,
)
from onyx.server.gateway.chat_completion_translation import (
    translate_chat_completion_request,
    translate_messages,
    translate_reasoning_effort,
    translate_tool_choice,
)
from onyx.server.gateway.models import ChatCompletionRequest


def test_translates_a_tool_call_round_trip() -> None:
    prompt = translate_messages(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "developer", "content": [{"type": "text", "text": "Use tools."}]},
            {"role": "user", "content": "List the cycles."},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should call the tool.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "plane-1_cycle",
                            "arguments": '{"action": "list"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
        ]
    )

    assert prompt[0] == SystemMessage(content="Be brief.")
    assert prompt[1] == SystemMessage(content="Use tools.")
    assert prompt[2] == UserMessage(content="List the cycles.")
    assistant = prompt[3]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.content is None
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].id == "call_1"
    assert assistant.tool_calls[0].function.name == "plane-1_cycle"
    assert assistant.tool_calls[0].function.arguments == '{"action": "list"}'
    assert prompt[4] == ToolMessage(content="[]", tool_call_id="call_1")


def test_tool_call_arguments_object_is_serialized() -> None:
    prompt = translate_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c", "function": {"name": "t", "arguments": {"a": 1}}}
                ],
            }
        ]
    )
    assistant = prompt[0]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].function.arguments == '{"a": 1}'


def test_user_content_parts_keep_images() -> None:
    prompt = translate_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA"},
                    },
                ],
            }
        ]
    )
    user = prompt[0]
    assert isinstance(user, UserMessage)
    assert isinstance(user.content, list)
    assert user.content[0] == TextContentPart(text="What is this?")
    assert isinstance(user.content[1], ImageContentPart)
    assert user.content[1].image_url.url == "data:image/png;base64,AA"


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "robot", "content": "hi"}],
        [{"role": "tool", "content": "x"}],
        [{"role": "system", "content": [{"type": "image_url", "image_url": "u"}]}],
    ],
)
def test_invalid_messages_are_rejected(messages: list[dict]) -> None:
    with pytest.raises(OnyxError):
        translate_messages(messages)


def test_tool_choice() -> None:
    assert translate_tool_choice(None) is None
    assert translate_tool_choice("required") == ToolChoiceOptions.REQUIRED
    assert translate_tool_choice(
        {"type": "function", "function": {"name": "search"}}
    ) == NamedToolChoice(name="search")
    with pytest.raises(OnyxError):
        translate_tool_choice("sometimes")


def test_reasoning_effort() -> None:
    assert translate_reasoning_effort(None) == ReasoningEffort.AUTO
    assert translate_reasoning_effort("none") == ReasoningEffort.OFF
    assert translate_reasoning_effort("minimal") == ReasoningEffort.LOW
    assert translate_reasoning_effort("HIGH") == ReasoningEffort.HIGH
    with pytest.raises(OnyxError):
        translate_reasoning_effort("max")


def test_request_options_are_carried_over() -> None:
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    call_args = translate_chat_completion_request(
        ChatCompletionRequest(
            model="1/gemma4:31b-cloud",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            max_tokens=100,
            max_completion_tokens=50,
            reasoning_effort="medium",
            reasoningSummary="auto",  # unknown opencode field, accepted and ignored
        )
    )
    assert call_args.tools == tools
    assert call_args.max_tokens == 50
    assert call_args.reasoning_effort == ReasoningEffort.MEDIUM
