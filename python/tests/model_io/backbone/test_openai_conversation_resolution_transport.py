from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    ConversationResolutionTurnPrompt,
)
from fervis.memory.conversation_context import ConversationContextSource
from fervis.model_io.backbone.dto import ProviderOutputMode, ProviderRunRequest
from fervis.model_io.providers.chat_runtime import ChatProviderConfig
from fervis.model_io.providers.openai_compatible_adapter import (
    loop_adapter as openai_loop,
)
from fervis.model_io.providers.openai_compatible_adapter.loop_adapter import (
    OpenAICompatibleLoopRuntime,
)


def test_openai_conversation_resolution_uses_root_object_with_nested_outcome():
    prompt = ConversationResolutionTurnPrompt(
        question="Can you show the shade names too?",
        context_sources=(
            ConversationContextSource(
                source_id="prior_1",
                kind="prior_user_question",
                text="Which product names were sold?",
            ),
        ),
        conversation_context={},
    )
    invocation = prompt.to_model_invocation()
    payload = OpenAICompatibleLoopRuntime(config=_openai_test_config()).request_payload(
        ProviderRunRequest(
            provider="openai",
            prompt=invocation.prompt_text,
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=invocation.tool_specs,
        )
    )

    tools = openai_loop._completion_kwargs(payload)["tools"]
    schema = tools[0]["function"]["parameters"]

    assert [tool["function"]["name"] for tool in tools] == [
        CONVERSATION_RESOLUTION_TOOL_NAME
    ]
    assert schema["type"] == "object"
    assert "oneOf" not in schema
    assert [
        variant["properties"]["kind"]["enum"][0]
        for variant in schema["properties"]["outcome"]["anyOf"]
    ] == ["resolved", "multiple_meanings", "missing_input"]


def test_openai_conversation_resolution_projects_no_retained_parts_as_empty_array():
    prompt = ConversationResolutionTurnPrompt(
        question="How many stores are in the selected area?",
        context_sources=(),
        conversation_context={},
    )
    invocation = prompt.to_model_invocation()
    payload = OpenAICompatibleLoopRuntime(config=_openai_test_config()).request_payload(
        ProviderRunRequest(
            provider="openai",
            prompt=invocation.prompt_text,
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=invocation.tool_specs,
        )
    )

    schema = openai_loop._completion_kwargs(payload)["tools"][0]["function"][
        "parameters"
    ]
    retained_parts = schema["properties"]["outcome"]["anyOf"][0]["properties"][
        "clauses"
    ]["items"]["properties"]["retained_frame_parts"]

    assert {
        "maximum_items": retained_parts["maxItems"],
        "item_type": retained_parts["items"]["type"],
        "item_fields": retained_parts["items"]["required"],
    } == {
        "maximum_items": 0,
        "item_type": "object",
        "item_fields": ["kind", "frame_id", "part_id"],
    }


def _openai_test_config() -> ChatProviderConfig:
    return ChatProviderConfig(
        provider_name="openai",
        model_name="gpt-5.4-mini",
        api_key_env_var="OPENAI_API_KEY",
        sdk_name="openai-compatible-chat-completions",
        default_base_url="https://api.openai.com/v1",
        max_output_tokens_parameter="max_completion_tokens",
    )
