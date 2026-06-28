from fervis.memory.conversation_context import ConversationContextSource
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    ConversationResolutionTurnPrompt,
)
from fervis.model_io.backbone.dto import ProviderOutputMode, ProviderRunRequest
from fervis.model_io.providers.anthropic_adapter import (
    loop_adapter as anthropic_loop,
)
from fervis.model_io.providers.anthropic_adapter.loop_adapter import (
    AnthropicLoopRuntime,
)
from fervis.model_io.providers.chat_runtime import ChatProviderConfig


def test_anthropic_conversation_resolution_uses_canonical_single_tool():
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
    payload = AnthropicLoopRuntime(config=_anthropic_test_config()).request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt=invocation.prompt_text,
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=invocation.tool_specs,
        )
    )

    tools = anthropic_loop._message_kwargs(payload)["tools"]

    assert [tool["name"] for tool in tools] == [CONVERSATION_RESOLUTION_TOOL_NAME]
    assert "clause_resolutions" in tools[0]["input_schema"]["properties"]


def _anthropic_test_config() -> ChatProviderConfig:
    return ChatProviderConfig(
        provider_name="anthropic",
        model_name="claude-haiku-4-5-20251001",
        api_key_env_var="ANTHROPIC_API_KEY",
        sdk_name="anthropic-messages",
    )
