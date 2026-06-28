"""Anthropic provider registration for the Fervis provider backbone."""

from __future__ import annotations

from fervis.model_io.backbone.registry import (
    ProviderRegistration,
    get_provider,
    register_provider,
)
from fervis.model_io.providers.specs import supported_provider_specs
from fervis.model_io.providers.shared_runtime import (
    DefaultHitlRuntime,
    DefaultHooksRuntime,
    DefaultStreamingRuntime,
    DefaultTraceRuntime,
)
from fervis.model_io.providers.chat_runtime import (
    ChatProviderConfig,
    ConfiguredChatModelAdapter,
)
from fervis.model_io.providers.session_runtime import ProviderSessionRuntime

from .loop_adapter import AnthropicLoopRuntime, anthropic_tool_specs_for_budget


def build_anthropic_registration() -> ProviderRegistration:
    spec = supported_provider_specs()["anthropic"]
    config = ChatProviderConfig(
        provider_name=spec.name,
        model_name=spec.default_model,
        api_key_env_var=spec.api_key_env,
        sdk_name=spec.transport,
    )
    loop_runtime = AnthropicLoopRuntime(config=config)
    return ProviderRegistration(
        name="anthropic",
        model_adapter=ConfiguredChatModelAdapter(
            loop_runtime=loop_runtime,
            config=config,
        ),
        loop_runtime=loop_runtime,
        stream_runtime=DefaultStreamingRuntime(),
        session_runtime=ProviderSessionRuntime(provider_name="anthropic"),
        hitl_runtime=DefaultHitlRuntime(),
        hooks_runtime=DefaultHooksRuntime(),
        trace_runtime=DefaultTraceRuntime(),
        budget_tool_specs=anthropic_tool_specs_for_budget,
    )


def register_anthropic_provider() -> None:
    try:
        get_provider("anthropic")
        return
    except KeyError:
        pass
    register_provider(build_anthropic_registration())
