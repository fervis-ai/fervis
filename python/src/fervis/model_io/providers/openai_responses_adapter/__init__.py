"""OpenAI Responses API provider registration."""

from __future__ import annotations

from fervis.model_io.backbone.registry import (
    ProviderRegistration,
    get_provider,
    register_provider,
)
from fervis.model_io.providers.chat_runtime import (
    ChatProviderConfig,
    ConfiguredChatModelAdapter,
)
from fervis.model_io.providers.openai_compatible_adapter import provider_config_for_spec
from fervis.model_io.providers.shared_runtime import (
    DefaultHitlRuntime,
    DefaultHooksRuntime,
    DefaultStreamingRuntime,
    DefaultTraceRuntime,
)
from fervis.model_io.providers.session_runtime import ProviderSessionRuntime
from fervis.model_io.providers.specs import supported_provider_spec

from .loop_adapter import OpenAIResponsesLoopRuntime


OPENAI_RESPONSES_PROVIDER_CONFIG = provider_config_for_spec(
    supported_provider_spec("openai")
)


def build_openai_responses_registration(
    config: ChatProviderConfig = OPENAI_RESPONSES_PROVIDER_CONFIG,
) -> ProviderRegistration:
    loop_runtime = OpenAIResponsesLoopRuntime(config=config)
    return ProviderRegistration(
        name=config.provider_name,
        model_adapter=ConfiguredChatModelAdapter(
            loop_runtime=loop_runtime,
            config=config,
        ),
        loop_runtime=loop_runtime,
        stream_runtime=DefaultStreamingRuntime(),
        session_runtime=ProviderSessionRuntime(provider_name=config.provider_name),
        hitl_runtime=DefaultHitlRuntime(),
        hooks_runtime=DefaultHooksRuntime(),
        trace_runtime=DefaultTraceRuntime(),
    )


def register_openai_responses_provider() -> None:
    try:
        get_provider(OPENAI_RESPONSES_PROVIDER_CONFIG.provider_name)
    except KeyError:
        register_provider(build_openai_responses_registration())
