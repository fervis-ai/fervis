"""Provider registrations for OpenAI-compatible chat-completions APIs."""

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
from fervis.model_io.providers.specs import (
    OPENAI_CHAT_COMPLETIONS,
    ProviderSpec,
    supported_provider_specs,
)
from fervis.model_io.providers.shared_runtime import (
    DefaultHitlRuntime,
    DefaultHooksRuntime,
    DefaultStreamingRuntime,
    DefaultTraceRuntime,
)
from fervis.model_io.providers.session_runtime import ProviderSessionRuntime

from .loop_adapter import OpenAICompatibleLoopRuntime


def _config_for_spec(spec: ProviderSpec) -> ChatProviderConfig:
    return ChatProviderConfig(
        provider_name=spec.name,
        model_name=spec.default_model,
        api_key_env_var=spec.api_key_env,
        sdk_name=spec.transport,
        base_url_env_var=spec.base_url_env,
        default_base_url=spec.default_base_url,
        temperature=spec.temperature,
        max_output_tokens_parameter=spec.max_output_tokens_parameter,
    )


OPENAI_COMPATIBLE_PROVIDER_CONFIGS = (
    *(
        _config_for_spec(spec)
        for spec in supported_provider_specs().values()
        if spec.transport == OPENAI_CHAT_COMPLETIONS
    ),
)


def build_openai_compatible_registration(
    config: ChatProviderConfig,
) -> ProviderRegistration:
    loop_runtime = OpenAICompatibleLoopRuntime(config=config)
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


def register_openai_compatible_providers() -> None:
    for config in OPENAI_COMPATIBLE_PROVIDER_CONFIGS:
        try:
            get_provider(config.provider_name)
            continue
        except KeyError:
            pass
        register_provider(build_openai_compatible_registration(config))
