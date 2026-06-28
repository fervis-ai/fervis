"""Declarative model-provider support matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    transport: str
    api_key_env: str
    base_url_env: str | None = None
    default_base_url: str | None = None
    default_model: str = ""
    supports_strict_tools: bool = True
    supports_required_tool_choice: bool = True
    supports_parallel_tool_disable: bool = True
    max_output_tokens_parameter: str = "max_tokens"
    temperature: float = 0.0

    @property
    def strict_tool_certified(self) -> bool:
        return (
            self.supports_strict_tools
            and self.supports_required_tool_choice
            and self.supports_parallel_tool_disable
        )


OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
ANTHROPIC_MESSAGES = "anthropic_messages"


_PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="anthropic",
        transport=ANTHROPIC_MESSAGES,
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-haiku-4-5-20251001",
    ),
    ProviderSpec(
        name="baseten",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env="BASETEN_API_KEY",
        base_url_env="BASETEN_BASE_URL",
        default_base_url="https://inference.baseten.co/v1",
        default_model="deepseek-ai/DeepSeek-V4-Pro",
    ),
    ProviderSpec(
        name="fireworks",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env="FIREWORKS_API_KEY",
        base_url_env="FIREWORKS_BASE_URL",
        default_base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/kimi-k2-instruct-0905",
    ),
    ProviderSpec(
        name="opencode",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env="OPENCODE_API_KEY",
        base_url_env="OPENCODE_BASE_URL",
        default_base_url="https://opencode.ai/zen/v1",
        default_model="deepseek-v4-pro",
    ),
    ProviderSpec(
        name="openai",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-5.4-mini",
        max_output_tokens_parameter="max_completion_tokens",
    ),
)


def supported_provider_specs() -> dict[str, ProviderSpec]:
    return {spec.name: spec for spec in _PROVIDER_SPECS}


def supported_provider_spec(name: str) -> ProviderSpec:
    specs = supported_provider_specs()
    try:
        return specs[name]
    except KeyError as exc:
        supported = ", ".join(sorted(specs))
        raise ValueError(
            f"Unsupported Fervis provider {name!r}. Supported: {supported}"
        ) from exc


def required_strict_tool_providers() -> tuple[str, ...]:
    specs = supported_provider_specs()
    return tuple(
        sorted(name for name, spec in specs.items() if spec.strict_tool_certified)
    )
