from fervis.model_io.backbone.factory import (
    build_provider_backbone,
    reset_provider_backbone_for_tests,
)
from fervis.model_io.backbone.dto import ProviderOutputMode
from fervis.model_io.backbone.dto import ProviderRunRequest
from fervis.model_io.backbone.model_routing import resolve_model_route
from fervis.model_io.backbone.registry import list_providers
from fervis.model_io.models import ModelRef
from fervis.model_io.pricing import ModelPricing
from fervis.model_io.providers.specs import (
    required_strict_tool_providers,
    supported_provider_specs,
)
from fervis.model_io.providers.anthropic_adapter import build_anthropic_registration
from fervis.model_io.providers.openai_compatible_adapter import (
    OPENAI_COMPATIBLE_PROVIDER_CONFIGS,
    build_openai_compatible_registration,
)
from fervis.model_io.providers.chat_runtime import (
    ChatProviderConfig,
    ConfiguredChatLoopRuntime,
    ConfiguredChatModelAdapter,
)
from fervis.model_io.routing import ModelRouter


class _CapturingAdapter:
    def __init__(self):
        self.model_ids = []

    def generate(
        self,
        *,
        model_id=None,
        prompt,
        max_thinking_tokens,
        system_prompt="",
        output_mode=ProviderOutputMode.TEXT,
        tool_specs=(),
    ):
        del prompt, max_thinking_tokens, system_prompt, output_mode, tool_specs
        self.model_ids.append(model_id)
        return {
            "answer": "ok",
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
                "costSource": "provider_usage_unpriced",
            },
        }


class _StaticChatLoopRuntime(ConfiguredChatLoopRuntime):
    def sdk_available(self) -> bool:
        return True

    def worker(self):
        def _worker(payload, result_queue):
            del payload
            result_queue.put(
                {
                    "ok": True,
                    "answer": "ok",
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                }
            )

        return _worker

    def request_payload(self, request: ProviderRunRequest):
        return {"model": request.model_id or self.config.model_name}


def test_model_routes_include_fervis_models(fervis_foundation_reset):
    expected = {
        "HAIKU": ("anthropic", "claude-haiku-4-5-20251001"),
        "GPT_5_4_MINI": ("openai", "gpt-5.4-mini"),
    }

    assert {
        model_key: (
            resolve_model_route(model_key).provider,
            resolve_model_route(model_key).model_id,
        )
        for model_key in expected
    } == expected


def test_unknown_explicit_model_key_fails_fast(fervis_foundation_reset):
    try:
        resolve_model_route("GPT")
    except ValueError as exc:
        assert "Unknown Fervis model key: GPT" in str(exc)
    else:
        raise AssertionError("Unknown model keys must be rejected.")


def test_provider_backbone_bootstraps_fervis_model_providers(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()

    build_provider_backbone()

    assert set(list_providers()) >= {
        "anthropic",
        "baseten",
        "fireworks",
        "opencode",
        "openai",
    }


def test_provider_resolution_uses_requested_model_key(fervis_foundation_reset):
    reset_provider_backbone_for_tests()

    backbone = build_provider_backbone()

    assert {
        model_key: backbone.resolve_provider(model_key=model_key)
        for model_key in (
            "GPT_5_4_MINI",
            "fireworks:accounts/fireworks/models/kimi-k2-instruct-0905",
            "baseten:deepseek-ai/DeepSeek-V4-Pro",
            "opencode:deepseek-v4-pro",
        )
    } == {
        "GPT_5_4_MINI": "openai",
        "fireworks:accounts/fireworks/models/kimi-k2-instruct-0905": "fireworks",
        "baseten:deepseek-ai/DeepSeek-V4-Pro": "baseten",
        "opencode:deepseek-v4-pro": "opencode",
    }


def test_provider_resolution_uses_requested_provider_without_model_key(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()

    backbone = build_provider_backbone()

    assert backbone.resolve_provider("openai") == "openai"


def test_model_router_overrides_model_only_for_literal_model_refs():
    adapter = _CapturingAdapter()
    router = ModelRouter(adapters={"openai": adapter})

    router.generate(
        provider="openai",
        model_key="GPT_5_4_MINI",
        prompt="prompt",
        max_thinking_tokens=64,
        system_prompt="system",
    )
    router.generate(
        provider="openai",
        model_key="openai:gpt-5.4-mini",
        prompt="prompt",
        max_thinking_tokens=64,
        system_prompt="system",
    )

    assert adapter.model_ids == [None, "gpt-5.4-mini"]


def test_chat_runtime_raw_metadata_uses_requested_model_id(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    captured_pricing_keys = []
    from fervis.model_io.providers import chat_runtime

    def pricing_for_request(*, provider, model_key):
        captured_pricing_keys.append((provider, model_key))
        return ModelPricing(
            input_cost_per_million_tokens=1,
            output_cost_per_million_tokens=1,
            thinking_cost_per_million_tokens=1,
            pricing_version="models.dev:test/override-model",
            cost_source="models_dev",
        )

    monkeypatch.setattr(
        chat_runtime,
        "resolve_model_pricing",
        pricing_for_request,
    )
    runtime = _StaticChatLoopRuntime(
        config=ChatProviderConfig(
            provider_name="test",
            model_name="default-model",
            api_key_env_var="TEST_API_KEY",
            sdk_name="test_sdk",
        )
    )

    result = runtime.run(
        ProviderRunRequest(
            provider="test",
            model_id="override-model",
            prompt="prompt",
            max_thinking_tokens=64,
            system_prompt="system",
        )
    )

    assert result.raw_payload["model"] == "override-model"
    assert captured_pricing_keys == [("test", "override-model")]


def test_provider_registrations_share_model_adapter_boundary(
    fervis_foundation_reset,
):
    assert isinstance(
        build_anthropic_registration().model_adapter,
        ConfiguredChatModelAdapter,
    )

    for config in OPENAI_COMPATIBLE_PROVIDER_CONFIGS:
        assert isinstance(
            build_openai_compatible_registration(config).model_adapter,
            ConfiguredChatModelAdapter,
        )


def test_openai_compatible_specs_use_expected_runtime_config(
    fervis_foundation_reset,
):
    configs = {
        config.provider_name: config for config in OPENAI_COMPATIBLE_PROVIDER_CONFIGS
    }

    assert {
        provider: (
            configs[provider].temperature,
            configs[provider].base_url,
            configs[provider].api_key_env_var,
        )
        for provider in ("openai", "fireworks", "baseten", "opencode")
    } == {
        "openai": (0.0, "https://api.openai.com/v1", "OPENAI_API_KEY"),
        "fireworks": (
            0.0,
            "https://api.fireworks.ai/inference/v1",
            "FIREWORKS_API_KEY",
        ),
        "baseten": (0.0, "https://inference.baseten.co/v1", "BASETEN_API_KEY"),
        "opencode": (0.0, "https://opencode.ai/zen/v1", "OPENCODE_API_KEY"),
    }


def test_openai_uses_max_completion_tokens_parameter(
    fervis_foundation_reset,
):
    configs = {
        config.provider_name: config for config in OPENAI_COMPATIBLE_PROVIDER_CONFIGS
    }

    assert {
        provider: configs[provider].max_output_tokens_parameter
        for provider in ("openai", "fireworks", "baseten")
    } == {
        "openai": "max_completion_tokens",
        "fireworks": "max_tokens",
        "baseten": "max_tokens",
    }


def test_model_ref_parses_literal_provider_model_reference():
    ref = ModelRef.parse("fireworks:accounts/fireworks/models/kimi-k2-instruct-0905")

    assert ref.provider == "fireworks"
    assert ref.model_id == "accounts/fireworks/models/kimi-k2-instruct-0905"
    assert str(ref) == "fireworks:accounts/fireworks/models/kimi-k2-instruct-0905"


def test_supported_provider_specs_are_strict_tool_certified():
    specs = supported_provider_specs()

    assert {
        name: (
            spec.transport,
            spec.api_key_env,
            spec.supports_strict_tools,
            spec.supports_required_tool_choice,
            spec.supports_parallel_tool_disable,
        )
        for name, spec in specs.items()
    } == {
        "anthropic": (
            "anthropic_messages",
            "ANTHROPIC_API_KEY",
            True,
            True,
            True,
        ),
        "baseten": (
            "openai_chat_completions",
            "BASETEN_API_KEY",
            True,
            True,
            True,
        ),
        "fireworks": (
            "openai_chat_completions",
            "FIREWORKS_API_KEY",
            True,
            True,
            True,
        ),
        "opencode": (
            "openai_chat_completions",
            "OPENCODE_API_KEY",
            True,
            True,
            True,
        ),
        "openai": (
            "openai_chat_completions",
            "OPENAI_API_KEY",
            True,
            True,
            True,
        ),
    }
    assert required_strict_tool_providers() == tuple(sorted(specs))
