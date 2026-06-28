from fervis.model_io.backbone.factory import (
    build_provider_backbone,
    reset_provider_backbone_for_tests,
)
from fervis.model_io.backbone.registry import (
    ProviderRegistration,
    list_providers,
    register_provider,
)

from tests.model_io.backbone.test_adapter_contracts import (
    StubHitlRuntime,
    StubHooksRuntime,
    StubLoopRuntime,
    StubModelAdapter,
    StubSessionRuntime,
    StubStreamRuntime,
    StubTraceRuntime,
)


def test_provider_registry_selects_adapter_from_explicit_provider(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()
    loop = StubLoopRuntime()
    register_provider(
        ProviderRegistration(
            name="stub",
            model_adapter=StubModelAdapter(loop),
            loop_runtime=loop,
            stream_runtime=StubStreamRuntime(),
            session_runtime=StubSessionRuntime(),
            hitl_runtime=StubHitlRuntime(),
            hooks_runtime=StubHooksRuntime(),
            trace_runtime=StubTraceRuntime(),
        )
    )

    backbone = build_provider_backbone("stub")

    assert backbone.provider_name == "stub"


def test_provider_registry_bootstraps_anthropic_provider(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()

    backbone = build_provider_backbone("anthropic")

    assert backbone.provider_name == "anthropic"


def test_provider_registry_lists_bootstrapped_anthropic_provider(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()

    build_provider_backbone("anthropic")

    assert "anthropic" in list_providers()


def test_provider_registry_registers_openai_provider(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()

    build_provider_backbone("anthropic")

    assert "openai" in list_providers()


def test_provider_resolution_routes_model_key_to_provider(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()

    backbone = build_provider_backbone()

    assert backbone.resolve_provider(model_key="HAIKU") == "anthropic"


def test_explicit_provider_does_not_override_model_key_routing(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()

    backbone = build_provider_backbone()

    assert backbone.resolve_provider("openai", model_key="HAIKU") == "anthropic"
