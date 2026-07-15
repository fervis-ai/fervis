from __future__ import annotations

import importlib.util

import pytest

from fervis import FervisConfig, HostConfig, ModelConfig, RuntimeRoutes
from fervis.integrations import (
    create_framework_integration,
    framework_integration_spec,
)


@pytest.mark.parametrize("framework", ["django", "fastapi", "flask"])
def test_builtin_framework_integrations_resolve_through_registry(
    framework: str,
) -> None:
    spec = framework_integration_spec(framework)

    assert spec.framework == framework
    assert spec.integration_type.__module__.startswith(
        f"fervis.integrations.{framework}."
    )


def test_registry_constructs_framework_integration() -> None:
    config = FervisConfig(
        host=HostConfig(timezone="UTC"),
        routes=RuntimeRoutes(prefix="/facts/"),
        model=ModelConfig(default_provider="test", default_model_key="test"),
        sources=[],
    )

    integration = create_framework_integration("django", config=config)

    assert integration.framework == "django"
    assert integration.config is config
    assert integration.routes.prefix == "/facts/"


def test_registry_rejects_unknown_framework() -> None:
    with pytest.raises(
        LookupError, match="Unsupported framework integration 'unknown'"
    ):
        framework_integration_spec("unknown")


@pytest.mark.parametrize("framework", ["django", "fastapi", "flask"])
def test_framework_integrations_do_not_occupy_top_level_packages(
    framework: str,
) -> None:
    assert importlib.util.find_spec(f"fervis.{framework}") is None
