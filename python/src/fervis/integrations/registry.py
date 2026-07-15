from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import re
from typing import Any, Protocol, cast

from fervis.project.integration import FervisConfig, RuntimeRoutes


class FrameworkIntegration(Protocol):
    """The framework-neutral surface exposed by a configured integration."""

    @property
    def config(self) -> FervisConfig: ...

    @property
    def framework(self) -> str: ...

    @property
    def routes(self) -> RuntimeRoutes: ...


@dataclass(frozen=True)
class FrameworkIntegrationSpec:
    """Registration for one host-framework integration implementation."""

    framework: str
    integration_type: type[FrameworkIntegration]

    def create(
        self, *, config: FervisConfig, **options: object
    ) -> FrameworkIntegration:
        integration_type = cast(Any, self.integration_type)
        return cast(FrameworkIntegration, integration_type(config=config, **options))


_FRAMEWORK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_REGISTRY: dict[str, FrameworkIntegrationSpec] = {}


def register_framework_integration(
    framework: str,
    integration_type: type[FrameworkIntegration],
) -> FrameworkIntegrationSpec:
    """Register one integration without requiring a central framework switch."""
    if not _FRAMEWORK_NAME.fullmatch(framework):
        raise ValueError(f"Invalid framework integration name {framework!r}.")
    existing = _REGISTRY.get(framework)
    if existing is not None and existing.integration_type is not integration_type:
        raise ValueError(f"Framework integration {framework!r} is already registered.")
    spec = FrameworkIntegrationSpec(
        framework=framework,
        integration_type=integration_type,
    )
    _REGISTRY[framework] = spec
    return spec


def framework_integration_spec(framework: str) -> FrameworkIntegrationSpec:
    """Resolve a built-in or previously registered framework integration."""
    spec = _REGISTRY.get(framework)
    if spec is not None:
        return spec
    if not _FRAMEWORK_NAME.fullmatch(framework):
        raise LookupError(f"Unsupported framework integration {framework!r}.")
    module_name = f"fervis.integrations.{framework}"
    try:
        import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise LookupError(
                f"Unsupported framework integration {framework!r}."
            ) from exc
        raise
    spec = _REGISTRY.get(framework)
    if spec is None:
        raise LookupError(
            f"Framework package {module_name!r} did not register an integration."
        )
    return spec


def create_framework_integration(
    framework: str,
    *,
    config: FervisConfig,
    **options: object,
) -> FrameworkIntegration:
    return framework_integration_spec(framework).create(config=config, **options)
