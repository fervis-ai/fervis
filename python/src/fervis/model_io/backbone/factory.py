"""Factory for provider-portable runtime composition."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from fervis.model_io.providers.bootstrap import (
    bootstrap_default_providers,
    reset_provider_bootstrap_for_tests,
)
from fervis.model_io.routing.router import ModelRouter

from .dto import TraceEvent
from .model_routing import resolve_model_route
from .registry import ProviderRegistration, get_provider, registrations


@dataclass
class ProviderBackbone:
    provider_name: str
    registration: ProviderRegistration
    model_router: ModelRouter

    def resolve_provider(
        self, requested_provider: str | None = None, *, model_key: str | None = None
    ) -> str:
        if model_key:
            return resolve_model_route(model_key).provider
        if requested_provider:
            return str(requested_provider).strip()
        return self.provider_name

    def build_hooks(self) -> list[Any]:
        return self.registration.hooks_runtime.build_hooks()

    def trace(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        self.registration.trace_runtime.record(
            TraceEvent(
                event_type=event_type,
                payload=dict(payload),
                correlation_id=correlation_id,
            )
        )


def build_provider_backbone(provider_name: str | None = None) -> ProviderBackbone:
    bootstrap_default_providers()

    configured_provider = provider_name or os.getenv("FERVIS_PROVIDER")
    if not configured_provider:
        configured_provider = resolve_model_route(None).provider
    default_provider = str(configured_provider or "").strip()
    if not default_provider:
        raise ValueError("Fervis provider resolution returned an empty provider.")

    registration = get_provider(default_provider)

    adapters = {name: item.model_adapter for name, item in registrations().items()}
    model_router = ModelRouter(adapters=adapters)

    return ProviderBackbone(
        provider_name=default_provider,
        registration=registration,
        model_router=model_router,
    )


def build_test_provider_backbone(
    *,
    adapters: dict[str, Any],
    provider_name: str = "anthropic",
) -> ProviderBackbone:
    bootstrap_default_providers()
    registration = get_provider(provider_name)
    return ProviderBackbone(
        provider_name=provider_name,
        registration=registration,
        model_router=ModelRouter(adapters=dict(adapters)),
    )


def reset_provider_backbone_for_tests() -> None:
    reset_provider_bootstrap_for_tests()
    from .registry import reset_registry_for_tests

    reset_registry_for_tests()
