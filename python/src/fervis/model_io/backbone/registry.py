"""Provider adapter registry used by the provider backbone factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .contracts import (
    HitlRuntime,
    HooksRuntime,
    LoopRuntime,
    ProviderModelAdapter,
    SessionRuntime,
    StreamingRuntime,
    TraceRuntime,
)
from .dto import ToolSpec


def _identity_tool_specs(
    tool_specs: tuple[ToolSpec, ...],
) -> tuple[Any, ...]:
    return tool_specs


@dataclass(frozen=True)
class ProviderRegistration:
    name: str
    model_adapter: ProviderModelAdapter
    loop_runtime: LoopRuntime
    stream_runtime: StreamingRuntime
    session_runtime: SessionRuntime
    hitl_runtime: HitlRuntime
    hooks_runtime: HooksRuntime
    trace_runtime: TraceRuntime
    budget_tool_specs: Callable[[tuple[ToolSpec, ...]], tuple[Any, ...]] = (
        _identity_tool_specs
    )


_REGISTRY: dict[str, ProviderRegistration] = {}


def register_provider(registration: ProviderRegistration) -> None:
    _REGISTRY[registration.name] = registration


def get_provider(name: str) -> ProviderRegistration:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY.keys())) or "none"
        raise KeyError(
            f"Unknown provider '{name}'. Registered providers: {available}"
        ) from exc


def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


def registrations() -> dict[str, ProviderRegistration]:
    return dict(_REGISTRY)


def reset_registry_for_tests() -> None:
    _REGISTRY.clear()
