"""Lookup service composition for the Django interface."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from django.conf import settings

from fervis.host_api.context import get_host_api_context
from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.model_io.backbone.factory import (
    ProviderBackbone,
    reset_provider_backbone_for_tests,
)
from fervis.observability.django import DjangoObservabilityQuery
from fervis.questions.contracts import AskRequestLimits

if TYPE_CHECKING:
    from fervis.lookup.orchestration.service import LookupService


_RUNTIME: LookupService | None = None
RUN_CONTEXT_KEY = "_runtimeContext"
_REQUEST_RUNTIME_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar(
    "fervis_request_runtime_context",
    default=None,
)


def get_runtime() -> LookupService:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _new_lookup_service()
    return _RUNTIME


def reset_runtime_for_tests(
    *,
    provider_backbone: ProviderBackbone | None = None,
) -> None:
    global _RUNTIME
    if provider_backbone is None:
        reset_provider_backbone_for_tests()
    _RUNTIME = _new_lookup_service(provider_backbone=provider_backbone)


@contextmanager
def request_runtime_context(runtime_context: dict[str, Any] | None):
    token = _REQUEST_RUNTIME_CONTEXT.set(normalize_runtime_context(runtime_context))
    try:
        yield
    finally:
        _REQUEST_RUNTIME_CONTEXT.reset(token)


def current_request_runtime_context() -> dict[str, str]:
    return dict(_REQUEST_RUNTIME_CONTEXT.get() or {})


def normalize_runtime_context(
    runtime_context: dict[str, Any] | None,
) -> dict[str, str] | None:
    if not runtime_context:
        return None
    normalized = {
        str(key): str(value)
        for key, value in runtime_context.items()
        if value not in (None, "")
    }
    return normalized or None


def runtime_context_from_conversation(
    conversation_context: dict[str, Any],
) -> dict[str, Any]:
    raw = conversation_context.get(RUN_CONTEXT_KEY)
    return dict(raw) if isinstance(raw, dict) else {}


def lookup_conversation_context(
    conversation_context: dict[str, Any],
) -> dict[str, Any]:
    output = dict(conversation_context)
    output.pop(RUN_CONTEXT_KEY, None)
    return output


def question_run_request_limits() -> AskRequestLimits:
    return AskRequestLimits(
        max_budget_usd=getattr(settings, "FERVIS_MAX_REQUEST_BUDGET_USD", 10.0),
        max_thinking_tokens=getattr(settings, "FERVIS_MAX_THINKING_TOKENS", 4096),
    )


def _new_lookup_service(
    *, provider_backbone: ProviderBackbone | None = None
) -> LookupService:
    from fervis.lookup.orchestration.service import LookupService
    from fervis.interfaces.django.question_run_ports import DjangoQuestionLifecyclePort

    return LookupService(
        provider_backbone=provider_backbone,
        provider_name=_settings_provider(),
        host_api_context=get_host_api_context(),
        observability_query=DjangoObservabilityQuery(),
        lineage_recorder=DjangoLineageRecorder(),
        prior_program_invocations=DjangoQuestionLifecyclePort(),
    )


def _settings_provider() -> str | None:
    value = str(getattr(settings, "FERVIS_PROVIDER", "") or "").strip()
    if not value:
        return None
    return value
