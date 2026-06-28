"""Resolve Django principals for host API execution."""

from __future__ import annotations

from typing import Any

from django.contrib.auth import get_user_model

from fervis.host_api.contracts.authority import ReadContextRef


def capture_django_read_context(request: Any) -> ReadContextRef:
    user = getattr(request, "user", None)
    if user is None or not getattr(user, "is_authenticated", True):
        return ReadContextRef(scheme="anonymous")
    return django_read_context_ref(user)


def django_read_context_ref(principal: Any) -> ReadContextRef:
    key = getattr(principal, "pk", None) or principal
    return ReadContextRef(scheme="django_principal", key=str(key))


def resolve_django_read_context_ref(read_context_ref: ReadContextRef) -> Any:
    if read_context_ref.scheme == "django_principal" and read_context_ref.key:
        return resolve_django_execution_user(read_context_ref.key)
    raise ValueError(
        f"Unsupported Django read context reference: {read_context_ref.scheme}"
    )


def resolve_django_execution_user(principal: Any) -> Any:
    user_model = get_user_model()
    if isinstance(principal, user_model):
        return principal
    return user_model._default_manager.get(pk=str(principal))
