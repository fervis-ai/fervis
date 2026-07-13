"""Authentication checks for Fervis Django interface endpoints."""

from __future__ import annotations

from django.conf import settings
from django.utils.module_loading import import_string


class MissingAuthenticationError(PermissionError):
    """Raised when caller is not an authenticated host subject."""


def require_authenticated(request) -> None:
    user = getattr(request, "user", None)
    if user is None or not bool(getattr(user, "is_authenticated", False)):
        raise MissingAuthenticationError("Authenticated host read context required.")


def require_fervis_access(request) -> None:
    require_authenticated(request)
    check = _fervis_access_check()
    if check is None:
        return
    if not bool(check(request)):
        raise MissingAuthenticationError("Fervis fervis access denied by host.")


def _fervis_access_check():
    value = getattr(settings, "FERVIS_ACCESS_CHECK", None)
    if value is None:
        return None
    if isinstance(value, str):
        value = import_string(value)
    if not callable(value):
        raise MissingAuthenticationError("FERVIS_ACCESS_CHECK must be callable.")
    return value
