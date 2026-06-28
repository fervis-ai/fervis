"""Read-context capture helpers shared by framework interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from fervis.host_api.contracts.authority import ReadContextRef


@dataclass(frozen=True)
class ReadContextCaptureError(PermissionError):
    message: str = "Fervis could not capture an authenticated read context."
    code: str = "read_context_required"


def validate_read_context_ref(
    read_context_ref: ReadContextRef,
    *,
    require_read_context: bool,
) -> ReadContextRef:
    if require_read_context and (
        read_context_ref.scheme == "anonymous" or read_context_ref.key is None
    ):
        raise ReadContextCaptureError()
    return read_context_ref


def read_context_ref_from_dependency_principal(
    principal: Any,
    *,
    principal_id_attr: str,
) -> ReadContextRef:
    if isinstance(principal, ReadContextRef):
        return principal
    key = (
        principal.get(principal_id_attr)
        if isinstance(principal, Mapping)
        else getattr(principal, principal_id_attr, None)
    )
    return ReadContextRef(
        scheme="fastapi_principal",
        key=key,
    )
