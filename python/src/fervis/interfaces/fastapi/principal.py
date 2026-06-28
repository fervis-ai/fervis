"""FastAPI request principal extraction for Fervis interfaces."""

from __future__ import annotations

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.interfaces.common.questions import InterfacePrincipal
from fervis.interfaces.common.read_contexts import (
    read_context_ref_from_dependency_principal,
    validate_read_context_ref,
)


def principal_from_request(
    request,
    *,
    read_context_capture=None,
    delegated_credential_capture=None,
    dependency_principal=None,
    principal_id_attr: str = "id",
    require_read_context: bool = False,
) -> InterfacePrincipal:
    read_context_ref = _read_context_ref_from_request(
        request,
        read_context_capture=read_context_capture,
        dependency_principal=dependency_principal,
        principal_id_attr=principal_id_attr,
        require_read_context=require_read_context,
    )
    principal_id = str(read_context_ref.key or "anonymous")
    return InterfacePrincipal(
        principal_id=principal_id,
        tenant_id="default",
        raw=request,
        read_context_ref=read_context_ref,
        delegated_credential=(
            None
            if delegated_credential_capture is None
            else delegated_credential_capture(request)
        ),
    )


def _read_context_ref_from_request(
    request,
    *,
    read_context_capture,
    dependency_principal,
    principal_id_attr: str,
    require_read_context: bool,
) -> ReadContextRef:
    if dependency_principal is not None:
        return validate_read_context_ref(
            read_context_ref_from_dependency_principal(
                dependency_principal,
                principal_id_attr=principal_id_attr,
            ),
            require_read_context=require_read_context,
        )
    if read_context_capture is None:
        return validate_read_context_ref(
            ReadContextRef(scheme="anonymous"),
            require_read_context=require_read_context,
        )
    captured = read_context_capture(request)
    if isinstance(captured, ReadContextRef):
        read_context_ref = captured
    else:
        read_context_ref = ReadContextRef.from_storage_dict(captured)
    return validate_read_context_ref(
        read_context_ref,
        require_read_context=require_read_context,
    )
