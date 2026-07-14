"""Django request principal extraction for Fervis interfaces."""

from __future__ import annotations

from fervis.host_api.context import get_host_api_context
from fervis.host_api.contracts.authority import ReadAuthority
from fervis.interfaces.common.questions import InterfacePrincipal
from fervis.interfaces.common.read_contexts import validate_read_context_ref


def principal_from_request(request) -> InterfacePrincipal:
    adapter = get_host_api_context().adapter
    read_context_ref = validate_read_context_ref(
        adapter.capture_read_context(request),
        require_read_context=True,
    )
    delegated_credential = adapter.capture_delegated_credential(request)
    authority = ReadAuthority.from_read_context(
        read_context_ref,
        delegated_credential=delegated_credential,
    )
    return InterfacePrincipal(
        principal_id=str(read_context_ref.key),
        tenant_id=authority.tenant_id,
        raw=request.user,
        read_context_ref=read_context_ref,
        delegated_credential=delegated_credential,
    )
