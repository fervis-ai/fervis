"""Django request principal extraction for Fervis interfaces."""

from __future__ import annotations

from fervis.host_api.context import get_host_api_context
from fervis.interfaces.common.questions import InterfacePrincipal
from fervis.interfaces.common.read_contexts import validate_read_context_ref


def tenant_from_request(request) -> str:
    return "default"


def principal_from_request(request) -> InterfacePrincipal:
    tenant_id = tenant_from_request(request)
    adapter = get_host_api_context().adapter
    return InterfacePrincipal(
        principal_id=str(request.user.pk),
        tenant_id=tenant_id,
        raw=request.user,
        read_context_ref=validate_read_context_ref(
            adapter.capture_read_context(request),
            require_read_context=True,
        ),
        delegated_credential=adapter.capture_delegated_credential(request),
    )
