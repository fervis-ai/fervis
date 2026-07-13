"""Django framework adapter for Fervis endpoint access."""

from __future__ import annotations

from typing import Any

from fervis.host_api.contracts import EndpointContract
from fervis.host_api.adapters.http import (
    execute_http_read,
    http_read_config_from_auth_schema,
)
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.credentials import (
    credential_overlay_from_auth_schema,
    delegated_credential_from_request,
)
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.project.source_scope import DjangoSourceScope

from .catalog import get_endpoint_contracts
from .catalog import get_endpoint_contract
from .principal import capture_django_read_context, resolve_django_read_context_ref
from .executor import execute_get_endpoint


class DjangoHostApiAdapter:
    def __init__(
        self,
        *,
        sources: tuple[DjangoSourceScope, ...],
        auth_schema: dict[str, object] | None = None,
    ):
        self.sources = tuple(sources)
        self.auth_schema = auth_schema

    def close(self) -> None:
        pass

    def describe_sources(self) -> tuple[EndpointContract, ...]:
        return get_endpoint_contracts(sources=self.sources)

    def build_relation_catalog(self) -> RelationCatalog:
        return relation_catalog_from_endpoint_contracts(self.describe_sources())

    def capture_read_context(self, request: Any) -> ReadContextRef:
        return capture_django_read_context(request)

    def capture_delegated_credential(
        self,
        request: Any,
    ) -> DelegatedReadCredential | None:
        return delegated_credential_from_request(
            schema=self.auth_schema,
            request=request,
        )

    def execute_read(
        self,
        *,
        authority: ReadAuthority,
        invocation: ReadInvocation,
    ) -> EndpointExecutionResult:
        http_config = http_read_config_from_auth_schema(self.auth_schema)
        if http_config is not None:
            contract = get_endpoint_contract(
                invocation.endpoint_name,
                sources=self.sources,
            )
            if contract is None:
                raise EndpointExecutionError(
                    f"Unknown endpoint: {invocation.endpoint_name}"
                )
            return execute_http_read(
                contract=contract,
                authority=authority,
                invocation=invocation,
                config=http_config,
            )
        user = resolve_django_read_context_ref(authority.read_context_ref)
        return execute_get_endpoint(
            endpoint_name=invocation.endpoint_name,
            user=user,
            sources=self.sources,
            path_params=dict(invocation.path_params),
            query_params=dict(invocation.query_params),
            page_policy=(
                None if invocation.page_policy is None else dict(invocation.page_policy)
            ),
            transport_overlay=credential_overlay_from_auth_schema(
                schema=self.auth_schema,
                credential=authority.delegated_credential,
            ),
        )
