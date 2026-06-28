"""FastAPI framework adapter for configured endpoint access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fervis.host_api.adapters.http import (
    execute_http_read,
    http_read_config_from_auth_schema,
)
from fervis.host_api.contracts import EndpointContract
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
from fervis.project.integration import FastAPIAppSource
from fervis.project.importing import project_import_context

from . import catalog
from .executor import FastAPIDependencyOverride, execute_get_endpoint


class FastAPIHostApiAdapter:
    def __init__(
        self,
        *,
        sources: tuple[FastAPIAppSource, ...],
        project_root: Path,
        auth_schema: dict[str, object] | None = None,
    ) -> None:
        self.sources = tuple(sources)
        self.project_root = project_root
        self.auth_schema = auth_schema

    def describe_sources(self) -> tuple[EndpointContract, ...]:
        return catalog.get_fastapi_endpoint_contracts(
            sources=self.sources,
            project_root=self.project_root,
        )

    def capture_read_context(self, request: Any) -> ReadContextRef:
        user = getattr(getattr(request, "state", None), "user", None)
        if user is None:
            return ReadContextRef(scheme="anonymous")
        id_attr = _principal_id_attr(self.auth_schema) or "id"
        return ReadContextRef(
            scheme="fastapi_principal",
            key=str(getattr(user, id_attr, user)),
        )

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
            contract = self._endpoint_contract(invocation.endpoint_name)
            with project_import_context(self.project_root):
                return execute_http_read(
                    contract=contract,
                    authority=authority,
                    invocation=invocation,
                    config=http_config,
                )
        dependency_override: FastAPIDependencyOverride | None = None
        if authority.read_context_ref.scheme != "anonymous":
            dependency_override = self._dependency_override(authority)
        return execute_get_endpoint(
            endpoint_name=invocation.endpoint_name,
            sources=self.sources,
            project_root=self.project_root,
            path_params=dict(invocation.path_params),
            query_params=dict(invocation.query_params),
            page_policy=(
                None if invocation.page_policy is None else dict(invocation.page_policy)
            ),
            dependency_override=dependency_override,
            transport_overlay=credential_overlay_from_auth_schema(
                schema=self.auth_schema,
                credential=authority.delegated_credential,
            ),
        )

    def _dependency_override(
        self,
        authority: ReadAuthority,
    ) -> FastAPIDependencyOverride:
        principal = _principal_schema(self.auth_schema)
        if not principal:
            raise ValueError(
                "FastAPI reads require configured principal reauthorization."
            )
        return FastAPIDependencyOverride(
            dependency=str(principal["dependency"]),
            resolver=str(principal["resolver"]),
            key=authority.read_context_ref.key,
            tenant_id=authority.tenant_id,
            principal_id_attr=str(principal.get("id_attr") or "id"),
        )

    def _endpoint_contract(self, endpoint_name: str) -> EndpointContract:
        contract = next(
            (
                contract
                for contract in self.describe_sources()
                if contract.endpoint_name == endpoint_name
            ),
            None,
        )
        if contract is None:
            raise EndpointExecutionError(f"Unknown endpoint: {endpoint_name}")
        return contract


def _principal_schema(schema: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(schema, dict):
        return {}
    principal = schema.get("principal")
    if not isinstance(principal, dict):
        return {}
    if principal.get("source") != "fastapi_dependency":
        return {}
    return principal


def _principal_id_attr(schema: dict[str, object] | None) -> str:
    principal = _principal_schema(schema)
    return str(principal.get("id_attr") or "")
