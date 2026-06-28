"""Flask HostApiAdapter implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fervis.host_api.adapters.http import (
    execute_http_read,
    http_read_config_from_auth_schema,
)
from fervis.host_api.contracts import EndpointContract, ReadContextRef
from fervis.host_api.contracts.authority import ReadAuthority
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.host_api.credentials import (
    credential_overlay_from_auth_schema,
    delegated_credential_from_request,
)
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)
from fervis.project.integration import FlaskAppSource

from .catalog import get_flask_endpoint_contracts
from .executor import execute_get_endpoint
from .principal import capture_flask_read_context, flask_principal_override


class FlaskHostApiAdapter:
    def __init__(
        self,
        *,
        sources: tuple[FlaskAppSource, ...],
        project_root: Path,
        auth_schema: dict[str, object] | None = None,
    ) -> None:
        self.sources = sources
        self.project_root = project_root
        self.auth_schema = auth_schema

    def describe_sources(self) -> tuple[EndpointContract, ...]:
        return get_flask_endpoint_contracts(
            sources=self.sources,
            project_root=self.project_root,
        )

    def capture_read_context(self, request: Any) -> ReadContextRef:
        return capture_flask_read_context(self.auth_schema, request=request)

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
            return execute_http_read(
                contract=self._endpoint_contract(invocation.endpoint_name),
                authority=authority,
                invocation=invocation,
                config=http_config,
            )
        principal_override = None
        if authority.read_context_ref.scheme != "anonymous":
            principal_override = flask_principal_override(
                self.auth_schema,
                read_context_ref=authority.read_context_ref,
                tenant_id=authority.tenant_id,
            )
        return execute_get_endpoint(
            endpoint_name=invocation.endpoint_name,
            sources=self.sources,
            project_root=self.project_root,
            path_params=dict(invocation.path_params),
            query_params=dict(invocation.query_params),
            page_policy=(
                None if invocation.page_policy is None else dict(invocation.page_policy)
            ),
            principal_override=principal_override,
            transport_overlay=credential_overlay_from_auth_schema(
                schema=self.auth_schema,
                credential=authority.delegated_credential,
            ),
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
