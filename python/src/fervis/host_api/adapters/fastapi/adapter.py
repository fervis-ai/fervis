"""FastAPI framework adapter for configured endpoint access."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import RLock
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
from fervis.project.importing import import_object, project_import_context

from . import catalog
from .executor import FastAPIApplicationRuntime, FastAPIDependencyOverride
from .loading import load_fastapi_app


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
        self._load_lock = RLock()
        self._loaded = False
        self._closed = False
        self._dependency: Callable[..., object] | None = None
        self._resolver: Callable[..., object] | None = None
        self._contracts: tuple[EndpointContract, ...] = ()
        self._runtimes: list[FastAPIApplicationRuntime] = []
        self._runtimes_by_endpoint: dict[str, FastAPIApplicationRuntime] = {}

    def _load(self) -> None:
        with self._load_lock:
            if self._closed:
                raise EndpointExecutionError("FastAPI host adapter is closed.")
            if self._loaded:
                return
            self._load_applications()
            self._loaded = True

    def _load_applications(self) -> None:
        bindings: list[tuple[EndpointContract, FastAPIApplicationRuntime]] = []
        runtimes: list[FastAPIApplicationRuntime] = []
        with project_import_context(self.project_root):
            if http_read_config_from_auth_schema(self.auth_schema) is None:
                dependency_functions = _dependency_functions(self.auth_schema)
            else:
                dependency_functions = (None, None)
            self._dependency, self._resolver = dependency_functions
            for source in self.sources:
                for import_path in source.import_paths:
                    app = load_fastapi_app(import_path)
                    runtime = FastAPIApplicationRuntime(
                        app,
                        project_root=self.project_root,
                    )
                    app_contracts = catalog.endpoint_contracts_from_fastapi_app(
                        app,
                        source=source,
                        import_path=import_path,
                    )
                    runtimes.append(runtime)
                    for contract in app_contracts:
                        bindings.append((contract, runtime))
        sorted_bindings = tuple(
            sorted(bindings, key=lambda binding: binding[0].path_template)
        )
        runtimes_by_endpoint: dict[str, FastAPIApplicationRuntime] = {}
        for contract, runtime in sorted_bindings:
            runtimes_by_endpoint.setdefault(contract.endpoint_name, runtime)
        self._contracts = tuple(contract for contract, _ in sorted_bindings)
        self._runtimes = runtimes
        self._runtimes_by_endpoint = runtimes_by_endpoint

    def describe_sources(self) -> tuple[EndpointContract, ...]:
        self._load()
        return self._contracts

    def close(self) -> None:
        with self._load_lock:
            if self._closed:
                return
            self._closed = True
            for runtime in reversed(self._runtimes):
                runtime.close()

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
        self._load()
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
        contract = self._endpoint_contract(invocation.endpoint_name)
        runtime = self._runtimes_by_endpoint.get(invocation.endpoint_name)
        if runtime is None:
            raise EndpointExecutionError(
                f"Unknown endpoint: {invocation.endpoint_name}"
            )
        return runtime.execute_get(
            contract=contract,
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
        if self._dependency is None or self._resolver is None:
            raise RuntimeError("FastAPI principal functions were not configured.")
        return FastAPIDependencyOverride(
            dependency=self._dependency,
            resolver=self._resolver,
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


def _dependency_functions(
    schema: dict[str, object] | None,
) -> tuple[Callable[..., object] | None, Callable[..., object] | None]:
    principal = _principal_schema(schema)
    if not principal:
        return None, None
    dependency = import_object(str(principal["dependency"]))
    resolver = import_object(str(principal["resolver"]))
    if not callable(dependency) or not callable(resolver):
        raise TypeError("FastAPI principal dependency and resolver must be callable.")
    return dependency, resolver
