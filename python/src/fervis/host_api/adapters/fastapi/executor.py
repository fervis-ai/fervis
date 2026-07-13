"""Execute configured FastAPI GET endpoint contracts in-process."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from fervis.host_api.adapters.get_execution import (
    PreparedGet,
    execute_prepared_get_async,
    prepare_get_endpoint,
)
from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)
from fervis.project.importing import project_import_context

from ..response_body import response_body
from ..runtime_output import suppress_host_output


@dataclass(frozen=True)
class FastAPIDependencyOverride:
    dependency: Callable[..., object]
    resolver: Callable[..., object]
    key: str | None
    tenant_id: str | None = None
    principal_id_attr: str = "id"


class FastAPIApplicationRuntime:
    """One FastAPI application, lifespan, event loop, and request client."""

    def __init__(self, app: FastAPI, *, project_root: Path) -> None:
        self._app = app
        self._project_root = project_root
        self._lock = RLock()
        self._resources: AsyncExitStack | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._client: AsyncClient | None = None
        self._closed = False

    def execute_get(
        self,
        *,
        contract: EndpointContract,
        path_params: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        page_policy: dict[str, Any] | None = None,
        dependency_override: FastAPIDependencyOverride | None = None,
        transport_overlay: ReadTransportOverlay | None = None,
    ) -> EndpointExecutionResult:
        with self._lock, project_import_context(self._project_root):
            if self._closed:
                raise EndpointExecutionError("FastAPI application runtime is closed.")
            event_loop, client = self._open()
            overrides = _resolved_overrides(dependency_override)
            prepared = prepare_get_endpoint(
                contract=contract,
                path_params=path_params,
                query_params=query_params,
                page_policy=page_policy,
                transport_overlay=transport_overlay,
            )
            with _dependency_overrides(self._app, overrides):
                execution = _execute_asgi_get(
                    client,
                    contract=contract,
                    prepared=prepared,
                    page_policy=page_policy,
                )
                return event_loop.run_until_complete(execution)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            resources = self._resources
            event_loop = self._event_loop
            self._resources = None
            self._event_loop = None
            self._client = None
            if resources is not None and event_loop is not None:
                with suppress_host_output():
                    try:
                        event_loop.run_until_complete(resources.aclose())
                    finally:
                        event_loop.close()

    def _open(self) -> tuple[asyncio.AbstractEventLoop, AsyncClient]:
        if self._event_loop is not None and self._client is not None:
            return self._event_loop, self._client
        event_loop = asyncio.new_event_loop()
        try:
            with suppress_host_output():
                resources, client = event_loop.run_until_complete(
                    _open_application_client(self._app)
                )
        except BaseException:
            with suppress_host_output():
                event_loop.close()
            raise
        self._resources = resources
        self._event_loop = event_loop
        self._client = client
        return event_loop, client


async def _open_application_client(
    app: FastAPI,
) -> tuple[AsyncExitStack, AsyncClient]:
    resources = AsyncExitStack()
    lifespan = LifespanManager(
        app,
        startup_timeout=None,
        shutdown_timeout=None,
    )
    try:
        await resources.enter_async_context(lifespan)
        transport = ASGITransport(app=lifespan.app)
        client = await resources.enter_async_context(
            AsyncClient(
                transport=transport,
                base_url="http://fervis.host",
            )
        )
    except BaseException:
        await resources.aclose()
        raise
    return resources, client


async def _execute_asgi_get(
    client: AsyncClient,
    *,
    contract: EndpointContract,
    prepared: PreparedGet,
    page_policy: dict[str, Any] | None,
) -> EndpointExecutionResult:
    client.cookies.clear()
    client.cookies.update(prepared.cookies or {})
    try:
        return await execute_prepared_get_async(
            contract=contract,
            prepared=prepared,
            page_policy=page_policy,
            get_page=lambda url, params: _get_page(
                client,
                url,
                params,
                headers=prepared.headers or {},
            ),
        )
    finally:
        client.cookies.clear()


async def _get_page(
    client: AsyncClient,
    url: str,
    query_params: dict[str, Any],
    *,
    headers: dict[str, str],
) -> tuple[int, Any]:
    with suppress_host_output():
        response = await client.get(
            url,
            params=query_params,
            headers=headers,
        )
    return response.status_code, response_body(response)


@contextmanager
def _dependency_overrides(
    app: FastAPI,
    overrides: dict[Callable[..., object], Callable[..., object]],
) -> Iterator[None]:
    if not overrides:
        yield
        return
    existing = getattr(app, "dependency_overrides", None)
    if not isinstance(existing, dict):
        raise EndpointExecutionError(
            "FastAPI app does not expose dependency_overrides."
        )
    original = dict(existing)
    try:
        existing.update(overrides)
        yield
    finally:
        existing.clear()
        existing.update(original)


def _resolved_overrides(
    override: FastAPIDependencyOverride | None,
) -> dict[Callable[..., object], Callable[..., object]]:
    if override is None:
        return {}
    principal = override.resolver(override.key, override.tenant_id)
    _validate_resolved_principal(principal, override=override)
    return {override.dependency: lambda: principal}


def _validate_resolved_principal(
    principal: object,
    *,
    override: FastAPIDependencyOverride,
) -> None:
    if principal is None:
        raise EndpointExecutionError(
            f"FastAPI resolver could not resolve principal: {override.key}"
        )
    actual = getattr(principal, override.principal_id_attr, None)
    if actual is None and isinstance(principal, dict):
        actual = principal.get(override.principal_id_attr)
    if str(actual or "") != str(override.key or ""):
        raise EndpointExecutionError(
            "FastAPI resolver returned a different principal than requested."
        )
