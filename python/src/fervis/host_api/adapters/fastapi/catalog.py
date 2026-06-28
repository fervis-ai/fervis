"""FastAPI/OpenAPI endpoint catalog translation."""

from __future__ import annotations

from pathlib import Path

from fervis.host_api.contracts import (
    EndpointContract,
    FrameworkKind,
    SourceNamespaceKind,
)
from fervis.host_api.adapters.openapi import endpoint_contracts_from_openapi
from fervis.project.integration import FastAPIAppSource
from fervis.project.source_paths import (
    normalize_source_path_prefixes,
)

from .loading import fastapi_openapi_schema, import_fastapi_app


def get_fastapi_endpoint_contracts(
    *,
    sources: tuple[FastAPIAppSource, ...],
    project_root: Path,
) -> tuple[EndpointContract, ...]:
    contracts: list[EndpointContract] = []
    for source in sources:
        prefixes = _normalized_path_prefixes(tuple(source.path_prefixes))
        for import_path in source.import_paths:
            app = import_fastapi_app(import_path, project_root=project_root)
            schema = fastapi_openapi_schema(app, import_path=import_path)
            contracts.extend(
                endpoint_contracts_from_openapi(
                    schema,
                    source_name=source.name,
                    import_path=import_path,
                    path_prefixes=prefixes,
                    framework_kind=FrameworkKind.FASTAPI.value,
                    source_namespace_kind=SourceNamespaceKind.FASTAPI_APP.value,
                    source_namespace_path=(source.name,),
                )
            )
    return tuple(sorted(contracts, key=lambda item: item.path_template))


def _normalized_path_prefixes(path_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    try:
        return normalize_source_path_prefixes(path_prefixes)
    except ValueError as exc:
        raise ValueError(f"FastAPIAppSource.path_prefixes invalid: {exc}") from exc
