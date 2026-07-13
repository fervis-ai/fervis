"""Lineage records emitted by deterministic source execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fervis.lineage.enums import SourceReadStatus
from fervis.lineage.ports import SourceReadRecorderPort
from fervis.lineage.recorder import CatalogEndpointWrite, SourceReadWrite
from fervis.lookup.relation_catalog import CatalogEndpointMetadata
from fervis.lookup.source_reads.response import (
    SourceReadFailedError,
    SourceReadObservation,
)


def source_read_key_from_index(index: int) -> str:
    if index <= 0:
        raise ValueError("source read index must be positive")
    return str(index)


@dataclass(frozen=True)
class SourceReadLineageScope:
    run_id: str
    step_id: str
    recorder: SourceReadRecorderPort
    source_read_id_prefix: str = "source_read"

    def source_read_id(self, source_read_key: str) -> str:
        source_read_key = _required_source_read_key(source_read_key)
        digest = hashlib.sha256(
            f"{self.run_id}:{self.step_id}:{source_read_key}".encode("utf-8")
        ).hexdigest()[:24]
        return f"{self.source_read_id_prefix}_{digest}"


def record_source_read_observation(
    scope: SourceReadLineageScope | None,
    *,
    source_read_key: str,
    endpoint_name: str,
    catalog_endpoint: CatalogEndpointMetadata | None,
    args: dict[str, Any],
    observation: SourceReadObservation,
    completeness_json: dict[str, Any] | None = None,
) -> str | None:
    if scope is None:
        return None
    source_read_id = scope.source_read_id(source_read_key)
    recorded_endpoint = _record_catalog_endpoint(
        scope,
        endpoint_name=endpoint_name,
        catalog_endpoint=catalog_endpoint,
    )
    if observation.succeeded:
        scope.recorder.record_source_read(
            SourceReadWrite(
                source_read_id=source_read_id,
                run_id=scope.run_id,
                step_id=scope.step_id,
                catalog_endpoint_id=recorded_endpoint.catalog_endpoint_id,
                args_json=dict(args),
                status=SourceReadStatus.SUCCEEDED,
                row_count=observation.row_count,
                completeness_json=completeness_json or {"pageCount": 1},
                response_hash=observation.response_hash,
            )
        )
        return source_read_id
    _record_failed_source_read(
        scope,
        source_read_id=source_read_id,
        catalog_endpoint=recorded_endpoint,
        args=args,
        error_json=observation.error_json,
        response_hash=observation.response_hash,
    )
    return source_read_id


def require_catalog_endpoint_for_lineage(
    *,
    source_read_lineage: SourceReadLineageScope | None,
    endpoint_name: str,
    catalog_endpoint: CatalogEndpointMetadata | None,
) -> None:
    if source_read_lineage is not None and catalog_endpoint is None:
        raise SourceReadFailedError(
            endpoint_name=endpoint_name,
            error_json=_missing_catalog_endpoint_error(endpoint_name),
        )


def record_source_read_error(
    scope: SourceReadLineageScope | None,
    *,
    source_read_key: str,
    endpoint_name: str,
    catalog_endpoint: CatalogEndpointMetadata | None = None,
    args: dict[str, Any],
    error_json: dict[str, Any],
) -> None:
    if scope is None:
        return
    recorded_endpoint = _record_catalog_endpoint(
        scope,
        endpoint_name=endpoint_name,
        catalog_endpoint=catalog_endpoint,
    )
    _record_failed_source_read(
        scope,
        source_read_id=scope.source_read_id(source_read_key),
        catalog_endpoint=recorded_endpoint,
        args=args,
        error_json=error_json,
        response_hash=str(error_json.get("responseHash") or ""),
    )


def _record_catalog_endpoint(
    scope: SourceReadLineageScope,
    *,
    endpoint_name: str,
    catalog_endpoint: CatalogEndpointMetadata | None,
) -> CatalogEndpointWrite:
    recorded_endpoint = _catalog_endpoint_write(
        scope=scope,
        endpoint_name=endpoint_name,
        catalog_endpoint=catalog_endpoint,
    )
    scope.recorder.record_catalog_endpoint(recorded_endpoint)
    return recorded_endpoint


def _record_failed_source_read(
    scope: SourceReadLineageScope,
    *,
    source_read_id: str,
    catalog_endpoint: CatalogEndpointWrite,
    args: dict[str, Any],
    error_json: dict[str, Any],
    response_hash: str = "",
) -> None:
    scope.recorder.record_source_read(
        SourceReadWrite(
            source_read_id=source_read_id,
            run_id=scope.run_id,
            step_id=scope.step_id,
            catalog_endpoint_id=catalog_endpoint.catalog_endpoint_id,
            args_json=dict(args),
            status=SourceReadStatus.FAILED,
            response_hash=response_hash,
            error_json=error_json,
        )
    )


def _catalog_endpoint_write(
    *,
    scope: SourceReadLineageScope,
    endpoint_name: str,
    catalog_endpoint: CatalogEndpointMetadata | None,
) -> CatalogEndpointWrite:
    if catalog_endpoint is None:
        raise SourceReadFailedError(
            endpoint_name=endpoint_name,
            error_json=_missing_catalog_endpoint_error(endpoint_name),
        )
    catalog_endpoint_key = _required_attr(catalog_endpoint, "catalog_endpoint_key")
    return CatalogEndpointWrite(
        catalog_endpoint_id=_catalog_endpoint_id(
            run_id=scope.run_id,
            catalog_endpoint_key=catalog_endpoint_key,
        ),
        run_id=scope.run_id,
        catalog_endpoint_key=catalog_endpoint_key,
        endpoint_name=_required_attr(catalog_endpoint, "endpoint_name"),
        framework_kind=_required_attr(catalog_endpoint, "framework_kind"),
        source_namespace_kind=_required_attr(catalog_endpoint, "source_namespace_kind"),
        source_namespace_path_json=tuple(catalog_endpoint.source_namespace_path),
        route_method=_required_attr(catalog_endpoint, "route_method"),
        route_path_template=_required_attr(catalog_endpoint, "route_path_template"),
        route_name=catalog_endpoint.route_name,
        api_schema_operation_id=catalog_endpoint.api_schema_operation_id,
        handler_ref=_required_attr(catalog_endpoint, "handler_ref"),
        domain_resource_names_json=tuple(catalog_endpoint.domain_resource_names),
    )


def _catalog_endpoint_id(*, run_id: str, catalog_endpoint_key: str) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"fervis:catalog_endpoint:{run_id}:{catalog_endpoint_key}",
        )
    )


def _required_attr(catalog_endpoint: CatalogEndpointMetadata, key: str) -> str:
    return _required_value(getattr(catalog_endpoint, key), key)


def _required_value(raw_value: object, key: str) -> str:
    value = str(raw_value or "")
    if not value:
        raise ValueError(f"catalog endpoint metadata requires {key}")
    return value


def _required_source_read_key(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("source read key is required")
    return value


def _missing_catalog_endpoint_error(endpoint_name: str) -> dict[str, str]:
    return {
        "errorType": "MissingCatalogEndpointMetadata",
        "error": (
            f"{endpoint_name} is missing catalog endpoint metadata for "
            "source-read lineage"
        ),
    }
