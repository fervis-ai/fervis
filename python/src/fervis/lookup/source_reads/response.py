"""Shared endpoint response access for source-read execution and audit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from fervis.lookup.canonical_data import canonical_runtime_json
from fervis.lookup.relation_catalog import RowCardinality
from fervis.lookup.fact_plan.row_sources.model import RowSource


MISSING = object()


class EndpointResponseError(ValueError):
    pass


class SourceReadFailedError(RuntimeError):
    def __init__(self, *, endpoint_name: str, error_json: dict[str, Any]) -> None:
        self.endpoint_name = endpoint_name
        self.error_json = dict(error_json)
        detail = str(
            self.error_json.get("error")
            or self.error_json.get("responseStatus")
            or "source read failed"
        )
        super().__init__(f"{endpoint_name} source read failed: {detail}")


@dataclass(frozen=True)
class SourceReadObservation:
    response_hash: str
    row_count: int | None = None
    error_json: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return not self.error_json


def observe_source_read_response(
    result: dict[str, Any],
    *,
    endpoint_name: str,
) -> SourceReadObservation:
    status = result.get("responseStatus")
    response_hash = response_body_hash(result)
    if not isinstance(status, int):
        return SourceReadObservation(
            response_hash=response_hash,
            error_json={
                "responseStatus": status,
                "error": f"{endpoint_name} response missing integer HTTP status",
            },
        )
    if status < 200 or status >= 300:
        return SourceReadObservation(
            response_hash=response_hash,
            error_json={
                "responseStatus": status,
                "responseHash": response_hash,
                "error": f"{endpoint_name} returned HTTP {status}",
            },
        )
    return SourceReadObservation(
        response_hash=response_hash,
        row_count=_response_body_row_count(result.get("responseBody")),
    )


def response_body_hash(result: dict[str, Any]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_runtime_json(result.get("responseBody")).encode("utf-8")
        ).hexdigest()
    )


def source_read_completeness(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "pageCount": int(result.get("pageCount") or 1),
        "truncated": bool(result.get("truncated") is True),
    }


def extract_source_read_rows(
    result: dict[str, Any],
    *,
    endpoint_name: str,
    row_source: RowSource,
) -> tuple[dict[str, Any], ...]:
    body = endpoint_response_body(result, endpoint_name=endpoint_name)
    return extract_row_source_rows(body, row_source=row_source)


def endpoint_response_body(result: dict[str, Any], *, endpoint_name: str) -> Any:
    status = result.get("responseStatus")
    if not isinstance(status, int):
        raise EndpointResponseError(f"{endpoint_name} response missing HTTP status")
    if status < 200 or status >= 300:
        raise EndpointResponseError(f"{endpoint_name} returned HTTP {status}")
    return result.get("responseBody")


def extract_row_source_rows(
    body: Any, *, row_source: RowSource
) -> tuple[dict[str, Any], ...]:
    if not row_source.parent_row_path:
        return extract_response_rows(
            body,
            row_source.row_path,
            cardinality=row_source.row_cardinality,
        )
    parent_rows = extract_response_rows(
        body,
        row_source.parent_row_path,
        cardinality=RowCardinality.MANY,
    )
    child_path = relative_response_path(row_source.row_path, row_source.parent_row_path)
    rows: list[dict[str, Any]] = []
    for parent in parent_rows:
        child_rows = extract_response_rows(
            parent,
            child_path,
            cardinality=row_source.row_cardinality,
        )
        parent_context = _parent_row_context(parent, child_path=child_path)
        rows.extend({**parent_context, **child} for child in child_rows)
    return tuple(rows)


def extract_response_rows(
    body: Any,
    row_path: str,
    *,
    cardinality: RowCardinality,
) -> tuple[dict[str, Any], ...]:
    value = path_value(body, row_path, missing=MISSING)
    if isinstance(value, list):
        if cardinality != RowCardinality.MANY:
            raise EndpointResponseError(
                f"response row path {row_path or '<root>'} expected one row"
            )
        if any(not isinstance(item, dict) for item in value):
            raise EndpointResponseError(
                f"response row path {row_path or '<root>'} contains non-object rows"
            )
        return tuple(item for item in value if isinstance(item, dict))
    if isinstance(value, dict):
        if cardinality != RowCardinality.ONE:
            raise EndpointResponseError(
                f"response row path {row_path or '<root>'} expected many rows"
            )
        return (value,)
    if value is MISSING:
        raise EndpointResponseError(
            f"response row path {row_path or '<root>'} is unavailable"
        )
    raise EndpointResponseError(
        f"response row path {row_path or '<root>'} is not an object row"
    )


def _response_body_row_count(body: Any) -> int:
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        return len(body["data"])
    if isinstance(body, list):
        return len(body)
    if isinstance(body, dict):
        return 1
    return 0


def required_response_path_value(payload: Any, path: str) -> Any:
    value = path_value(payload, path, missing=MISSING)
    if value is MISSING:
        raise EndpointResponseError(
            f"response field path {path or '<root>'} is unavailable"
        )
    return value


def path_value(payload: Any, path: str, *, missing: object = None) -> Any:
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        current = _path_part_value(current, part, missing=missing)
        if current is missing:
            return missing
    return current


def relative_response_path(field_path: str, row_path: str) -> str:
    if not row_path:
        return field_path
    prefix = f"{row_path}."
    if field_path == row_path:
        return ""
    if field_path.startswith(prefix):
        return field_path[len(prefix) :]
    return field_path


def _path_part_value(value: Any, part: str, *, missing: object = None) -> Any:
    if isinstance(value, dict):
        return value[part] if part in value else missing
    if isinstance(value, list):
        output: list[Any] = []
        for item in value:
            child = _path_part_value(item, part, missing=missing)
            if isinstance(child, list):
                output.extend(child)
            elif child is not missing:
                output.append(child)
        return output if output else missing
    return missing


def _parent_row_context(parent: dict[str, Any], *, child_path: str) -> dict[str, Any]:
    child_root = child_path.split(".", 1)[0] if child_path else ""
    return {key: value for key, value in parent.items() if key != child_root}
