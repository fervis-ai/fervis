"""Current-response structure, distinct from declared schema and identity authority."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.relation_catalog import (
    CatalogField,
    EndpointRead,
    RowCardinality,
    RowPath,
)
from fervis.lookup.source_reads.response import (
    endpoint_response_body,
    response_body_hash,
)


@dataclass(frozen=True)
class ReadRepresentationObservation:
    read: EndpointRead
    response_hash: str


def observe_read_representation(
    read: EndpointRead, result: dict[str, Any]
) -> ReadRepresentationObservation:
    """Describe only the values observed in one authorized response.

    A sampled id is not a key; sampled values are not an enum. The shape does
    not claim that an unobserved property is absent from the API's schema.
    """
    if result.get("responseFormat") != "json":
        raise ValueError(
            "Response inspection requires explicit JSON representation evidence."
        )
    body = endpoint_response_body(result, endpoint_name=read.endpoint_name)
    fields: list[CatalogField] = []
    paths: list[RowPath] = list(read.row_paths)

    def describe_rows(value, path: str, parent: str, row_id: str):
        if isinstance(value, list):
            if any(not isinstance(item, dict) for item in value):
                raise ValueError(
                    "This response needs a primitive-value row representation."
                )
            rows, cardinality = value, RowCardinality.MANY
        elif isinstance(value, dict):
            rows, cardinality = [value], RowCardinality.ONE
        else:
            raise ValueError(
                "This response needs a primitive-value row representation."
            )
        declared = next((existing for existing in paths if existing.path == path), None)
        if declared is not None:
            if declared.cardinality is not cardinality:
                raise ValueError(
                    "Observed rows contradict declared response cardinality."
                )
            row_id = declared.id
        else:
            paths.append(
                RowPath(
                    id=row_id, path=path, cardinality=cardinality, parent_path=parent
                )
            )
        describe_objects(rows, path, row_id)

    def describe_objects(rows: list[dict], prefix: str, row_id: str):
        names = sorted({name for row in rows for name in row})
        for name in names:
            if not isinstance(name, str) or "." in name or not name:
                raise ValueError(
                    "The response requires a structured field-path representation."
                )
            path = f"{prefix}.{name}" if prefix else name
            values = [row[name] for row in rows if name in row]
            types = {_value_type(value) for value in values if value is not None}
            field_type = _joined_type(types)
            fields.append(
                CatalogField(
                    ref=f"field.{path}",
                    path=path,
                    row_path_id=row_id,
                    type=field_type,
                    nullable=len(values) != len(rows)
                    or any(value is None for value in values),
                    metadata={
                        "name": name,
                        "representation_authority": "observed_response",
                    },
                )
            )
            objects = [value for value in values if isinstance(value, dict)]
            if field_type == "object":
                describe_objects(objects, path, row_id)
            elif field_type == "array":
                elements = [
                    item
                    for value in values
                    if isinstance(value, list)
                    for item in value
                ]
                if all(isinstance(item, dict) for item in elements):
                    describe_rows(
                        elements,
                        path,
                        next(p.path for p in paths if p.id == row_id),
                        f"row:{path}",
                    )

    if read.response_envelope.results_path:
        from fervis.lookup.source_reads.response import required_response_path_value

        result_path = read.response_envelope.results_path
        describe_rows(
            required_response_path_value(body, result_path),
            result_path,
            "",
            f"row:{result_path}",
        )
    else:
        describe_rows(body, "", "", "root")
    observed = replace(
        read,
        row_paths=tuple(paths),
        fields=tuple(fields),
        source_metadata={
            **(read.source_metadata or {}),
            "representation_authority": "observed_response",
        },
    )
    return ReadRepresentationObservation(observed, response_body_hash(result))


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "any"


def _joined_type(types: set[str]) -> str:
    if not types:
        return "any"
    if len(types) == 1:
        return next(iter(types))
    if types <= {"integer", "number"}:
        return "number"
    return "any"


def can_inspect_representation(read: EndpointRead) -> bool:
    """Missing fields do not erase declared structure or read capability."""
    from fervis.lookup.relation_catalog.model import requires_caller_supplied_input

    metadata = read.source_metadata or {}
    return (
        not read.fields
        and (
            not read.row_paths
            or metadata.get("representation_authority") == "unobserved"
        )
        and metadata.get("representation_authority") != "observed_response"
        and metadata.get("representation_status") not in {"unavailable", "read_failed"}
        and not any(requires_caller_supplied_input(param) for param in read.params)
    )


def inspect_selected_representations(
    catalog, *, read_ids, data_access_port, on_response=None, on_failure=None
):
    """Inspect selected, directly invokable routes under the caller's read port."""
    selected = set(read_ids)
    reads = []
    for read in catalog.reads:
        if read.id not in selected or not can_inspect_representation(read):
            reads.append(read)
            continue
        try:
            result = data_access_port.read(endpoint_name=read.endpoint_name, args={})
        except Exception as exc:
            result = {
                "responseStatus": None,
                "responseBody": None,
                "inspectionError": type(exc).__name__,
            }
        if on_response is not None:
            on_response(read, result)
        # Discovery may defer a failed candidate while inspecting other routes.
        # Execution requires its selected read and keeps the default strict path.
        from fervis.lookup.source_reads.response import EndpointResponseError
        try:
            endpoint_response_body(result, endpoint_name=read.endpoint_name)
        except EndpointResponseError as exc:
            if on_failure is None:
                raise
            on_failure(exc)
            reads.append(replace(read, source_metadata={
                **(read.source_metadata or {}), "representation_status": "read_failed",
                "representation_failure": str(exc),
            }))
            continue
        try:
            observation = observe_read_representation(read, result)
        except ValueError as exc:
            reads.append(
                replace(
                    read,
                    source_metadata={
                        **(read.source_metadata or {}),
                        "representation_status": "unavailable",
                        "representation_error": type(exc).__name__,
                    },
                )
            )
        else:
            reads.append(observation.read)
    return replace(catalog, reads=tuple(reads))
