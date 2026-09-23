"""Current-response structure, distinct from declared schema and identity authority."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
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
from fervis.lookup.canonical_data import canonical_runtime_json


@dataclass(frozen=True)
class ReadRepresentationObservation:
    read: EndpointRead
    response_hash: str


def inspection_request_fingerprint(args: dict[str, Any]) -> str:
    return "sha256:" + sha256(canonical_runtime_json(args).encode("utf-8")).hexdigest()


def observe_read_representation(
    read: EndpointRead, result: dict[str, Any], *, request_args: dict[str, Any] | None = None
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

    def primitive(value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    def describe_rows(value, path: str, parent: str, row_id: str, *, primitive_name: str = "value"):
        if isinstance(value, list):
            if value and all(primitive(item) for item in value):
                rows = [{primitive_name: item} for item in value]
                primitive_rows = True
            elif all(isinstance(item, dict) for item in value):
                rows = value
                primitive_rows = False
            else:
                raise ValueError(
                    "A response row array cannot mix objects and primitive values."
                )
            cardinality = RowCardinality.MANY
        elif isinstance(value, dict):
            rows, cardinality = [value], RowCardinality.ONE
            primitive_rows = False
        elif primitive(value):
            rows, cardinality = [{primitive_name: value}], RowCardinality.ONE
            primitive_rows = True
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
        if primitive_rows:
            fields.append(CatalogField(
                ref=f"observed_primitive:{row_id}",
                path=f"{path}.{primitive_name}" if path else primitive_name,
                row_path_id=row_id,
                type=_joined_type({_value_type(row[primitive_name]) for row in rows if row[primitive_name] is not None}),
                nullable=any(row[primitive_name] is None for row in rows),
                metadata={"name": primitive_name, "representation_authority": "observed_response"},
            ))
            return
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
                if all(isinstance(item, dict) or primitive(item) for item in elements):
                    reserved = {key for row in rows for key in row}
                    primitive_name = "value"
                    suffix = 2
                    while primitive_name in reserved:
                        primitive_name = f"value_{suffix}"
                        suffix += 1
                    describe_rows(
                        elements,
                        path,
                        next(p.path for p in paths if p.id == row_id),
                        f"row:{path}",
                        primitive_name=primitive_name,
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
            "observed_request_fingerprint": inspection_request_fingerprint(request_args or {}),
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


def can_inspect_representation(
    read: EndpointRead, *, inspection_args: dict[str, Any] | None = None
) -> bool:
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
        and {
            param.ref for param in read.params if requires_caller_supplied_input(param)
        } <= set(inspection_args or {})
        and set(inspection_args or {}) <= {param.ref for param in read.params}
    )


def inspect_selected_representations(
    catalog, *, read_ids, data_access_port, on_response=None, on_failure=None,
    inspection_args_by_read=None,
):
    """Inspect selected routes with exact, type-checked caller arguments."""
    selected = set(read_ids)
    supplied = inspection_args_by_read or {}
    if set(supplied) - selected:
        raise ValueError("inspection arguments target an unselected read")
    reads = []
    for read in catalog.reads:
        args = supplied.get(read.id)
        if read.id not in selected:
            reads.append(read)
            continue
        if not can_inspect_representation(read, inspection_args=args):
            if args is not None:
                raise ValueError("inspection arguments do not match an inspectable read")
            reads.append(read)
            continue
        from fervis.lookup.relation_catalog.parameter_values import (
            parse_catalog_parameter_value,
        )

        params = {param.ref: param for param in read.params}
        checked_args = {
            ref: parse_catalog_parameter_value(
                value, type_name=params[ref].type, choices=params[ref].choices
            )
            for ref, value in (args or {}).items()
        }
        try:
            result = data_access_port.read(
                endpoint_name=read.endpoint_name, args=checked_args
            )
        except Exception as exc:
            result = {
                "responseStatus": None,
                "responseBody": None,
                "inspectionError": type(exc).__name__,
            }
        if on_response is not None:
            if checked_args:
                on_response(read, result, args=checked_args)
            else:
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
            observation = observe_read_representation(
                read, result, request_args=checked_args
            )
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
