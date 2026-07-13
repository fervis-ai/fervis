"""Translate JSON:API-style resource schema metadata into Fervis endpoint facts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fervis.host_api.adapters.marshmallow_schema import (
    response_fields_from_marshmallow_schema,
)
from fervis.host_api.contracts import EndpointContract, ResponseFieldContract


def enrich_contract_from_jsonapi_resource(
    contract: EndpointContract,
    *,
    view: object | None,
) -> EndpointContract:
    resource = _view_resource(view)
    schema = _schema_instance(getattr(resource, "schema", None))
    if schema is None:
        return contract
    fields = response_fields_from_marshmallow_schema(schema)
    if not fields:
        return contract
    cardinality = _resource_cardinality(resource, contract=contract)
    return replace(
        contract,
        response_fields=_jsonapi_response_fields(fields, cardinality=cardinality),
        response_schema=_response_schema_payload(fields, cardinality=cardinality),
        response_schema_source="jsonapi",
        response_cardinality=cardinality,
    )


def _view_resource(view: object | None) -> object | None:
    if view is None:
        return None
    return getattr(view, "view_class", None) or view


def _schema_instance(schema: object | None) -> object | None:
    if schema is None:
        return None
    if isinstance(schema, type):
        return schema()
    return schema


def _jsonapi_response_fields(
    fields: tuple[ResponseFieldContract, ...],
    *,
    cardinality: str,
) -> tuple[ResponseFieldContract, ...]:
    data_field = ResponseFieldContract(
        name="data",
        path="data",
        type="array" if cardinality == "many" else "object",
    )
    attribute_fields = tuple(
        replace(
            field,
            path=(f"data.attributes.{field.path}" if field.path else "data.attributes"),
        )
        for field in fields
    )
    return (data_field, *attribute_fields)


def _response_schema_payload(
    fields: tuple[ResponseFieldContract, ...],
    *,
    cardinality: str,
) -> dict[str, Any]:
    attributes = {
        f"data.attributes.{field.path}" if field.path else "data.attributes": {
            "type": field.type
        }
        for field in fields
    }
    return {
        "data": {"type": "array" if cardinality == "many" else "object"},
        **attributes,
    }


def _resource_cardinality(
    resource: object | None,
    *,
    contract: EndpointContract,
) -> str:
    resource_type = resource if isinstance(resource, type) else resource.__class__
    class_names = {
        cls.__name__.lower()
        for cls in getattr(resource_type, "__mro__", ())
        if hasattr(cls, "__name__")
    }
    if any("resourcelist" in name for name in class_names):
        return "many"
    if any("resourcedetail" in name for name in class_names):
        return "one"
    return "one" if contract.path_params else "many"
