"""Flask-APISpec route metadata enrichment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fervis.host_api.adapters.marshmallow_schema import (
    marshmallow_schema_cardinality,
    query_params_from_marshmallow_fields,
    response_fields_from_marshmallow_schema,
)
from fervis.host_api.contracts import EndpointContract, ResponseFieldContract


def enrich_contract_from_flask_apispec(
    contract: EndpointContract,
    *,
    view: object | None,
) -> EndpointContract:
    metadata = getattr(view, "__apispec__", None)
    if not isinstance(metadata, dict):
        return contract

    response_schema = _response_schema(metadata)
    query_fields = _query_fields(metadata)
    response_fields = (
        response_fields_from_marshmallow_schema(response_schema)
        if response_schema is not None
        else contract.response_fields
    )
    query_params = (
        query_params_from_marshmallow_fields(query_fields)
        if query_fields
        else contract.query_params
    )
    if (
        response_fields == contract.response_fields
        and query_params == contract.query_params
    ):
        return contract
    return replace(
        contract,
        query_params=query_params,
        response_fields=response_fields,
        response_schema=_response_schema_payload(response_fields),
        query_schema_source="flask_apispec"
        if query_fields
        else contract.query_schema_source,
        response_schema_source=(
            "flask_apispec"
            if response_schema is not None
            else contract.response_schema_source
        ),
        response_cardinality=(
            marshmallow_schema_cardinality(response_schema)
            if response_schema is not None
            else contract.response_cardinality
        ),
    )


def _response_schema(metadata: dict[str, Any]) -> object | None:
    schemas = metadata.get("schemas")
    if not isinstance(schemas, list):
        return None
    for annotation in schemas:
        for option in _annotation_options(annotation):
            if not isinstance(option, dict):
                continue
            response = option.get(200) or option.get("200")
            if not isinstance(response, dict):
                response = option.get("default")
            if not isinstance(response, dict):
                continue
            schema = response.get("schema")
            return _schema_instance(schema)
    return None


def _query_fields(metadata: dict[str, Any]) -> dict[str, object]:
    args = metadata.get("args")
    if not isinstance(args, list):
        return {}
    fields: dict[str, object] = {}
    for annotation in args:
        for option in _annotation_options(annotation):
            if not isinstance(option, dict):
                continue
            kwargs = option.get("kwargs")
            location = kwargs.get("location") if isinstance(kwargs, dict) else None
            if location not in {None, "query", "querystring"}:
                continue
            option_args = option.get("args")
            if isinstance(option_args, dict):
                fields.update(option_args)
    return fields


def _annotation_options(annotation: object) -> tuple[object, ...]:
    options = getattr(annotation, "options", None)
    return tuple(options) if isinstance(options, list) else ()


def _schema_instance(schema: object) -> object | None:
    if schema is None:
        return None
    if isinstance(schema, type):
        return schema()
    return schema


def _response_schema_payload(
    response_fields: tuple[ResponseFieldContract, ...],
) -> dict[str, Any]:
    return {field.path: {"type": field.type} for field in response_fields}
