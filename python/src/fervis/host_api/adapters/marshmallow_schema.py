"""Translate Marshmallow schema metadata into Fervis endpoint facts."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.host_api.contracts import (
    ParameterContract,
    ResponseFieldContract,
)


def response_fields_from_marshmallow_schema(
    schema: object,
) -> tuple[ResponseFieldContract, ...]:
    fields = _schema_fields(schema)
    return tuple(
        ResponseFieldContract(
            name=name,
            path=name,
            type=_field_type(field),
            description=_description(field),
        )
        for name, field in fields.items()
        if not bool(getattr(field, "load_only", False))
    )


def query_params_from_marshmallow_fields(
    fields_by_name: Mapping[str, object],
) -> tuple[ParameterContract, ...]:
    return tuple(
        ParameterContract(
            name=str(name),
            type=_field_type(field),
            required=bool(getattr(field, "required", False)),
            description=_description(field),
            choices=_choices(field),
            source="query",
        )
        for name, field in fields_by_name.items()
    )


def marshmallow_schema_cardinality(schema: object) -> str:
    return "many" if bool(getattr(schema, "many", False)) else "one"


def _schema_fields(schema: object) -> Mapping[str, object]:
    fields = getattr(schema, "fields", None)
    return fields if isinstance(fields, Mapping) else {}


def _field_type(field: object) -> str:
    class_name = field.__class__.__name__.lower()
    if "integer" in class_name or class_name in {"int"}:
        return "integer"
    if any(name in class_name for name in ("decimal", "float", "number")):
        return "decimal"
    if "boolean" in class_name or class_name == "bool":
        return "boolean"
    if class_name == "date":
        return "date"
    if "datetime" in class_name:
        return "datetime"
    if "list" in class_name:
        return "array"
    if "nested" in class_name:
        return "object"
    return "string"


def _description(field: object) -> str:
    metadata = getattr(field, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("description") or "")


def _choices(field: object) -> tuple[str, ...]:
    validators = getattr(field, "validators", ()) or ()
    for validator in validators:
        choices = getattr(validator, "choices", None)
        if choices is not None:
            return tuple(str(choice) for choice in choices)
    return ()
