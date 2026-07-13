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
    class_names = {
        cls.__name__.lstrip("_").casefold() for cls in field.__class__.__mro__
    }
    if class_names & {"integer", "int"}:
        return "integer"
    if class_names & {"decimal", "float", "number"}:
        return "decimal"
    if class_names & {"boolean", "bool"}:
        return "boolean"
    if "datetime" in class_names:
        return "datetime"
    if "date" in class_names:
        return "date"
    if "list" in class_names:
        return "array"
    if "nested" in class_names:
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
