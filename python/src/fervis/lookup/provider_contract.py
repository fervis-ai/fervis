"""Small helpers for provider-output DTO contracts."""

from __future__ import annotations

from typing import ClassVar, Self


class ProviderOutput:
    optional_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, **payload: object) -> None:
        for field_name in self.field_names():
            setattr(self, field_name, payload.get(field_name))

    @classmethod
    def schema(cls, properties: dict[str, object]) -> dict[str, object]:
        field_names = frozenset(cls.field_names())
        unsupported = set(properties) - field_names
        if unsupported:
            raise ValueError(f"{cls.__name__} schema has unsupported fields")
        missing_required = set(cls.required_field_names()) - set(properties)
        if missing_required:
            raise ValueError(f"{cls.__name__} schema is missing required fields")
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": [
                field_name
                for field_name in cls.field_names()
                if field_name in properties and field_name not in cls.optional_fields
            ],
        }

    @classmethod
    def parse(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise ValueError(f"{cls.__name__} must be an object")
        field_names = frozenset(cls.field_names())
        unexpected = set(value) - field_names
        if unexpected:
            field = sorted(unexpected)[0]
            raise ValueError(
                f"{cls.__name__} contains unparsed fields: {field}; unexpected field"
            )
        missing = set(cls.required_field_names()) - set(value)
        if missing:
            raise ValueError(
                f"{cls.__name__} missing required field: {sorted(missing)[0]}"
            )
        return cls(**{field_name: value.get(field_name) for field_name in field_names})

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__annotations__)

    @classmethod
    def required_field_names(cls) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in cls.field_names()
            if field_name not in cls.optional_fields
        )


def provider_output_type(
    name: str,
    fields: tuple[str, ...],
    *,
    optional_fields: tuple[str, ...] = (),
) -> type[ProviderOutput]:
    return type(
        name,
        (ProviderOutput,),
        {
            "__annotations__": {field_name: object for field_name in fields},
            "optional_fields": frozenset(optional_fields),
        },
    )
