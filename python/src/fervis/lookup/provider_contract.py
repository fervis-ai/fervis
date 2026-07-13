"""Strict decoding for explicitly typed provider-output contracts."""

from __future__ import annotations

from types import UnionType
from typing import TypeVar, Union, get_args, get_origin, get_type_hints
from typing_extensions import Self


class ProviderOutput:
    """Base for closed DTOs authored by a model tool call."""

    @classmethod
    def schema(cls, properties: dict[str, object]) -> dict[str, object]:
        declared = cls._field_names()
        unsupported = set(properties) - set(declared)
        if unsupported:
            raise ValueError(f"{cls.__name__} schema has unsupported fields")
        required = cls._required_field_names()
        missing_required = set(required) - set(properties)
        if missing_required:
            raise ValueError(f"{cls.__name__} schema is missing required fields")
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": [name for name in declared if name in required],
        }

    @classmethod
    def parse(cls, value: object) -> Self:
        payload = _object_payload(value, contract_name=cls.__name__)
        declared = cls._field_names()
        unexpected = set(payload) - set(declared)
        if unexpected:
            field_name = sorted(unexpected)[0]
            raise ValueError(
                f"{cls.__name__} contains unparsed fields: {field_name}; "
                "unexpected field"
            )
        missing = set(cls._required_field_names()) - set(payload)
        if missing:
            raise ValueError(
                f"{cls.__name__} missing required field: {sorted(missing)[0]}"
            )
        hints = get_type_hints(cls)
        decoded = {
            field_name: _decode_value(
                payload.get(field_name),
                expected_type=hints[field_name],
                path=f"{cls.__name__}.{field_name}",
            )
            for field_name in declared
        }
        return cls(**decoded)

    @classmethod
    def _field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__annotations__)

    @classmethod
    def _required_field_names(cls) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in cls._field_names()
            if field_name not in cls.__dict__
        )


class ProviderObject:
    """Opaque object used only where a discriminator selects the concrete DTO."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def discriminator(self, field_name: str) -> str:
        value = self._payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"provider discriminator {field_name} must be text")
        return value

    def has_field(self, field_name: str) -> bool:
        return field_name in self._payload

    def parse_as(self, contract: type[_ProviderOutputT]) -> _ProviderOutputT:
        return contract.parse(self)

    def named(self, contract: type[_ProviderOutputT]) -> dict[str, _ProviderOutputT]:
        return {key: contract.parse(value) for key, value in self._payload.items()}


_ProviderOutputT = TypeVar("_ProviderOutputT", bound=ProviderOutput)


def _object_payload(value: object, *, contract_name: str) -> dict[str, object]:
    if isinstance(value, ProviderObject):
        return value._payload
    if not isinstance(value, dict):
        raise ValueError(f"{contract_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{contract_name} field names must be strings")
    return dict(value)


def _decode_value(value: object, *, expected_type: object, path: str) -> object:
    if expected_type is ProviderObject:
        return ProviderObject(_object_payload(value, contract_name=path))
    if isinstance(expected_type, type) and issubclass(expected_type, ProviderOutput):
        return expected_type.parse(value)
    origin = get_origin(expected_type)
    arguments = get_args(expected_type)
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        item_type = arguments[0]
        return tuple(
            _decode_value(item, expected_type=item_type, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if origin is dict:
        payload = _object_payload(value, contract_name=path)
        value_type = arguments[1]
        return {
            key: _decode_value(item, expected_type=value_type, path=f"{path}.{key}")
            for key, item in payload.items()
        }
    if origin in {Union, UnionType}:
        if value is None and type(None) in arguments:
            return None
        variants = tuple(argument for argument in arguments if argument is not type(None))
        for variant in variants:
            try:
                return _decode_value(value, expected_type=variant, path=path)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"{path} does not match any declared type")
    if expected_type is str:
        if not isinstance(value, str):
            raise ValueError(f"{path} must be text")
        return value
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError(f"{path} must be a number")
        return value
    if expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be boolean")
        return value
    raise TypeError(f"{path} has unsupported provider field type {expected_type!r}")


__all__ = ["ProviderObject", "ProviderOutput"]
