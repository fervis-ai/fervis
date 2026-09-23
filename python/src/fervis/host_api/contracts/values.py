"""Serialized values at the public host-contract boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias


ContractScalar: TypeAlias = str | int | float | bool | None
ContractValue: TypeAlias = (
    ContractScalar | list["ContractValue"] | Mapping[str, "ContractValue"]
)


def serialize_parameter_default(value: object, *, choice_tokens: bool = False) -> ContractValue:
    """Normalize a declared default to the public parameter value representation."""
    from enum import Enum
    from typing import cast
    from pydantic_core import to_jsonable_python

    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    if choice_tokens:
        return str(value)
    return cast(ContractValue, to_jsonable_python(value))
