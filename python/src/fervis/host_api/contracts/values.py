"""Serialized values at the public host-contract boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias


ContractScalar: TypeAlias = str | int | float | bool | None
ContractValue: TypeAlias = (
    ContractScalar | list["ContractValue"] | Mapping[str, "ContractValue"]
)
