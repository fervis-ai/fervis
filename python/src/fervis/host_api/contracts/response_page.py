"""An HTTP page carries representation kind separately from its data."""

from dataclasses import dataclass
from typing import Any
from fervis.types.enums import StrEnum


class ResponseFormat(StrEnum):
    JSON = "json"
    TEXT = "text"
    EMPTY = "empty"


@dataclass(frozen=True)
class ResponsePage:
    status: int
    body: Any
    format: ResponseFormat = ResponseFormat.JSON
