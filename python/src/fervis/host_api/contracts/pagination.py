"""Host-declared mechanics for bounded pagination traversal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fervis.types.enums import StrEnum
from fervis.host_api.contracts.values import ContractValue


class PaginationKind(StrEnum):
    OFFSET = "offset"
    PAGE_NUMBER = "page_number"


@dataclass(frozen=True)
class PaginationContract:
    kind: PaginationKind
    position_query_param: str
    page_size_query_param: str
    results_path: str
    page_size: int
    max_page_size: int
    total_path: str = ""
    continuation_path: str = ""

    def __post_init__(self) -> None:
        if not self.results_path:
            raise ValueError("pagination result path is required")
        if not self.position_query_param:
            raise ValueError("pagination position query parameter is required")
        if self.page_size < 1:
            raise ValueError("pagination page size is invalid")
        if self.max_page_size < 1:
            raise ValueError("pagination maximum page size is invalid")
        if self.page_size > self.max_page_size:
            raise ValueError("pagination page size exceeds its maximum")
        if not self.total_path and not self.continuation_path:
            raise ValueError("pagination completeness evidence is required")

    def to_public_dict(self) -> dict[str, ContractValue]:
        payload: dict[str, ContractValue] = {
            "kind": self.kind.value,
            "positionQueryParam": self.position_query_param,
            "pageSizeQueryParam": self.page_size_query_param,
            "resultsPath": self.results_path,
            "pageSize": self.page_size,
            "maxPageSize": self.max_page_size,
        }
        if self.total_path:
            payload["totalPath"] = self.total_path
        if self.continuation_path:
            payload["continuationPath"] = self.continuation_path
        return payload

    @classmethod
    def from_public_dict(
        cls,
        value: Mapping[str, ContractValue],
    ) -> PaginationContract:
        return cls(
            kind=PaginationKind(_required_text(value, "kind")),
            position_query_param=_required_text(value, "positionQueryParam"),
            page_size_query_param=_required_text(value, "pageSizeQueryParam"),
            results_path=_required_text(value, "resultsPath"),
            page_size=_required_integer(value, "pageSize"),
            max_page_size=_required_integer(value, "maxPageSize"),
            total_path=_optional_text(value, "totalPath"),
            continuation_path=_optional_text(value, "continuationPath"),
        )


def _required_text(value: Mapping[str, ContractValue], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"pagination {key} must be a non-empty string")
    return item


def _optional_text(value: Mapping[str, ContractValue], key: str) -> str:
    item = value.get(key)
    if item is None:
        return ""
    if not isinstance(item, str):
        raise ValueError(f"pagination {key} must be a string")
    return item


def _required_integer(value: Mapping[str, ContractValue], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"pagination {key} must be an integer")
    return item
