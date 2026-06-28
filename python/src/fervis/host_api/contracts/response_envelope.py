from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TOTAL_COUNT_FIELD = "count"
PAGINATION_FIELD = "pagination"
HAS_MORE_FIELD = "has_more"
HAS_MORE_FIELDS = (HAS_MORE_FIELD, "hasMore")


def pagination_mapping(body: Any) -> Mapping[str, Any]:
    if not isinstance(body, Mapping):
        return {}
    value = body.get(PAGINATION_FIELD)
    return value if isinstance(value, Mapping) else {}


def has_more_value(body: Any) -> Any:
    pagination = pagination_mapping(body)
    for field in HAS_MORE_FIELDS:
        if field in pagination:
            return pagination.get(field)
    return None
