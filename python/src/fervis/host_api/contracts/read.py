"""Host API read contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ReadInvocation:
    """Selected safe host API read."""

    endpoint_name: str
    path_params: Mapping[str, Any] = field(default_factory=dict)
    query_params: Mapping[str, Any] = field(default_factory=dict)
    page_policy: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        endpoint_name = str(self.endpoint_name or "").strip()
        if not endpoint_name:
            raise ValueError("read invocation endpoint_name is required")
        object.__setattr__(self, "endpoint_name", endpoint_name)
        object.__setattr__(self, "path_params", dict(self.path_params or {}))
        object.__setattr__(self, "query_params", dict(self.query_params or {}))
        object.__setattr__(
            self,
            "page_policy",
            None if self.page_policy is None else dict(self.page_policy),
        )

    def to_execution_kwargs(self) -> dict[str, Any]:
        return {
            "endpoint_name": self.endpoint_name,
            "path_params": dict(self.path_params),
            "query_params": dict(self.query_params),
            "page_policy": None if self.page_policy is None else dict(self.page_policy),
        }
