"""Normalized OpenAPI operations used by Fervis catalog translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.host_api.contracts import (
    CandidateKeyContract,
    EntityReferenceContract,
    PaginationContract,
)


@dataclass(frozen=True)
class OpenApiParameter:
    name: str
    location: str
    schema: dict[str, Any]
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class OpenApiOperation:
    operation_id: str
    method: str
    path_template: str
    summary: str
    tags: tuple[str, ...]
    parameters: tuple[OpenApiParameter, ...]
    response_schema: dict[str, Any]
    pagination: PaginationContract | None
    candidate_keys: tuple[CandidateKeyContract, ...]
    entity_references: tuple[EntityReferenceContract, ...]
