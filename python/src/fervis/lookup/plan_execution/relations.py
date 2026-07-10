"""Runtime relation rows and completeness proof metadata."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
from typing import Mapping

from fervis.lookup.canonical_data import canonical_runtime_json

from fervis.lookup.relation_catalog.model import (
    CompletenessPolicy,
    EndpointRead,
    PaginationMode,
)


Row = Mapping[str, object]


class CompletenessStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class CompletenessSourceKind(StrEnum):
    API_READ = "api_read"
    GENERATED_CALENDAR = "generated_calendar"
    MEMORY_READ = "memory_read"
    OPERATION_OUTPUT = "operation_output"
    UNKNOWN = "unknown"


class RelationSetKind(StrEnum):
    UNIVERSE = "universe"
    OBSERVATION = "observation"
    UNKNOWN = "unknown"


class PaginationCompleteness(StrEnum):
    NOT_PAGINATED = "not_paginated"
    TERMINAL = "terminal"
    PAGE_CAP_REACHED = "page_cap_reached"
    TRUNCATED = "truncated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CompletenessProof:
    status: CompletenessStatus = CompletenessStatus.UNKNOWN
    source_kind: CompletenessSourceKind = CompletenessSourceKind.UNKNOWN
    set_kind: RelationSetKind = RelationSetKind.UNKNOWN
    scope_fingerprint: str = ""
    proof_refs: tuple[str, ...] = ()
    row_count: int | None = None
    pagination: PaginationCompleteness = PaginationCompleteness.UNKNOWN


@dataclass(frozen=True)
class RelationEvidence:
    source_refs: tuple[str, ...] = ()
    read_refs: tuple[str, ...] = ()
    authority_refs: tuple[str, ...] = ()
    snapshot_hash: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationRows:
    id: str
    rows: tuple[Row, ...] = ()
    grain_keys: tuple[str, ...] = ()
    field_types: Mapping[str, str] | None = None
    field_answer_output_ids: Mapping[str, tuple[str, ...]] | None = None
    completeness: CompletenessProof = CompletenessProof()
    identity_type: str = ""
    evidence: RelationEvidence = RelationEvidence()

    def __post_init__(self) -> None:
        if self.completeness.row_count is None:
            object.__setattr__(
                self,
                "completeness",
                replace(self.completeness, row_count=len(self.rows)),
            )
    def with_filtered_rows(
        self,
        rows: tuple[Row, ...],
        *,
        proof_refs: tuple[str, ...],
        scope_fingerprint: str,
    ) -> RelationRows:
        combined_proof_refs = tuple(
            dict.fromkeys(
                (
                    *self.evidence.proof_refs,
                    *self.completeness.proof_refs,
                    *proof_refs,
                )
            )
        )
        return replace(
            self,
            rows=rows,
            evidence=replace(
                self.evidence,
                snapshot_hash=relation_snapshot_hash(rows),
                proof_refs=combined_proof_refs,
            ),
            completeness=replace(
                self.completeness,
                row_count=len(rows),
                proof_refs=combined_proof_refs,
                scope_fingerprint=scope_fingerprint,
            ),
        )

    def with_scope(
        self,
        *,
        proof_refs: tuple[str, ...],
        scope_fingerprint: str,
    ) -> RelationRows:
        combined_proof_refs = tuple(
            dict.fromkeys(
                (
                    *self.evidence.proof_refs,
                    *self.completeness.proof_refs,
                    *proof_refs,
                )
            )
        )
        return replace(
            self,
            evidence=replace(self.evidence, proof_refs=combined_proof_refs),
            completeness=replace(
                self.completeness,
                proof_refs=combined_proof_refs,
                scope_fingerprint=scope_fingerprint,
            ),
        )


def relation_snapshot_hash(rows: tuple[Row, ...]) -> str:
    payload = canonical_runtime_json([dict(row) for row in rows])
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class RowContextStore:
    by_relation_id: Mapping[str, tuple[Row, ...]] = field(default_factory=dict)

    def rows_for_relation(self, relation_id: str) -> tuple[Row, ...]:
        return self.by_relation_id.get(relation_id, ())


def api_read_completeness_proof(
    read: EndpointRead,
    *,
    row_count: int,
    set_kind: RelationSetKind = RelationSetKind.UNKNOWN,
    scope_fingerprint: str = "",
    reached_terminal_page: bool = False,
    page_cap_reached: bool = False,
    truncated: bool = False,
    proof_refs: tuple[str, ...] = (),
) -> CompletenessProof:
    if truncated:
        return _api_proof(
            status=CompletenessStatus.INCOMPLETE,
            row_count=row_count,
            set_kind=set_kind,
            scope_fingerprint=scope_fingerprint,
            pagination=PaginationCompleteness.TRUNCATED,
            proof_refs=proof_refs,
        )
    if page_cap_reached:
        return _api_proof(
            status=CompletenessStatus.INCOMPLETE,
            row_count=row_count,
            set_kind=set_kind,
            scope_fingerprint=scope_fingerprint,
            pagination=PaginationCompleteness.PAGE_CAP_REACHED,
            proof_refs=proof_refs,
        )

    pagination = read.pagination
    if pagination is None or pagination.mode == PaginationMode.NONE:
        complete = _unpaginated_read_is_complete(read)
        return _api_proof(
            status=(
                CompletenessStatus.COMPLETE if complete else CompletenessStatus.UNKNOWN
            ),
            row_count=row_count,
            set_kind=set_kind,
            scope_fingerprint=scope_fingerprint,
            pagination=PaginationCompleteness.NOT_PAGINATED,
            proof_refs=proof_refs,
        )

    if reached_terminal_page:
        return _api_proof(
            status=CompletenessStatus.COMPLETE,
            row_count=row_count,
            set_kind=set_kind,
            scope_fingerprint=scope_fingerprint,
            pagination=PaginationCompleteness.TERMINAL,
            proof_refs=proof_refs,
        )
    return _api_proof(
        status=CompletenessStatus.UNKNOWN,
        row_count=row_count,
        set_kind=set_kind,
        scope_fingerprint=scope_fingerprint,
        pagination=PaginationCompleteness.UNKNOWN,
        proof_refs=proof_refs,
    )


def _unpaginated_read_is_complete(read: EndpointRead) -> bool:
    pagination = read.pagination
    if pagination is None:
        return True
    return pagination.completeness_policy in {
        CompletenessPolicy.COMPLETE,
        CompletenessPolicy.BOUNDED,
    }


def _api_proof(
    *,
    status: CompletenessStatus,
    row_count: int,
    set_kind: RelationSetKind,
    scope_fingerprint: str,
    pagination: PaginationCompleteness,
    proof_refs: tuple[str, ...],
) -> CompletenessProof:
    return CompletenessProof(
        status=status,
        source_kind=CompletenessSourceKind.API_READ,
        set_kind=set_kind,
        scope_fingerprint=scope_fingerprint,
        proof_refs=proof_refs,
        row_count=row_count,
        pagination=pagination,
    )
