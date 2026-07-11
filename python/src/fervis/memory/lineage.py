"""Memory artifacts persisted by answer lineage."""

from __future__ import annotations

from typing import Any, Protocol

from fervis.lineage.enums import MemoryArtifactSourceKind
from fervis.lineage.memory_artifacts import MemoryArtifactRow
from fervis.memory.addresses import fact_address_from_payload
from fervis.memory.artifacts import (
    FactArtifact,
    FactOutcome,
    build_fact_artifact,
)

DEFAULT_RECENT_MEMORY_RUN_LIMIT = 32
MEMORY_ARTIFACT_SCHEMA = "fervis.memory_artifact"
MEMORY_ARTIFACT_SCHEMA_REV = 1


class MemoryArtifactQueryPort(Protocol):
    def memory_artifact_rows_for_run_ids(
        self,
        run_ids: tuple[str, ...],
    ) -> tuple[MemoryArtifactRow, ...]: ...


class LineageMemoryArtifactService:
    """Projects canonical persisted memory artifacts into planner memory."""

    def __init__(self, query: MemoryArtifactQueryPort) -> None:
        self._query = query

    def for_runs(
        self,
        run_ids: tuple[str, ...],
    ) -> tuple[FactArtifact, ...]:
        return fact_artifacts_from_memory_rows(
            self._query.memory_artifact_rows_for_run_ids(run_ids)
        )


def fact_artifacts_from_memory_rows(
    rows: tuple[MemoryArtifactRow, ...],
) -> tuple[FactArtifact, ...]:
    return tuple(_fact_artifact_from_memory_row(row) for row in rows)


def memory_artifact_payload(
    *,
    artifact: FactArtifact,
    source_kind: MemoryArtifactSourceKind,
) -> dict[str, Any]:
    return {"sourceKind": source_kind.value, **artifact.to_dict()}


def _fact_artifact_from_memory_row(row: MemoryArtifactRow) -> FactArtifact:
    _validate_memory_payload_version(row)
    _validate_memory_payload_identity(row)
    return _fact_artifact_from_payload(row.payload_json, source_kind=row.source_kind)


def _validate_memory_payload_version(row: MemoryArtifactRow) -> None:
    if row.payload_schema != MEMORY_ARTIFACT_SCHEMA:
        raise ValueError(
            f"memory artifact {row.memory_artifact_id!r} has unsupported schema"
        )
    if row.payload_schema_rev != MEMORY_ARTIFACT_SCHEMA_REV:
        raise ValueError(
            f"memory artifact {row.memory_artifact_id!r} has unsupported schema revision"
        )


def _validate_memory_payload_identity(row: MemoryArtifactRow) -> None:
    if row.payload_json.get("artifactId") != row.memory_artifact_id:
        raise ValueError(
            f"memory artifact {row.memory_artifact_id!r} payload artifactId mismatch"
        )


def _fact_artifact_from_payload(
    payload: dict[str, Any],
    *,
    source_kind: MemoryArtifactSourceKind,
) -> FactArtifact:
    payload_source_kind = MemoryArtifactSourceKind(str(payload.get("sourceKind") or ""))
    if payload_source_kind is not source_kind:
        raise ValueError("memory artifact source kind mismatch")
    outcome = payload.get("outcome")
    if not outcome:
        raise ValueError("memory artifact requires outcome")
    artifact = build_fact_artifact(
        artifact_id=str(payload.get("artifactId") or ""),
        outcome=FactOutcome(str(outcome)),
        addresses=tuple(
            fact_address_from_payload(item) for item in payload.get("addresses") or ()
        ),
        provenance=dict(payload.get("provenance") or {}),
        source_question=str(payload.get("sourceQuestion") or ""),
        source_answer=str(payload.get("sourceAnswer") or ""),
    )
    _validate_source_payload(source_kind=source_kind, artifact=artifact)
    return artifact


def _validate_source_payload(
    *,
    source_kind: MemoryArtifactSourceKind,
    artifact: FactArtifact,
) -> None:
    if source_kind is MemoryArtifactSourceKind.REQUESTED_FACT:
        if artifact.addresses:
            raise ValueError("requested_fact memory artifacts cannot carry addresses")
        return
    if not artifact.addresses:
        raise ValueError(f"{source_kind.value} memory artifacts require addresses")
