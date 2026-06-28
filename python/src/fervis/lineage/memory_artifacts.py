"""Shared lineage memory artifact row contract."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lineage.enums import MemoryArtifactSourceKind

JsonObject = dict[str, object]


@dataclass(frozen=True)
class MemoryArtifactRow:
    memory_artifact_id: str
    run_id: str
    produced_by_step_id: str
    source_kind: MemoryArtifactSourceKind
    payload_schema: str
    payload_schema_rev: int
    payload_json: JsonObject
    requested_fact_id: str | None = None
    fact_result_id: str | None = None
