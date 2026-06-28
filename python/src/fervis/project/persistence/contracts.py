"""Persistence contracts for Fervis project checks and migrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection


class MigrationStatus(StrEnum):
    APPLIED = "applied"
    BLOCKED = "blocked"
    FAILED = "failed"
    PENDING = "pending"
    UP_TO_DATE = "up_to_date"


@dataclass(frozen=True)
class ResolvedPersistenceTarget:
    kind: str
    location: str


@dataclass(frozen=True)
class PersistenceCheck:
    id: str
    passed: bool
    message: str
    fix: dict[str, object] | None = None


@dataclass(frozen=True)
class MigrationResult:
    target: ResolvedPersistenceTarget
    status: MigrationStatus
    current_revision: str | None
    target_revision: str
    pending_revisions: list[str] = field(default_factory=list)
    applied_revisions: list[str] = field(default_factory=list)
    already_applied: bool = False
    error: str | None = None

    @property
    def exit_code(self) -> int:
        return (
            0
            if self.status in {MigrationStatus.APPLIED, MigrationStatus.UP_TO_DATE}
            else 2
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "target": self.target.kind,
            "location": self.target.location,
            "status": self.status.value,
            "current_revision": self.current_revision,
            "target_revision": self.target_revision,
            "pending_revisions": self.pending_revisions,
            "applied_revisions": self.applied_revisions,
            "already_applied": self.already_applied,
        }
        if self.error:
            payload["error"] = self.error
        return payload


class PersistenceBackend(Protocol):
    target: ResolvedPersistenceTarget
    target_revision: str

    def inspect(self) -> list[PersistenceCheck]: ...

    def migrate(self) -> MigrationResult: ...


@dataclass(frozen=True)
class PersistenceRequest:
    project: ProjectInspection
    loaded_config: LoadedFervisConfig
    project_root: Path
