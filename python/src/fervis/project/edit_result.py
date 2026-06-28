"""Shared project file-edit result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockedEdit:
    file: str
    reason: str


@dataclass(frozen=True)
class ProjectEditResult:
    changed_files: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    blocked_edits: list[BlockedEdit] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_edits)

    def to_payload(self) -> dict[str, object]:
        return {
            "changed_files": self.changed_files,
            "skipped_existing": self.skipped_existing,
            "blocked_edits": [
                {"file": item.file, "reason": item.reason}
                for item in self.blocked_edits
            ],
        }


def blocked_edit(*, file: str, reason: str) -> ProjectEditResult:
    return ProjectEditResult(blocked_edits=[BlockedEdit(file=file, reason=reason)])
