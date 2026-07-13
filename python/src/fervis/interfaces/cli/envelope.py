"""Shared agent command envelope for Fervis CLI output."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.project import ProjectInspection


COMMAND_RESULT_SCHEMA = "fervis-command-result.v0.1"


@dataclass(frozen=True)
class CommandEnvelope:
    command: str
    status: str
    exit_code: int
    project: dict[str, object]
    payload_schema: str
    payload: dict[str, object]
    next_actions: list[dict[str, object]] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": COMMAND_RESULT_SCHEMA,
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "project": self.project,
            "next_actions": self.next_actions,
            "payload_schema": self.payload_schema,
            "payload": self.payload,
        }


def command_envelope(
    *,
    command: str,
    status: str,
    exit_code: int,
    project: ProjectInspection,
    payload_schema: str,
    payload: dict[str, object],
    next_actions: list[dict[str, object]] | None = None,
) -> CommandEnvelope:
    return CommandEnvelope(
        command=command,
        status=status,
        exit_code=exit_code,
        project=project.to_envelope_project(),
        next_actions=list(next_actions or []),
        payload_schema=payload_schema,
        payload=payload,
    )


def status_for_exit_code(exit_code: int) -> str:
    if exit_code == 0:
        return "succeeded"
    if exit_code == 2:
        return "blocked"
    return "failed"
