"""Adjacent execution issues and undefined-operation signals."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum

from fervis.lookup.outcomes.model import (
    UndefinedOperationRef,
    UndefinedReasonCode,
)


class ExecutionIssueKind(StrEnum):
    EXECUTION_FAILURE = "execution_failure"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    VALIDATION_ERROR = "validation_error"
    PROVIDER_ERROR = "provider_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


@dataclass(frozen=True)
class ExecutionIssue:
    kind: ExecutionIssueKind
    message: str
    relation_id: str = ""
    proof_refs: tuple[str, ...] = ()


class IncompleteEvidenceError(Exception):
    def __init__(
        self,
        *,
        relation_id: str,
        proof_refs: tuple[str, ...] = (),
    ) -> None:
        super().__init__("incomplete evidence")
        self.relation_id = relation_id
        self.proof_refs = proof_refs

    def issue(self) -> ExecutionIssue:
        return ExecutionIssue(
            kind=ExecutionIssueKind.INCOMPLETE_EVIDENCE,
            message="relation evidence is incomplete",
            relation_id=self.relation_id,
            proof_refs=self.proof_refs,
        )


class UndefinedOperationError(Exception):
    def __init__(
        self,
        *,
        reason_code: UndefinedReasonCode,
        input_refs: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason_code.value)
        self.reason_code = reason_code
        self.input_refs = input_refs

    def operation_ref(
        self,
        operation_id: str,
        *,
        proof_refs: tuple[str, ...] = (),
    ) -> UndefinedOperationRef:
        return UndefinedOperationRef(
            operation_id=operation_id,
            reason_code=self.reason_code,
            input_refs=self.input_refs,
            proof_refs=proof_refs,
        )
