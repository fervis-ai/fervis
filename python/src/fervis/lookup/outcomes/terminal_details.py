"""Structured terminal-result projections."""

from __future__ import annotations

from typing import Mapping

from fervis.lookup.clarification import clarifications_payload
from fervis.lookup.outcomes.model import (
    BlockedRequirement,
    EmptyRelation,
    FactResult,
    Impossible,
    NeedsClarification,
    NoData,
    Undefined,
    UndefinedOperationRef,
)

TERMINAL_OUTCOME_PAYLOAD_KEY = "terminalOutcome"
TERMINAL_KIND_KEY = "kind"
TERMINAL_EMPTY_RELATION_KEY = "emptyRelation"
TERMINAL_OPERATION_KEY = "operation"
TERMINAL_CLARIFICATIONS_KEY = "clarifications"
TERMINAL_BLOCKED_REQUIREMENTS_KEY = "blockedRequirements"
TERMINAL_REQUIRED_FOR_KEY = "requiredFor"
TERMINAL_RELATION_ID_KEY = "relationId"
TERMINAL_REASON_CODE_KEY = "reasonCode"
TERMINAL_REQUESTED_FACT_ID_KEY = "requestedFactId"
TERMINAL_FACT_REF_KEY = "factRef"


def fact_result_terminal_details(
    result: FactResult,
) -> Mapping[str, object] | None:
    outcome = result.outcome
    if isinstance(outcome, NoData):
        return {
            TERMINAL_EMPTY_RELATION_KEY: empty_relation_payload(outcome.empty_relation)
        }
    if isinstance(outcome, Undefined):
        return {TERMINAL_OPERATION_KEY: undefined_operation_payload(outcome.operation)}
    if isinstance(outcome, NeedsClarification):
        return needs_clarification_payload(outcome)
    if isinstance(outcome, Impossible):
        return impossible_payload(outcome)
    return None


def empty_relation_payload(empty: EmptyRelation) -> dict[str, object]:
    return {
        "kind": empty.kind.value,
        TERMINAL_RELATION_ID_KEY: empty.relation_id,
        "grainKeys": list(empty.grain_keys),
        "requestedFactIds": list(empty.requested_fact_ids),
        "scopeRef": empty.scope_ref,
        "proofRefs": list(empty.proof_refs),
    }


def undefined_operation_payload(operation: UndefinedOperationRef) -> dict[str, object]:
    return {
        "operationId": operation.operation_id,
        TERMINAL_REASON_CODE_KEY: operation.reason_code.value,
        "inputRefs": list(operation.input_refs),
        "proofRefs": list(operation.proof_refs),
    }


def needs_clarification_payload(outcome: NeedsClarification) -> dict[str, object]:
    return clarifications_payload(outcome.clarifications)


def impossible_payload(outcome: Impossible) -> dict[str, object]:
    return {
        TERMINAL_BLOCKED_REQUIREMENTS_KEY: [
            blocked_requirement_payload(item) for item in outcome.blocked_requirements
        ]
    }


def blocked_requirement_payload(
    requirement: BlockedRequirement,
) -> dict[str, object]:
    return {
        "kind": requirement.kind.value,
        "requestedFactId": requirement.requested_fact_id,
        TERMINAL_REQUIRED_FOR_KEY: requirement.required_for,
        TERMINAL_FACT_REF_KEY: requirement.fact_ref,
        "reviewedReadIds": list(requirement.reviewed_read_ids),
        "nearestFields": [
            {
                "readId": field.read_id,
                "fieldId": field.field_id,
            }
            for field in requirement.nearest_fields
        ],
        "proofRefs": list(requirement.proof_refs),
    }
