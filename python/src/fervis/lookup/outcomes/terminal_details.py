"""Structured terminal-result projections."""

from __future__ import annotations

from typing import Mapping

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
        return clarification_payload(outcome)
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


def clarification_payload(outcome: NeedsClarification) -> dict[str, object]:
    return {
        TERMINAL_CLARIFICATIONS_KEY: [
            clarification_item_payload(item) for item in outcome.clarifications
        ],
    }


def clarification_item_payload(item) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": item.id,
        TERMINAL_REQUESTED_FACT_ID_KEY: item.requested_fact_id,
        "basis": item.basis.value,
        "question": item.question,
    }
    if item.known_input_id:
        payload["knownInputId"] = item.known_input_id
    if item.required_catalog_input_id:
        payload["requiredCatalogInputId"] = item.required_catalog_input_id
    if item.required_catalog_choice_input_id:
        payload["requiredCatalogChoiceInputId"] = item.required_catalog_choice_input_id
    if item.available_options:
        payload["availableOptions"] = [
            {"id": option.id, "label": option.label}
            for option in item.available_options
        ]
    if item.candidate_refs:
        payload["candidateRefs"] = list(item.candidate_refs)
    if item.evidence_refs:
        payload["evidenceRefs"] = list(item.evidence_refs)
    if item.ambiguous_metric_phrase:
        payload["ambiguousMetricPhrase"] = item.ambiguous_metric_phrase
    if item.metric_needed_to_answer:
        payload["metricNeededToAnswer"] = item.metric_needed_to_answer
    if item.comparison_phrase:
        payload["comparisonPhrase"] = item.comparison_phrase
    if item.comparison_baseline_needed:
        payload["comparisonBaselineNeeded"] = item.comparison_baseline_needed
    return payload


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
