"""Terminal-result assembly for lookup runtime."""

from typing import Any

from fervis.lookup.errors import ErrorCode
from fervis.lookup.grounding.model import (
    GroundingCandidate,
    GroundingIssue,
    GroundingTerminalKind,
)
from fervis.lookup.outcomes.model import (
    FactResult,
    NeedsClarification,
    OutcomeKind,
)
from fervis.lookup.clarification import (
    Clarification,
    ClarificationEvidence,
    ClarificationEvidenceKind,
    ClarificationOption,
    MissingAnswerMetric,
    MissingCatalogChoice,
    MissingCatalogRequiredValue,
    TargetReferenceAmbiguous,
    TargetReferenceNotFound,
    TargetReferenceUnsupported,
    clarify,
)
from fervis.lookup.clarification.model import (
    CatalogInputTarget,
    ClarificationOwner,
    FactPlanningCatalogInputContinuation,
    SourceBindingCatalogInputContinuation,
)
from fervis.lookup.fact_plan.row_sources import build_row_source_catalog
from fervis.lookup.fact_plan.fact_plan import (
    MissingCatalogChoiceInput,
    MissingCatalogRequiredInput,
    PlanClarification,
)
from fervis.lookup.fact_planning.required_inputs import (
    clarifiable_required_inputs,
)
from fervis.lookup.fact_plan.row_sources import required_input_evidence_ref
from fervis.lookup.question_contract import (
    IncompleteFactualRequestKind,
    QuestionContractNeedsClarification,
)
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupRuntimePorts,
)
from fervis.lookup.orchestration.result import LookupResult, RunStatus
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.observability.event_contracts import EventPayloadKey

from fervis.lookup.lineage.results import record_runtime_error_lineage
from fervis.lookup.lineage.steps import (
    lineage_error_json,
    record_execution_step,
)


def _grounding_issue_fact_result(
    issues: tuple[GroundingIssue, ...],
) -> FactResult:
    clarifications = tuple(
        clarify(_grounding_issue_cause(group))
        for group in _grounding_issue_groups(issues)
    )
    return FactResult(outcome=NeedsClarification(clarifications=clarifications))


def _grounding_issue_groups(
    issues: tuple[GroundingIssue, ...],
) -> tuple[tuple[GroundingIssue, ...], ...]:
    grouped: dict[tuple[str, str], list[GroundingIssue]] = {}
    for issue in issues:
        grouped.setdefault(
            (issue.requested_fact_id, issue.known_input_id),
            [],
        ).append(issue)
    return tuple(tuple(group) for group in grouped.values())


def _grounding_issue_proof_refs(issue: GroundingIssue) -> tuple[str, ...]:
    return (*issue.proof_refs, f"grounding:{issue.kind.value}")


def _grounding_issue_cause(
    issues: tuple[GroundingIssue, ...],
) -> TargetReferenceNotFound | TargetReferenceAmbiguous | TargetReferenceUnsupported:
    issue = issues[0]
    clarification_id = f"clarify_{issue.known_input_id}_grounding"
    source_text = str(issue.known_input_text or "")
    target_label = _grounding_issue_target_label(issue)
    evidence = tuple(
        dict.fromkeys(
            evidence
            for grouped_issue in issues
            for evidence in _grounding_issue_structured_evidence(grouped_issue)
        )
    )
    proof_refs = tuple(
        dict.fromkeys(
            proof_ref
            for grouped_issue in issues
            for proof_ref in _grounding_issue_proof_refs(grouped_issue)
        )
    )
    ambiguous = tuple(
        grouped_issue
        for grouped_issue in issues
        if grouped_issue.kind == GroundingTerminalKind.AMBIGUOUS_REFERENCE
    )
    if ambiguous:
        return TargetReferenceAmbiguous(
            clarification_id=clarification_id,
            requested_fact_id=issue.requested_fact_id,
            known_input_id=issue.known_input_id,
            source_text=source_text,
            target_label=target_label,
            evidence=evidence,
            proof_refs=proof_refs,
            options=tuple(
                dict.fromkeys(
                    _clarification_option_from_grounding_candidate(candidate)
                    for grouped_issue in ambiguous
                    for candidate in grouped_issue.candidate_options
                )
            ),
        )
    if any(
        grouped_issue.kind == GroundingTerminalKind.UNRESOLVED_REFERENCE
        for grouped_issue in issues
    ):
        return TargetReferenceNotFound(
            clarification_id=clarification_id,
            requested_fact_id=issue.requested_fact_id,
            known_input_id=issue.known_input_id,
            source_text=source_text,
            target_label=target_label,
            evidence=evidence,
            proof_refs=proof_refs,
        )
    return TargetReferenceUnsupported(
        clarification_id=clarification_id,
        requested_fact_id=issue.requested_fact_id,
        known_input_id=issue.known_input_id,
        source_text=source_text,
        target_label=target_label,
        evidence=evidence,
        proof_refs=proof_refs,
    )


def _grounding_issue_target_label(issue: GroundingIssue) -> str:
    if issue.kind == GroundingTerminalKind.TIME_RESOLUTION_FAILED:
        return "date range"
    return str(issue.known_input_description or "entity").strip().lower()


def _grounding_issue_structured_evidence(
    issue: GroundingIssue,
) -> tuple[ClarificationEvidence, ...]:
    if not issue.resolver_read_id:
        return ()
    return (
        ClarificationEvidence(
            kind=ClarificationEvidenceKind.RESOLVER_READ,
            id=f"read:{issue.resolver_read_id}",
            read_id=issue.resolver_read_id,
            endpoint_name=issue.resolver_endpoint_name,
            field_id=issue.resolver_field_id,
            identity_field=issue.identity_field,
        ),
    )


def _clarification_option_from_grounding_candidate(
    candidate: GroundingCandidate,
) -> ClarificationOption:
    return ClarificationOption(
        id=candidate.id,
        label=candidate.label,
        value=candidate.matched_value,
        key=candidate.key,
        matched_label=candidate.matched_label,
        matched_field=candidate.matched_field,
        matched_value=candidate.matched_value,
        resolver_read_id=candidate.resolver_read_id,
        resolver_label=candidate.resolver_label,
    )


def _question_contract_clarification_fact_result(
    outcome: QuestionContractNeedsClarification,
) -> FactResult:
    clarifications: list[Clarification] = []
    for index, item in enumerate(outcome.missing, start=1):
        if (
            item.missing_kind
            is IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE
        ):
            known_input_id = f"question_contract:{item.source_text}"
            clarifications.append(
                clarify(
                    TargetReferenceNotFound(
                        clarification_id=f"clarify_question_contract_{index}",
                        requested_fact_id="question_contract",
                        known_input_id=known_input_id,
                        source_text=item.source_text,
                        target_label=item.target_label,
                        proof_refs=(
                            f"known_input:{known_input_id}",
                            "question_contract:needs_clarification",
                        ),
                    )
                )
            )
            continue
        clarifications.append(
            clarify(
                MissingAnswerMetric(
                    clarification_id=f"clarify_question_contract_{index}",
                    requested_fact_id="question_contract",
                    source_text=item.source_text,
                    metric_needed=item.why_question_is_incomplete,
                    proof_refs=(
                        "requested_fact:question_contract",
                        "question_contract:needs_clarification",
                    ),
                )
            )
        )
    return FactResult(outcome=NeedsClarification(clarifications=tuple(clarifications)))


def _plan_clarification_fact_result(
    outcome: PlanClarification,
    *,
    owner: ClarificationOwner,
    catalog,
    memory_relations,
) -> FactResult:
    row_sources = build_row_source_catalog(catalog, memory_relations=memory_relations)
    required_inputs = {
        item.id: item for item in clarifiable_required_inputs(row_sources)
    }
    clarifications: list[Clarification] = []
    for item in outcome.missing_catalog_inputs:
        if isinstance(item, MissingCatalogRequiredInput):
            required = required_inputs[item.required_catalog_input_id]
            clarifications.append(
                clarify(
                    MissingCatalogRequiredValue(
                        clarification_id=item.id,
                        requested_fact_id=item.requested_fact_id,
                        required_input_id=item.required_catalog_input_id,
                        label=required.param_label or required.param_id,
                        continuation=_catalog_continuation(
                            owner=owner,
                            requested_fact_id=item.requested_fact_id,
                            planning_requirement_id=item.id,
                            required=required,
                        ),
                        proof_refs=(
                            required_input_evidence_ref(
                                required_input_id=item.required_catalog_input_id,
                            ),
                        ),
                    )
                )
            )
            continue
        if isinstance(item, MissingCatalogChoiceInput):
            required = required_inputs[item.required_catalog_choice_input_id]
            choice_labels = dict(required.choice_labels or {})
            clarifications.append(
                clarify(
                    MissingCatalogChoice(
                        clarification_id=item.id,
                        requested_fact_id=item.requested_fact_id,
                        required_choice_input_id=item.required_catalog_choice_input_id,
                        label=required.param_label or required.param_id,
                        options=tuple(
                            ClarificationOption(
                                id=choice,
                                label=choice_labels.get(choice, choice),
                            )
                            for choice in required.choices
                        ),
                        continuation=_catalog_continuation(
                            owner=owner,
                            requested_fact_id=item.requested_fact_id,
                            planning_requirement_id=item.id,
                            required=required,
                        ),
                        proof_refs=(
                            required_input_evidence_ref(
                                required_input_id=item.required_catalog_choice_input_id,
                            ),
                        ),
                    )
                )
            )
    return FactResult(outcome=NeedsClarification(clarifications=tuple(clarifications)))


def _catalog_continuation(
    *,
    owner: ClarificationOwner,
    requested_fact_id: str,
    planning_requirement_id: str,
    required,
) -> SourceBindingCatalogInputContinuation | FactPlanningCatalogInputContinuation:
    target = CatalogInputTarget(
        row_source_id=required.row_source_id,
        param_id=required.param_id,
        param_ref=required.param_ref,
        value_type=required.param_type,
        choices=tuple(required.choices),
    )
    if owner is ClarificationOwner.SOURCE_BINDING:
        return SourceBindingCatalogInputContinuation(
            requested_fact_id=requested_fact_id,
            target=target,
        )
    if owner is ClarificationOwner.FACT_PLANNING:
        return FactPlanningCatalogInputContinuation(
            requested_fact_id=requested_fact_id,
            planning_requirement_id=planning_requirement_id,
            target=target,
        )
    raise ValueError("catalog clarification requires source-binding or fact-planning owner")


def _plan_validation_failed_result(
    *,
    request: LookupRequest,
    ports: LookupRuntimePorts,
    usage: dict[str, Any],
    exc: VerificationError,
) -> LookupResult:
    payload = _execution_failure_payload(
        request=request,
        error_code=ErrorCode.PLAN_VALIDATION_FAILED,
        exc=exc,
    )
    failed_step = record_execution_step(
        ports,
        error_json=lineage_error_json(payload),
    )
    record_runtime_error_lineage(
        request=request,
        ports=ports,
        failed_step_id=failed_step.step_id if failed_step is not None else None,
        error_code=ErrorCode.PLAN_VALIDATION_FAILED,
        message=str(exc),
    )
    return LookupResult(
        status=RunStatus.FAILED,
        error=ErrorCode.PLAN_VALIDATION_FAILED,
        usage=usage,
    )


def _execution_failure_payload(
    *,
    request: LookupRequest,
    error_code: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        EventPayloadKey.RUN_ID: request.run_id,
        EventPayloadKey.ERROR_CODE: error_code,
        EventPayloadKey.ERROR_CLASS: exc.__class__.__name__,
        EventPayloadKey.ERROR_CONTEXT: str(exc),
    }


def _status_for_fact_result(result: FactResult) -> str:
    if result.outcome.kind == OutcomeKind.NEEDS_CLARIFICATION:
        return RunStatus.NEEDS_CLARIFICATION
    return RunStatus.COMPLETED
