"""Presentation adapters for run-step summary payloads."""

from __future__ import annotations

from fervis.lineage.step_summary import (
    StepSummaryDetail,
    StepSemanticItem,
    step_semantic_items_from_json,
    step_summary_items_from_json,
)
from fervis.lineage.views.model import (
    SemanticConversationClauseView,
    SemanticIdentitySelectionView,
    SemanticInterpretedInputView,
    SemanticKnownInputView,
    SemanticRequestedFactView,
    SemanticResourceRecallView,
    SemanticResolverCandidateView,
    StepDecisionItemView,
    StepDecisionView,
    StepSemanticView,
)
from fervis.lineage.views.query import StepRow


def step_decision_views(step: StepRow) -> tuple[StepDecisionView, ...]:
    decisions = []
    for detail in StepSummaryDetail:
        items = tuple(
            item
            for item in step_summary_items_from_json(step.output_summary_json)
            if item.detail is detail
        )
        if items:
            decisions.append(
                StepDecisionView(
                    step_key=step.step_key.value,
                    lines=tuple(item.text for item in items),
                    detail=detail,
                    is_explanation=any(item.is_explanation for item in items),
                    items=tuple(
                        StepDecisionItemView(
                            text=item.text,
                            is_explanation=item.is_explanation,
                            path=item.path,
                            subject=item.subject,
                            disposition=item.disposition,
                            basis=item.basis,
                        )
                        for item in items
                    ),
                )
            )
    return tuple(decisions)


def step_semantic_view(step: StepRow) -> StepSemanticView:
    requested_facts: list[SemanticRequestedFactView] = []
    known_inputs: list[SemanticKnownInputView] = []
    resource_recalls: list[SemanticResourceRecallView] = []
    resolver_candidates: list[SemanticResolverCandidateView] = []
    identity_selections: list[SemanticIdentitySelectionView] = []
    interpreted_inputs: list[SemanticInterpretedInputView] = []
    conversation_clauses: list[SemanticConversationClauseView] = []
    for item in step_semantic_items_from_json(step.output_summary_json):
        if item.kind == "requested_fact":
            requested_fact = _requested_fact(item)
            if requested_fact is not None:
                requested_facts.append(requested_fact)
        elif item.kind == "known_input":
            known_input = _known_input(item)
            if known_input is not None:
                known_inputs.append(known_input)
        elif item.kind == "resource_recall":
            resource_recall = _resource_recall(item)
            if resource_recall is not None:
                resource_recalls.append(resource_recall)
        elif item.kind == "resolver_candidate":
            resolver_candidate = _resolver_candidate(item)
            if resolver_candidate is not None:
                resolver_candidates.append(resolver_candidate)
        elif item.kind == "identity_selection":
            identity_selection = _identity_selection(item)
            if identity_selection is not None:
                identity_selections.append(identity_selection)
        elif item.kind == "interpreted_input":
            interpreted_input = _interpreted_input(item)
            if interpreted_input is not None:
                interpreted_inputs.append(interpreted_input)
        elif item.kind == "conversation_clause":
            conversation_clause = _conversation_clause(item)
            if conversation_clause is not None:
                conversation_clauses.append(conversation_clause)
    return StepSemanticView(
        requested_facts=tuple(requested_facts),
        known_inputs=tuple(known_inputs),
        resource_recalls=tuple(resource_recalls),
        resolver_candidates=tuple(resolver_candidates),
        identity_selections=tuple(identity_selections),
        interpreted_inputs=tuple(interpreted_inputs),
        conversation_clauses=tuple(conversation_clauses),
    )


def _requested_fact(item: StepSemanticItem) -> SemanticRequestedFactView | None:
    requested_fact_id = _text(item.payload.get("requested_fact_id"))
    description = _text(item.payload.get("description"))
    if not requested_fact_id or not description:
        return None
    return SemanticRequestedFactView(
        requested_fact_id=requested_fact_id,
        description=description,
    )


def _known_input(item: StepSemanticItem) -> SemanticKnownInputView | None:
    input_id = _text(item.payload.get("input_id"))
    text = _text(item.payload.get("text"))
    if not input_id or not text:
        return None
    return SemanticKnownInputView(
        input_id=input_id,
        text=text,
        kind=_text(item.payload.get("kind")),
        role=_text(item.payload.get("role")),
        description=_text(item.payload.get("description")),
        resolved_value_text=_text(item.payload.get("resolved_value_text")),
    )


def _resolver_candidate(
    item: StepSemanticItem,
) -> SemanticResolverCandidateView | None:
    input_id = _text(item.payload.get("input_id"))
    basis = _text(item.payload.get("basis"))
    resolver_label = _text(item.payload.get("resolver_label"))
    resolver_read_id = _text(item.payload.get("resolver_read_id"))
    if not input_id or not (basis or resolver_label or resolver_read_id):
        return None
    return SemanticResolverCandidateView(
        input_id=input_id,
        resolver_read_id=resolver_read_id,
        resolver_label=resolver_label,
        basis=basis,
    )


def _resource_recall(item: StepSemanticItem) -> SemanticResourceRecallView | None:
    input_use_ref = _text(item.payload.get("input_use_ref"))
    resource_name = _text(item.payload.get("resource_name"))
    if not input_use_ref or not resource_name:
        return None
    return SemanticResourceRecallView(
        input_use_ref=input_use_ref,
        resource_name=resource_name,
    )


def _identity_selection(
    item: StepSemanticItem,
) -> SemanticIdentitySelectionView | None:
    input_id = _text(item.payload.get("input_id"))
    canonical_option_id = _text(item.payload.get("canonical_option_id"))
    resolver_route_id = _text(item.payload.get("resolver_route_id"))
    outcome = _text(item.payload.get("outcome"))
    if not input_id or not (canonical_option_id or resolver_route_id or outcome):
        return None
    return SemanticIdentitySelectionView(
        input_id=input_id,
        canonical_option_id=canonical_option_id,
        resolver_route_id=resolver_route_id,
        basis=_text(item.payload.get("basis")),
        outcome=outcome,
    )


def _interpreted_input(item: StepSemanticItem) -> SemanticInterpretedInputView | None:
    input_id = _text(item.payload.get("input_id"))
    kind = _text(item.payload.get("kind"))
    value = _text(item.payload.get("value"))
    if not input_id or not kind or not value:
        return None
    return SemanticInterpretedInputView(
        input_id=input_id,
        input_text=_text(item.payload.get("input_text")),
        kind=kind,
        value=value,
        label=_text(item.payload.get("label")),
        detail=_text(item.payload.get("detail")),
    )


def _conversation_clause(
    item: StepSemanticItem,
) -> SemanticConversationClauseView | None:
    current_clause_text = _text(item.payload.get("current_clause_text"))
    resolved_text = _text(item.payload.get("resolved_text"))
    if not current_clause_text and not resolved_text:
        return None
    raw_values = item.payload.get("resolved_values")
    resolved_values = raw_values if isinstance(raw_values, (list, tuple)) else ()
    return SemanticConversationClauseView(
        current_clause_text=current_clause_text,
        resolved_text=resolved_text,
        resolved_values=tuple(_text(value) for value in resolved_values),
    )


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""
