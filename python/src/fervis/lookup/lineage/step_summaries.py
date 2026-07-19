"""Stable lineage step summaries projected from model-turn payloads."""

from __future__ import annotations

from typing import Any

from fervis.lineage.step_summary import (
    StepSummaryDetail,
    StepSummaryItem,
    StepSemanticItem,
    merge_step_semantic_json,
    merge_step_summary_json,
    step_summary_json,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralValuePayload,
    NamedValuePayload,
    TimeValuePayload,
)
from fervis.lookup.fact_planning.lineage_summary import (
    fact_planning_step_summary,
)
from fervis.lookup.grounding.model import CanonicalInputLedger
from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_items,
    lineage_explanation_paths_from_payload,
)
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.source_binding.lineage_summary import (
    source_binding_step_summary,
)
from fervis.model_io.turns import ModelTurnPurpose
from fervis.observability.event_contracts import EventPayloadKey


def model_turn_output_summary(payload: dict[str, Any]) -> dict[str, object]:
    purpose = str(payload.get(EventPayloadKey.PURPOSE) or "")
    parsed = _dict_or_empty(payload.get(EventPayloadKey.PARSED_ARGUMENTS))
    submitted = _dict_or_empty(payload.get(EventPayloadKey.ARGUMENTS))
    derived = _dict_or_empty(payload.get(EventPayloadKey.DERIVED_ARGUMENTS))
    source = _summary_source(purpose=purpose, parsed=parsed, submitted=submitted)
    turn_summary = _turn_summary(purpose=purpose, source=source)
    return merge_step_summary_json(
        merge_step_semantic_json(
            turn_summary,
            *_semantic_items(purpose=purpose, source=source, derived=derived),
        ),
        *_generic_explanation_items(purpose=purpose, source=source, derived=derived),
    )


def add_grounding_result_semantics(
    summary: dict[str, object],
    *,
    ledger: CanonicalInputLedger,
    question_contract: QuestionContract,
) -> dict[str, object]:
    return merge_step_semantic_json(
        summary,
        *_grounding_result_semantic_items(
            ledger=ledger,
            question_contract=question_contract,
        ),
    )


def _summary_source(
    *,
    purpose: str,
    parsed: dict[str, Any],
    submitted: dict[str, Any],
) -> dict[str, Any]:
    if purpose in {ModelTurnPurpose.PATTERN_FACT_PLANNING, ModelTurnPurpose.FACT_PLAN}:
        return submitted or parsed
    return parsed or submitted


def _turn_summary(*, purpose: str, source: dict[str, Any]) -> dict[str, object]:
    if purpose == ModelTurnPurpose.READ_ELIGIBILITY:
        return _read_eligibility_step_summary(source)
    if purpose == ModelTurnPurpose.PLAN_SELECTION:
        return _plan_selection_step_summary(source)
    if purpose == ModelTurnPurpose.SOURCE_BINDING:
        return source_binding_step_summary(source)
    if purpose in {ModelTurnPurpose.PATTERN_FACT_PLANNING, ModelTurnPurpose.FACT_PLAN}:
        return fact_planning_step_summary(source)
    return {}


def _semantic_items(
    *,
    purpose: str,
    source: dict[str, Any],
    derived: dict[str, Any],
) -> tuple[StepSemanticItem, ...]:
    if purpose == ModelTurnPurpose.CONVERSATION_RESOLUTION:
        return _conversation_resolution_semantic_items(source, derived)
    if purpose == ModelTurnPurpose.QUESTION_CONTRACT:
        return _question_contract_semantic_items(source)
    if purpose == ModelTurnPurpose.QUERY_ENRICHMENT:
        return _query_enrichment_semantic_items(source)
    if purpose == ModelTurnPurpose.GROUNDING:
        return _grounding_semantic_items(source)
    return ()


def _conversation_resolution_semantic_items(
    payload: dict[str, Any],
    derived: dict[str, Any],
) -> tuple[StepSemanticItem, ...]:
    del derived
    clauses = _dicts(payload.get("clauses"))
    items: list[StepSemanticItem] = []
    for clause in clauses:
        current_clause_text = _text(clause.get("current_clause_text"))
        resolved_text = _text(clause.get("resolved_text"))
        resolved_values = tuple(
            text
            for value in _dicts(clause.get("values"))
            if (text := _text(value.get("resolved_text")))
        )
        if not current_clause_text and not resolved_text:
            continue
        items.append(
            StepSemanticItem(
                kind="conversation_clause",
                payload={
                    "current_clause_text": current_clause_text,
                    "resolved_text": resolved_text,
                    "resolved_values": resolved_values,
                },
            )
        )
    return tuple(items)


def _generic_explanation_items(
    *,
    purpose: str,
    source: dict[str, Any],
    derived: dict[str, Any],
) -> tuple[StepSummaryItem, ...]:
    if purpose in {
        ModelTurnPurpose.QUESTION_CONTRACT,
        ModelTurnPurpose.QUERY_ENRICHMENT,
        ModelTurnPurpose.GROUNDING,
        ModelTurnPurpose.READ_ELIGIBILITY,
        ModelTurnPurpose.PLAN_SELECTION,
    }:
        return ()
    return lineage_explanation_items(
        source,
        metadata=lineage_explanation_paths_from_payload(derived),
    )


def _dict_or_empty(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _question_contract_semantic_items(
    payload: dict[str, Any],
) -> tuple[StepSemanticItem, ...]:
    items: list[StepSemanticItem] = []
    for index, answer_request in enumerate(
        _dicts(payload.get("answer_requests")), start=1
    ):
        description = _text(answer_request.get("answer_fact"))
        if description:
            items.append(
                StepSemanticItem(
                    kind="requested_fact",
                    payload={
                        "requested_fact_id": f"fact_{index}",
                        "description": description,
                    },
                )
            )
    for raw_input in _dicts(payload.get("question_inputs")):
        input_id = _text(raw_input.get("input_ref") or raw_input.get("id"))
        text = _text(
            raw_input.get("source_text")
            or raw_input.get("reference_text")
            or raw_input.get("text")
        )
        if not input_id or not text:
            continue
        items.append(
            StepSemanticItem(
                kind="known_input",
                payload={
                    "input_id": input_id,
                    "text": text,
                    "kind": _text(raw_input.get("kind")),
                    "role": _text(raw_input.get("role")),
                    "description": _text(
                        raw_input.get("value_meaning_hint")
                        or raw_input.get("description")
                    ),
                    "resolved_value_text": _text(raw_input.get("resolved_value_text")),
                },
            )
        )
    return tuple(items)


def _query_enrichment_semantic_items(
    payload: dict[str, Any],
) -> tuple[StepSemanticItem, ...]:
    return tuple(
        StepSemanticItem(
            kind="resolver_candidate",
            payload={
                "input_id": _text(item.get("target_id")),
                "resolver_read_id": "",
                "resolver_label": _title_words(_text(term.get("term"))),
                "basis": _text(term.get("basis")),
            },
        )
        for item in _dicts(payload.get("entity_target_catalog_search_terms"))
        for term in _dicts(item.get("catalog_search_terms"))
        if _text(item.get("target_id")) and _text(term.get("term"))
    )


def _grounding_semantic_items(payload: dict[str, Any]) -> tuple[StepSemanticItem, ...]:
    reviews = _dict_or_empty(payload.get("known_input_binding_reviews"))
    items: list[StepSemanticItem] = []
    for input_id, raw_review in reviews.items():
        review = _dict_or_empty(raw_review)
        options = _dict_or_empty(review.get("option_reviews"))
        for option_id, raw_option in options.items():
            option = _dict_or_empty(raw_option)
            if _text(option.get("decision")) != "CAN_RESOLVE_LOOKUP_TEXT":
                continue
            items.append(
                StepSemanticItem(
                    kind="resolver_candidate",
                    payload={
                        "input_id": str(input_id),
                        "resolver_read_id": "",
                        "resolver_label": str(option_id),
                        "basis": _text(option.get("because")),
                    },
                )
            )
    return tuple(items)


def _grounding_result_semantic_items(
    *,
    ledger: CanonicalInputLedger,
    question_contract: QuestionContract,
) -> tuple[StepSemanticItem, ...]:
    inputs_by_id = {
        known.id: known
        for fact in question_contract.requested_facts
        for known in fact.known_inputs
    }
    return tuple(
        item
        for value in ledger.values
        if (item := _grounding_result_semantic_item(value, inputs_by_id=inputs_by_id))
        is not None
    )


def _grounding_result_semantic_item(
    value: FactValue,
    *,
    inputs_by_id: dict[str, Any],
) -> StepSemanticItem | None:
    payload = value.payload
    input_id = _known_input_id_from_proof_refs(value.proof_refs)
    if not input_id:
        return None
    known_input = inputs_by_id.get(input_id)
    if isinstance(payload, TimeValuePayload):
        return _interpreted_input_item(
            input_id=input_id,
            input_text=_text(getattr(known_input, "text", "")),
            kind="time",
            value=_time_interpretation_value(payload),
            label=payload.expression or value.label,
            detail=payload.granularity,
        )
    if isinstance(payload, NamedValuePayload):
        return _interpreted_input_item(
            input_id=input_id,
            input_text=_text(getattr(known_input, "text", "")),
            kind="named",
            value=payload.text,
            label=payload.reference_text or value.label,
        )
    if isinstance(payload, LiteralValuePayload):
        return _interpreted_input_item(
            input_id=input_id,
            input_text=_text(getattr(known_input, "text", "")),
            kind=f"literal_{payload.literal_type.value}",
            value=payload.value,
            label=value.label,
        )
    if isinstance(payload, IdentitySetValuePayload):
        return _interpreted_input_item(
            input_id=input_id,
            input_text=_text(getattr(known_input, "text", "")),
            kind="identity_set",
            value=(
                payload.display_value
                or f"{len(payload.keys)} {payload.entity_kind} identities"
            ),
            label=value.label,
            detail=(
                f"{payload.key_id}."
                + "+".join(
                    component.component_id for component in payload.keys[0].components
                )
            ),
        )
    if not isinstance(payload, IdentityValuePayload):
        return None
    resolver_read_id = value.source_refs[0] if value.source_refs else ""
    resolver_endpoint_name = (
        value.source_refs[1] if len(value.source_refs) > 1 else resolver_read_id
    )
    return StepSemanticItem(
        kind="grounding_result",
        payload={
            "input_id": input_id,
            "input_text": _text(getattr(known_input, "text", "")),
            "resolver_read_id": resolver_read_id,
            "resolver_label": _title_words(resolver_read_id or resolver_endpoint_name),
            "entity_kind": payload.entity_kind,
            "key_id": payload.key_id,
            "key_components": [
                {
                    "component_id": component.component_id,
                    "value": str(component.value),
                }
                for component in payload.key.components
            ],
            "matched_label": payload.display_value or value.label,
        },
    )


def _interpreted_input_item(
    *,
    input_id: str,
    input_text: str,
    kind: str,
    value: str,
    label: str = "",
    detail: str = "",
) -> StepSemanticItem | None:
    if not value:
        return None
    return StepSemanticItem(
        kind="interpreted_input",
        payload={
            "input_id": input_id,
            "input_text": input_text,
            "kind": kind,
            "value": value,
            "label": label,
            "detail": detail,
        },
    )


def _time_interpretation_value(payload: TimeValuePayload) -> str:
    if payload.resolved_start and payload.resolved_end:
        return f"{payload.resolved_start} to {payload.resolved_end}"
    return payload.resolved_start or payload.resolved_end or payload.expression


def _known_input_id_from_proof_refs(proof_refs: tuple[str, ...]) -> str:
    for ref in proof_refs:
        prefix, separator, value = ref.partition(":")
        if prefix == "known_input" and separator and value:
            return value
    return ""


def _read_eligibility_step_summary(payload: dict[str, Any]) -> dict[str, object]:
    reviews = _read_eligibility_reviews(payload)
    if not reviews:
        return {}
    retained = sum(1 for review in reviews if _review_decision(review) == "RETAIN")
    dropped = sum(1 for review in reviews if _review_decision(review) == "DROP")
    return step_summary_json(
        StepSummaryItem(
            text=(
                f"Read eligibility: retained {retained} source candidates, "
                f"dropped {dropped}."
            )
        ),
        *(
            StepSummaryItem(
                text=_read_eligibility_review_text(review),
                detail=StepSummaryDetail.VERBOSE,
                is_explanation=bool(_text(review.get("retention_basis"))),
                subject=(
                    f"{_text(review.get('source_candidate_id'))} "
                    f"{_text(review.get('read_id'))}"
                ).strip(),
                disposition=_review_decision(review),
                basis=_text(review.get("retention_basis")),
            )
            for review in reviews
        ),
    )


def _read_eligibility_reviews(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    reviews: list[dict[str, Any]] = []
    assessments = _dict_or_empty(payload.get("requested_fact_assessments"))
    for assessment in assessments.values():
        if not isinstance(assessment, dict):
            continue
        fact_reviews = _dict_or_empty(assessment.get("read_candidate_reviews"))
        reviews.extend(
            {"source_candidate_id": source_candidate_id, **review}
            for source_candidate_id, review in fact_reviews.items()
            if isinstance(review, dict)
        )
    return tuple(reviews)


def _read_eligibility_review_text(review: dict[str, Any]) -> str:
    source_candidate_id = _text(review.get("source_candidate_id")) or "unknown_source"
    read_id = _text(review.get("read_id"))
    decision = _review_decision(review) or "UNKNOWN"
    rows = len(_texts(review.get("relevant_row_path_tokens")))
    fields = len(_texts(review.get("relevant_field_tokens")))
    basis = _text(review.get("retention_basis"))
    label = f"{source_candidate_id} {read_id}".strip()
    parts = [f"{label}: {decision}", f"rows={rows}", f"fields={fields}"]
    if basis:
        parts.append(basis)
    return " - ".join(parts)


def _plan_selection_step_summary(payload: dict[str, Any]) -> dict[str, object]:
    reviews = _plan_selection_reviews(payload)
    if not reviews:
        return {}
    source_ids = tuple(
        source_id for review in reviews if (source_id := _source_candidate_id(review))
    )
    return step_summary_json(
        StepSummaryItem(
            text=f"Plan selection reviewed source candidates: {', '.join(source_ids)}."
        ),
        *(
            StepSummaryItem(
                text=_plan_selection_review_text(review),
                detail=StepSummaryDetail.VERBOSE,
                is_explanation=bool(_plan_selection_basis(review)),
                subject=_source_candidate_id(review),
                disposition=_plan_selection_disposition(review),
                basis=_plan_selection_basis(review),
            )
            for review in reviews
        ),
    )


def _plan_selection_reviews(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    outcome = _dict_or_empty(payload.get("outcome"))
    reviews_by_fact = _dict_or_empty(outcome.get("reviews_by_requested_fact"))
    if reviews_by_fact:
        return tuple(
            review
            for fact_reviews in reviews_by_fact.values()
            for review in _dict_or_empty(fact_reviews).values()
            if isinstance(review, dict)
        )
    reviews: list[dict[str, Any]] = []
    for assessment in _dicts(payload.get("answer_request_assessments")):
        reviews.extend(_dicts(assessment.get("source_candidate_reviews")))
    for assessment in _dicts(payload.get("requested_fact_assessments")):
        reviews.extend(_dicts(assessment.get("source_candidate_reviews")))
    return tuple(reviews)


def _plan_selection_review_text(review: dict[str, Any]) -> str:
    source_candidate_id = _source_candidate_id(review) or "unknown_source"
    alignment = _plan_selection_disposition(review) or "UNKNOWN"
    basis = _plan_selection_basis(review)
    if basis:
        return f"{source_candidate_id}: {alignment} - {basis}"
    return f"{source_candidate_id}: {alignment}"


def _plan_selection_disposition(review: dict[str, Any]) -> str:
    return _text(
        review.get("source_alignment")
        or review.get("alignment")
        or review.get("disposition")
    )


def _plan_selection_basis(review: dict[str, Any]) -> str:
    return _text(
        review.get("basis")
        or review.get("alignment_basis")
        or review.get("source_alignment_basis")
    )


def _review_decision(review: dict[str, Any]) -> str:
    return _text(review.get("decision") or review.get("retention_decision")).upper()


def _source_candidate_id(review: dict[str, Any]) -> str:
    return _text(
        review.get("source_candidate_id")
        or review.get("candidate_source_id")
        or review.get("source_id")
    )


def _dicts(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _title_words(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())
