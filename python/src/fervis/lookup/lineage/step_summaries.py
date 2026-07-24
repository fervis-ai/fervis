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
from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_items,
    lineage_explanation_paths_from_payload,
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


def _summary_source(
    *,
    purpose: str,
    parsed: dict[str, Any],
    submitted: dict[str, Any],
) -> dict[str, Any]:
    return parsed or submitted


def _turn_summary(*, purpose: str, source: dict[str, Any]) -> dict[str, object]:
    if purpose == ModelTurnPurpose.READ_ELIGIBILITY:
        return _read_eligibility_step_summary(source)
    if purpose == ModelTurnPurpose.PLAN_SELECTION:
        return _plan_selection_step_summary(source)
    if purpose == ModelTurnPurpose.SOURCE_BINDING:
        return _source_binding_step_summary(source)
    return {}


def _source_binding_step_summary(payload: dict[str, Any]) -> dict[str, object]:
    return step_summary_json(
        *(
            StepSummaryItem(
                text=basis,
                is_explanation=True,
                basis=basis,
                path=path,
            )
            for path, basis in _authored_bases(payload)
        )
    )


def _authored_bases(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> tuple[tuple[tuple[str, ...], str], ...]:
    if isinstance(value, dict):
        return tuple(
            item
            for key, child in value.items()
            for item in (
                (((*path, str(key)), child),)
                if key in {"mapping_basis", "role_match_basis"}
                and isinstance(child, str)
                and child.strip()
                else _authored_bases(child, path=(*path, str(key)))
            )
        )
    if isinstance(value, list):
        return tuple(
            item
            for index, child in enumerate(value)
            for item in _authored_bases(child, path=(*path, str(index)))
        )
    return ()


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
    if purpose == ModelTurnPurpose.READ_ELIGIBILITY:
        return _read_eligibility_semantic_items(source)
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
    outcome = _dict_or_empty(payload.get("outcome"))
    outcome_kind = _text(outcome.get("kind"))
    for index, answer_request in enumerate(_dicts(outcome.get("answer_requests")), 1):
        description = (
            _text(answer_request.get("return_request_basis"))
            if outcome_kind == "question_meaning"
            else _text(answer_request.get("description"))
        )
        requested_fact_id = _text(answer_request.get("requested_fact_ref"))
        if not description:
            description = _text(
                _dict_or_empty(answer_request.get("origin")).get("meaning")
            )
        if description:
            items.append(
                StepSemanticItem(
                    kind="requested_fact",
                    payload={
                        "requested_fact_id": requested_fact_id or f"fact_{index}",
                        "description": description,
                    },
                )
            )
    for supplied in _dicts(outcome.get("supplied_values")):
        raw_value = _dict_or_empty(supplied.get("value"))
        text = _operand_text(raw_value.get("operands"))
        origin = _dict_or_empty(raw_value.get("origin"))
        denotation = _dict_or_empty(supplied.get("denotation"))
        input_id = _text(origin.get("resolved_input_ref")) or text
        if not input_id or not text:
            continue
        denotation_kind = _text(denotation.get("kind"))
        input_kind = {
            "identity_reference": "IDENTITY_REFERENCE",
            "scalar": "NON_IDENTITY_SCALAR",
        }.get(denotation_kind, denotation_kind.upper())
        items.append(
            StepSemanticItem(
                kind="known_input",
                payload={
                    "input_id": input_id,
                    "text": text,
                    "kind": input_kind,
                    "role": "",
                    "description": _text(supplied.get("meaning")),
                    "resolved_value_text": text,
                },
            )
        )
    return tuple(items)


def _query_enrichment_semantic_items(
    payload: dict[str, Any],
) -> tuple[StepSemanticItem, ...]:
    return tuple(
        StepSemanticItem(
            kind="resource_recall",
            payload={
                "input_use_ref": _text(item.get("input_use_ref")),
                "resource_name": term,
            },
        )
        for item in _dicts(payload.get("input_resource_search_terms"))
        for term in _texts(item.get("catalog_search_terms"))
        if _text(item.get("input_use_ref")) and term
    )


def _grounding_semantic_items(payload: dict[str, Any]) -> tuple[StepSemanticItem, ...]:
    reviews = _dict_or_empty(payload.get("reference_reviews"))
    items: list[StepSemanticItem] = []
    for task_ref, raw_review in reviews.items():
        review = _dict_or_empty(raw_review)
        for raw_type_review in _dict_or_empty(
            review.get("resource_type_reviews")
        ).values():
            type_review = _dict_or_empty(raw_type_review)
            for route_ref, raw_route_review in _dict_or_empty(
                type_review.get("route_reviews")
            ).items():
                route_review = _dict_or_empty(raw_route_review)
                resolution = _dict_or_empty(route_review.get("resolution"))
                if _text(resolution.get("decision")) != "CAN_RESOLVE_LOOKUP_TEXT":
                    continue
                items.append(
                    StepSemanticItem(
                        kind="resolver_candidate",
                        payload={
                            "input_id": str(task_ref),
                            "resolver_read_id": str(route_ref),
                            "resolver_label": _title_words(str(route_ref)),
                            "basis": _text(route_review.get("assessment_basis")),
                        },
                    )
                )
    for task_ref, raw_resolution in _dict_or_empty(
        payload.get("time_resolutions")
    ).items():
        date_intent = _dict_or_empty(
            _dict_or_empty(raw_resolution).get("date_intent")
        )
        expression = _text(date_intent.get("expression"))
        intent = _dict_or_empty(date_intent.get("intent"))
        if expression:
            interpreted = _interpreted_input_item(
                input_id=str(task_ref),
                input_text=expression,
                kind="time",
                value=expression,
                label=expression,
                detail=_text(intent.get("unit") or intent.get("time_shape")),
            )
            if interpreted is not None:
                items.append(interpreted)
    return tuple(items)


def _read_eligibility_semantic_items(
    payload: dict[str, Any],
) -> tuple[StepSemanticItem, ...]:
    return tuple(
        StepSemanticItem(
            kind="identity_selection",
            payload={
                "input_id": str(input_ref),
                "canonical_option_id": _text(outcome.get("canonical_option_id")),
                "resolver_route_id": _text(outcome.get("resolver_route_id")),
                "basis": _identity_selection_basis(outcome),
                "outcome": _text(outcome.get("outcome")),
            },
        )
        for input_ref, raw_outcome in _dict_or_empty(
            payload.get("identity_outcomes")
        ).items()
        if (outcome := _dict_or_empty(raw_outcome))
        if (
            _text(outcome.get("canonical_option_id"))
            or _text(outcome.get("resolver_route_id"))
            or _text(outcome.get("outcome"))
        )
    )


def _identity_selection_basis(outcome: dict[str, Any]) -> str:
    return " ".join(
        dict.fromkeys(
            value
            for value in (
                _text(outcome.get("canonical_option_basis")),
                _text(outcome.get("resolver_route_basis")),
            )
            if value
        )
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
                is_explanation=bool(_text(review.get("assessment_basis"))),
                subject=_text(review.get("candidate_ref")),
                disposition=_review_decision(review),
                basis=_text(review.get("assessment_basis")),
            )
            for review in reviews
        ),
    )


def _read_eligibility_reviews(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "requested_fact_id": requested_fact_id,
            "candidate_ref": candidate_ref,
            "assessment_basis": _text(assessment.get("assessment_basis")),
            "decision": _text(assessment.get("decision")).upper(),
            "relevant_field_refs": _texts(assessment.get("relevant_field_refs")),
        }
        for requested_fact_id, raw_assessments in _dict_or_empty(
            payload.get("read_assessments_by_requested_fact")
        ).items()
        for candidate_ref, raw_assessment in _dict_or_empty(raw_assessments).items()
        if (assessment := _dict_or_empty(raw_assessment))
    )


def _read_eligibility_review_text(review: dict[str, Any]) -> str:
    source_candidate_id = _text(review.get("candidate_ref")) or "unknown_source"
    decision = _review_decision(review) or "UNKNOWN"
    fields = len(_texts(review.get("relevant_field_refs")))
    basis = _text(review.get("assessment_basis"))
    parts = [
        f"{source_candidate_id}: {decision}",
        f"fields={fields}",
    ]
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
            text=f"Plan selection assessed sources: {', '.join(source_ids)}."
        ),
        *(
            StepSummaryItem(
                text=_plan_selection_review_text(review),
                detail=StepSummaryDetail.VERBOSE,
                is_explanation=bool(_plan_selection_basis(review)),
                subject=_source_candidate_id(review),
                disposition=_text(review.get("alignment")),
                basis=_plan_selection_basis(review),
            )
            for review in reviews
        ),
    )


def _plan_selection_reviews(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "source_candidate_id": source_ref,
            "requested_fact_id": requested_fact_id,
            "basis": _text(assessment.get("basis")),
            "alignment": _text(assessment.get("alignment")),
        }
        for requested_fact_id, raw_assessments in _dict_or_empty(
            payload.get("source_assessments_by_requested_fact")
        ).items()
        for source_ref, raw_assessment in _dict_or_empty(raw_assessments).items()
        if (assessment := _dict_or_empty(raw_assessment))
    )


def _plan_selection_review_text(review: dict[str, Any]) -> str:
    source_candidate_id = _source_candidate_id(review) or "unknown_source"
    basis = _plan_selection_basis(review)
    alignment = _text(review.get("alignment")) or "UNKNOWN"
    if basis:
        return f"{source_candidate_id}: {alignment} - {basis}"
    return f"{source_candidate_id}: {alignment}"


def _plan_selection_basis(review: dict[str, Any]) -> str:
    return _text(
        review.get("basis")
        or review.get("alignment_basis")
        or review.get("source_alignment_basis")
    )


def _review_decision(review: dict[str, Any]) -> str:
    return _text(review.get("decision")).upper()


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
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _operand_text(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(_text(item) for item in value if _text(item))
    return _text(value)


def _title_words(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())
