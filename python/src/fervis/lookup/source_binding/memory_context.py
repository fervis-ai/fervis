"""Model-facing memory context for source binding."""

from __future__ import annotations

from typing import Any


def source_binding_memory_context_payload(
    memory_inputs: dict[str, Any],
    *,
    active_memory_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return compact memory context for model-facing source binding prompts."""

    active_ids = {str(item) for item in active_memory_ids if str(item)}
    output: dict[str, Any] = {}
    relations = [
        _compact_relation(relation)
        for relation in memory_inputs.get("memoryRelations") or ()
        if isinstance(relation, dict)
        and _include_relation(relation, active_ids=active_ids)
    ]
    if relations:
        output["memoryRelations"] = relations
    values = [
        _compact_value(value)
        for value in memory_inputs.get("memoryValues") or ()
        if isinstance(value, dict) and _include_value(value, active_ids=active_ids)
    ]
    if values:
        output["memoryValues"] = values
    outcomes = [
        _compact_outcome(outcome)
        for outcome in memory_inputs.get("memoryOutcomes") or ()
        if isinstance(outcome, dict)
        and _include_outcome(outcome, active_ids=active_ids)
    ]
    if outcomes:
        output["memoryOutcomes"] = outcomes
    return output


def _include_relation(relation: dict[str, Any], *, active_ids: set[str]) -> bool:
    relation_id = str(relation.get("id") or "")
    return not active_ids or relation_id in active_ids


def _include_value(value: dict[str, Any], *, active_ids: set[str]) -> bool:
    value_id = str(value.get("id") or "")
    if active_ids:
        return value_id in active_ids
    return not _has_source_link(value)


def _include_outcome(outcome: dict[str, Any], *, active_ids: set[str]) -> bool:
    outcome_id = str(outcome.get("id") or "")
    return not active_ids or outcome_id in active_ids


def _has_source_link(value: dict[str, Any]) -> bool:
    return bool(value.get("sourceRelationId") or value.get("sourceRowId"))


def _compact_relation(relation: dict[str, Any]) -> dict[str, Any]:
    output = _copy_non_empty(
        relation,
        keys=("id", "source", "grainKeys", "rowCount"),
    )
    fields = [
        _compact_field(field)
        for field in relation.get("fields") or ()
        if isinstance(field, dict)
    ]
    if fields:
        output["fields"] = fields
    completeness = relation.get("completeness")
    if isinstance(completeness, dict):
        compact_completeness = _copy_non_empty(
            completeness,
            keys=("status", "setKind", "pagination", "scopeFingerprint", "rowCount"),
        )
        if compact_completeness:
            output["completeness"] = compact_completeness
    return output


def _compact_field(field: dict[str, Any]) -> dict[str, Any]:
    return _copy_non_empty(
        field,
        keys=("id", "type", "grain", "sourceField", "prior_answer_output_ids"),
    )


def _compact_value(value: dict[str, Any]) -> dict[str, Any]:
    return _copy_non_empty(
        value,
        keys=(
            "id",
            "type",
            "value",
            "sourceRelationId",
            "sourceRowId",
            "sourceFieldId",
            "priorAnswerOutputIds",
        ),
    )


def _compact_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    return _copy_non_empty(
        outcome,
        keys=(
            "id",
            "terminal",
            "clarificationQuestions",
            "scope",
            "sourceQuestion",
            "sourceAnswer",
        ),
    )


def _copy_non_empty(
    payload: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in keys
        if key in payload and payload[key] not in (None, "", [], ())
    }
