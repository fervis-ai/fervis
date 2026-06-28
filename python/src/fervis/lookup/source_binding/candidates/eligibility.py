"""Fact-level eligibility for memory-backed source candidates."""

from ._shared import (
    Any,
    FactValue,
    SourceCandidateInputRequest,
    TimeValuePayload,
    ValueKind,
)
from .scope_values import (
    memory_relation_scope_value_segments,
    scope_values_include_time,
)


def _memory_candidate_with_fact_eligibility(
    candidate: dict[str, Any],
    *,
    request: SourceCandidateInputRequest,
) -> dict[str, Any] | None:
    applies_to = tuple(
        fact.id
        for fact in request.requested_facts
        if _candidate_applies_to_fact(
            candidate, requested_fact_id=fact.id, request=request
        )
    )
    if not applies_to:
        return None
    output = dict(candidate)
    output["applies_to_requested_facts"] = list(applies_to)
    return output


def _candidate_applies_to_fact(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
    request: SourceCandidateInputRequest,
) -> bool:
    active_times = _active_time_values_for_fact(
        requested_fact_id,
        request=request,
    )
    if not active_times:
        return True
    scope_segments = _candidate_scope_value_segments(
        candidate,
        memory_inputs=request.memory_inputs,
    )
    if not scope_segments:
        return False
    return all(
        any(
            scope_values_include_time(segment, value=time_value)
            for segment in scope_segments
        )
        for time_value in active_times
    )


def _active_time_values_for_fact(
    requested_fact_id: str,
    *,
    request: SourceCandidateInputRequest,
) -> tuple[FactValue, ...]:
    fact_known_input_ids = {
        known.id
        for fact in request.requested_facts
        if fact.id == requested_fact_id
        for known in fact.known_inputs
    }
    return tuple(
        value
        for value in request.available_values
        if value.kind == ValueKind.TIME
        and isinstance(value.payload, TimeValuePayload)
        and _value_applies_to_fact(
            value,
            requested_fact_id=requested_fact_id,
            fact_known_input_ids=fact_known_input_ids,
        )
    )


def _value_applies_to_fact(
    value: FactValue,
    *,
    requested_fact_id: str,
    fact_known_input_ids: set[str],
) -> bool:
    if value.applies_to_requested_fact_ids:
        return requested_fact_id in value.applies_to_requested_fact_ids
    known_input_id = _known_input_id_for_value(value)
    return bool(known_input_id and known_input_id in fact_known_input_ids)


def _candidate_scope_value_segments(
    candidate: dict[str, Any],
    *,
    memory_inputs: dict[str, Any],
) -> tuple[frozenset[str], ...]:
    if str(candidate.get("kind") or "") == "same_scope_api_read":
        return _same_scope_candidate_value_segments(candidate)
    relation_id = str(
        candidate.get("memory_relation_id") or candidate.get("source_relation_id") or ""
    )
    if not relation_id:
        return ()
    relation = _memory_relation(memory_inputs, relation_id=relation_id)
    if relation is None:
        return ()
    return memory_relation_scope_value_segments(relation)


def _same_scope_candidate_value_segments(
    candidate: dict[str, Any],
) -> tuple[frozenset[str], ...]:
    output: list[frozenset[str]] = []
    for invocation in _same_scope_invocations(candidate):
        values = frozenset(
            str(value)
            for item in invocation.get("bound_params") or ()
            if isinstance(item, dict)
            for value in (item.get("value"),)
            if value not in ("", None) and not isinstance(value, (dict, list))
        )
        if values:
            output.append(values)
    return tuple(output)


def _same_scope_invocations(candidate: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    invocations = tuple(
        item
        for item in candidate.get("source_invocations") or ()
        if isinstance(item, dict)
    )
    if invocations:
        return invocations
    if candidate.get("bound_params"):
        return ({"bound_params": candidate.get("bound_params")},)
    return ()


def _memory_relation(
    memory_inputs: dict[str, Any],
    *,
    relation_id: str,
) -> dict[str, Any] | None:
    for relation in memory_inputs.get("memoryRelations", ()) or ():
        if not isinstance(relation, dict):
            continue
        if str(relation.get("id") or "") == relation_id:
            return relation
    return None


def _known_input_id_for_value(value: FactValue) -> str:
    for proof_ref in value.proof_refs:
        text = str(proof_ref)
        if text.startswith("known_input:"):
            return text.removeprefix("known_input:")
    return ""
