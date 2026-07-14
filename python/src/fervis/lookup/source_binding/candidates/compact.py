"""Compact source-binding candidate prompt projection."""

from fervis.lookup.turn_prompts.projections import ApiReadResponseShapeProjector
from fervis.lookup.question_contract import (
    RequestedFact,
    RequestedFactAnswerExpressionFamily,
)

from ._shared import Any, RelationCatalog
from .candidate_tree import CandidateTreeContext, map_source_candidate_tree
from .row_predicates import candidate_row_path_ids


_FULFILLMENT_EVIDENCE_KEYS = (
    "metric_measure_evidence",
    "value_evidence",
    "row_count_basis_evidence",
    "entity_evidence",
)


def _compact_prompt_payload(
    payload: dict[str, Any],
    *,
    relation_catalog: RelationCatalog,
    requested_facts: tuple[RequestedFact, ...] = (),
) -> dict[str, Any]:
    return map_source_candidate_tree(
        payload,
        lambda candidate, context: _compact_candidate_for_tree(
            candidate,
            context=context,
            relation_catalog=relation_catalog,
            requested_facts=requested_facts,
        ),
    )


def _compact_candidate_for_tree(
    candidate: dict[str, Any],
    *,
    context: CandidateTreeContext,
    relation_catalog: RelationCatalog,
    requested_facts: tuple[RequestedFact, ...],
) -> dict[str, Any] | None:
    if context.top_level_key and candidate.get("model_visible") is False:
        return None
    return _compact_source_candidate(
        candidate,
        relation_catalog=relation_catalog,
        requested_fact=_requested_fact_for_context(
            context,
            requested_facts=requested_facts,
        ),
    )


def _compact_source_candidate(
    candidate: dict[str, Any],
    *,
    relation_catalog: RelationCatalog,
    requested_fact: RequestedFact | None = None,
) -> dict[str, Any]:
    if candidate.get("kind") == "same_scope_api_read":
        return _compact_api_read_candidate(
            candidate,
            relation_catalog=relation_catalog,
            requested_fact=requested_fact,
        )
    if candidate.get("kind") == "new_api_read":
        return _compact_api_read_candidate(
            candidate,
            relation_catalog=relation_catalog,
            requested_fact=requested_fact,
        )
    keys = (
        "source_candidate_id",
        "kind",
        "read_id",
        "row_source_id",
        "row_path_id",
        "memory_relation_id",
        "source_relation_id",
        "source_field_id",
        "calendar_id",
        "value_id",
        "cardinality",
        "description",
        "meaning",
        "use_when",
        "applied_filters",
        "bound_params",
        "source_invocations",
        "population_bindings",
        "params",
        "evidence_items",
        "applies_to_requested_facts",
        "scope_coverage",
        "result_grains",
    )
    output = {
        key: candidate[key]
        for key in keys
        if key in candidate and candidate[key] not in (None, "", [], ())
    }
    fulfillment_support_sets = _visible_fulfillment_support_sets(
        candidate,
        requested_fact=requested_fact,
    )
    if fulfillment_support_sets:
        output["fulfillment_support_sets"] = fulfillment_support_sets
    return output


def _compact_api_read_candidate(
    candidate: dict[str, Any],
    *,
    relation_catalog: RelationCatalog,
    requested_fact: RequestedFact | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        key: candidate[key]
        for key in (
            "source_candidate_id",
            "kind",
            "read_id",
            "row_source_id",
            "memory_relation_id",
        )
        if key in candidate and candidate[key] not in (None, "", [], ())
    }
    for key in ("description", "resource_names"):
        if candidate.get(key) not in (None, "", [], ()):
            output[key] = candidate[key]
    read_shape = ApiReadResponseShapeProjector(
        relation_catalog.read(str(candidate["read_id"]))
    )
    input_params = read_shape.input_params()
    if input_params:
        output["input_params"] = input_params
    output["response_rows"] = read_shape.response_rows(
        row_path_ids=candidate_row_path_ids(candidate),
    )
    if candidate.get("row_predicates"):
        output["row_predicates"] = candidate["row_predicates"]
    _copy_source_binding_controls(output, candidate, requested_fact=requested_fact)
    invocation_count = _same_scope_invocation_count(candidate)
    if invocation_count:
        output["prior_scope_invocation_count"] = invocation_count
    return output


def _copy_source_binding_controls(
    output: dict[str, Any],
    candidate: dict[str, Any],
    *,
    requested_fact: RequestedFact | None,
) -> None:
    keys = (
        "applied_filters",
        "bound_params",
        "source_invocations",
        "population_bindings",
        "params",
    )
    for key in keys:
        if candidate.get(key) not in (None, "", [], ()):
            output[key] = candidate[key]
    fulfillment_support_sets = _visible_fulfillment_support_sets(
        candidate,
        requested_fact=requested_fact,
    )
    if fulfillment_support_sets:
        output["fulfillment_choices"] = fulfillment_support_sets
    population_roles = [
        item
        for item in candidate.get("population_roles") or ()
        if isinstance(item, dict)
    ]
    if population_roles:
        output["population_roles"] = [dict(item) for item in population_roles]


def _visible_fulfillment_support_sets(
    candidate: dict[str, Any],
    *,
    requested_fact: RequestedFact | None,
) -> list[dict[str, Any]]:
    support_sets = _visible_support_sets(
        candidate,
        requested_fact=requested_fact,
    )
    if not support_sets:
        return []
    visible_slots_by_id = {
        str(slot.get("fulfillment_slot_id") or ""): slot
        for slot in _visible_fulfillment_slots(candidate, requested_fact=requested_fact)
        if str(slot.get("fulfillment_slot_id") or "")
    }
    output: list[dict[str, Any]] = []
    for support_set in support_sets:
        visible_slots = [
            visible_slots_by_id[slot_id]
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for slot_id in (str(slot.get("fulfillment_slot_id") or ""),)
            if slot_id in visible_slots_by_id
        ]
        if not visible_slots:
            continue
        output.append(
            {
                "fulfillment_choice_id": _visible_fulfillment_choice_id(
                    support_set,
                    visible_slots=tuple(visible_slots),
                ),
                "answer_output_id": str(support_set.get("answer_output_id") or ""),
                "fulfillment_slots": visible_slots,
            }
        )
    return output


def _visible_support_sets(
    candidate: dict[str, Any],
    *,
    requested_fact: RequestedFact | None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for item in candidate.get("fulfillment_support_sets") or ()
        if isinstance(item, dict)
    )


def _visible_fulfillment_choice_id(
    support_set: dict[str, Any],
    *,
    visible_slots: tuple[dict[str, Any], ...],
) -> str:
    visible_evidence_ids = _visible_fulfillment_evidence_ids(visible_slots)
    if len(visible_evidence_ids) == 1:
        return visible_evidence_ids[0]
    return str(support_set.get("fulfillment_support_set_id") or "")


def _visible_fulfillment_evidence_ids(
    visible_slots: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            evidence_id
            for slot in visible_slots
            for key in _FULFILLMENT_EVIDENCE_KEYS
            for item in slot.get(key) or ()
            if isinstance(item, dict)
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id
        )
    )


def _visible_fulfillment_slots(
    candidate: dict[str, Any],
    *,
    requested_fact: RequestedFact | None,
) -> list[dict[str, Any]]:
    slots = [
        item
        for item in candidate.get("fulfillment_slots") or ()
        if isinstance(item, dict)
    ]
    if not slots:
        slots = [
            slot
            for support_set in candidate.get("fulfillment_support_sets") or ()
            if isinstance(support_set, dict)
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
        ]
    if not slots:
        return []
    visible_evidence_ids = _visible_result_evidence_ids(candidate)
    output: list[dict[str, Any]] = []
    for slot in slots:
        slot_output = {
            key: slot[key]
            for key in (
                "fulfillment_slot_id",
                "answer_output_id",
                "compatibility_basis",
            )
            if key in slot and slot[key] not in (None, "", [], ())
        }
        answer_output_id = str(slot.get("answer_output_id") or "")
        for key in _visible_fulfillment_role_keys(
            requested_fact,
            answer_output_id=answer_output_id,
        ):
            visible_role_evidence = [
                item
                for item in slot.get(key) or ()
                if isinstance(item, dict)
                and (
                    not visible_evidence_ids
                    or str(item.get("evidence_id") or "") in visible_evidence_ids
                    or (
                        key == "row_count_basis_evidence"
                        and str(item.get("type") or "") == "row_population"
                    )
                )
            ]
            if visible_role_evidence:
                slot_output[key] = visible_role_evidence
            else:
                slot_output.pop(key, None)
        if not any(slot_output.get(key) for key in _FULFILLMENT_EVIDENCE_KEYS):
            continue
        output.append(slot_output)
    return output


def _visible_fulfillment_role_keys(
    requested_fact: RequestedFact | None,
    *,
    answer_output_id: str,
) -> tuple[str, ...]:
    family = (
        requested_fact.answer_expression.family
        if requested_fact is not None and requested_fact.answer_expression is not None
        else None
    )
    answer_output_role = _answer_output_role(
        requested_fact,
        answer_output_id=answer_output_id,
    )
    if (
        family == RequestedFactAnswerExpressionFamily.RANKED_SELECTION
        and answer_output_role != "MEASURED_VALUE"
    ):
        return ("entity_evidence", "value_evidence")
    return (
        "metric_measure_evidence",
        "value_evidence",
        "row_count_basis_evidence",
        "entity_evidence",
    )


def _answer_output_role(
    requested_fact: RequestedFact | None,
    *,
    answer_output_id: str,
) -> str:
    if requested_fact is None:
        return ""
    answer_output = next(
        (
            output
            for output in requested_fact.support_answer_outputs
            if output.id == answer_output_id
        ),
        None,
    )
    return answer_output.role if answer_output is not None else ""


def _requested_fact_for_context(
    context: CandidateTreeContext,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> RequestedFact | None:
    if not context.requested_fact_id:
        return None
    return next(
        (fact for fact in requested_facts if fact.id == context.requested_fact_id),
        None,
    )


def _visible_result_evidence_ids(candidate: dict[str, Any]) -> set[str]:
    result_grains = tuple(
        item for item in candidate.get("result_grains") or () if isinstance(item, dict)
    )
    return {
        str(evidence.get("evidence_id") or "")
        for grain in result_grains
        for evidence in grain.get("evidence_items") or ()
        if isinstance(evidence, dict) and evidence.get("evidence_id")
    }


def _same_scope_invocation_count(candidate: dict[str, Any]) -> int:
    invocations = candidate.get("source_invocations")
    if isinstance(invocations, list) and invocations:
        return len(invocations)
    return 1 if candidate.get("bound_params") else 0
