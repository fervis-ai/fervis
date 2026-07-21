from __future__ import annotations

from dataclasses import replace
from typing import Any

from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.operation_families.plan_selection_registry import (
    plan_selection_shape_specs_for_family,
)
from fervis.lookup.plan_selection.source_strategies import source_strategies_by_fact
from fervis.lookup.plan_selection.model import OperationEvidence
from fervis.lookup.question_contract import (
    GroupKeyDomainKind,
    GroupKeySourceKind,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactOrderingDirection,
    ResultSelectionKind,
    RequestedFactGroupKey,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
)
from fervis.lookup.read_eligibility import RetainedReadAssessment, ResolvedRetainedReadSet
from fervis.lookup.read_eligibility.candidate_identity import read_candidate_signature
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.source_binding.candidates.registry import (
    source_candidate_discovery_payload,
    source_candidate_discovery_registry,
)
from fervis.lookup.source_binding.model import SourceCandidateDiscoveryRequest

from tests.testkit.assertions import subset_mismatches
from tests.testkit.catalog import catalog_from_payload


def run_plan_selection_prompt_surface_case(payload: dict[str, Any]) -> list[str]:
    request = _request(payload["input"])
    read_eligibility = _read_eligibility(
        payload["input"].get("read_eligibility") or (),
        initial_candidate_payload=source_candidate_discovery_payload(request),
    )
    if read_eligibility is not None:
        request = replace(request, read_eligibility=read_eligibility)
    candidate_payload = source_candidate_discovery_payload(request)
    candidates_by_id = _source_candidates_by_id(candidate_payload)
    strategies = source_strategies_by_fact(
        source_candidate_discovery_registry(request),
        requested_facts=request.requested_facts,
        relation_catalog=request.relation_catalog,
        shape_specs_for_family=plan_selection_shape_specs_for_family,
    )
    strategy_alternatives_by_read: dict[str, dict[str, dict[str, Any]]] = {}
    for requested_fact_id, fact_strategies in strategies.items():
        for strategy in fact_strategies:
            for member in strategy.source_members:
                candidate = candidates_by_id.get(member.source_candidate_id)
                if candidate is None or not member.read_id:
                    continue
                strategies_by_operation = strategy_alternatives_by_read.setdefault(
                    member.read_id,
                    {},
                )
                strategies_by_operation[_operation_key(member.operation_evidence)] = {
                    "requested_fact_id": requested_fact_id,
                    "read_id": member.read_id,
                    "plan_shape": strategy.plan_shape,
                    **_evidence_summary(
                        candidate,
                        support_set_ids=member.fulfillment_support_set_ids,
                    ),
                }
    return subset_mismatches(
        actual={"strategy_alternatives_by_read": strategy_alternatives_by_read},
        expected_subset=payload["expect"]["result_contains"],
    )


def _request(payload: dict[str, Any]) -> SourceCandidateDiscoveryRequest:
    fact = _requested_fact(payload["requested_fact"])
    catalog = _catalog(payload["catalog"])
    selected_read_ids = tuple(str(item) for item in payload["selected_read_ids"])
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id=fact.id,
                query_terms=tuple(payload.get("query_terms") or ()),
                rankings=tuple(
                    CatalogSelectionRanking(read_id=read_id, score=100 - index)
                    for index, read_id in enumerate(selected_read_ids)
                ),
                selected_read_ids=selected_read_ids,
            ),
        ),
        selected_read_ids=selected_read_ids,
    )
    return SourceCandidateDiscoveryRequest(
        question=str(payload.get("question") or ""),
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        available_values=_fact_values(payload.get("available_values") or ()),
    )


def _read_eligibility(
    payload: object,
    *,
    initial_candidate_payload: dict[str, Any],
) -> ResolvedRetainedReadSet | None:
    if not isinstance(payload, list):
        return None
    candidates_by_read = _source_candidates_by_read(initial_candidate_payload)
    return ResolvedRetainedReadSet(
        retained_reads=tuple(
            RetainedReadAssessment(
                source_candidate_id=str(
                    candidates_by_read[str(item["read_id"])]["source_candidate_id"]
                ),
                source_candidate_signature=read_candidate_signature(
                    candidates_by_read[str(item["read_id"])],
                    requested_fact_id=str(item["requested_fact_id"]),
                ),
                requested_fact_id=str(item["requested_fact_id"]),
                read_id=str(item["read_id"]),
                relevant_row_path_ids=tuple(
                    str(value) for value in item.get("relevant_row_path_ids") or ()
                ),
                relevant_field_refs=tuple(
                    str(value) for value in item.get("relevant_field_refs") or ()
                ),
                retention_basis=str(
                    item.get("retention_basis") or "Selected by conformance fixture."
                ),
            )
            for item in payload
            if isinstance(item, dict)
        )
    )


def _source_candidates_by_read(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for candidate in _source_candidates_by_id(payload).values():
        read_id = str(candidate.get("read_id") or "")
        if read_id:
            output[read_id] = candidate
    return output


def _requested_fact(payload: dict[str, Any]) -> RequestedFact:
    family = RequestedFactAnswerExpressionFamily(
        str(payload["answer_expression_family"])
    )
    return RequestedFact(
        id=str(payload.get("id") or "fact_1"),
        description=str(payload["description"]),
        answer_expression=RequestedFactAnswerExpression(
            family=family,
            group_key=_group_key(payload.get("group_key")),
            selection_kind=(
                ResultSelectionKind(
                    str(payload.get("selection_kind") or "all_results")
                )
                if family
                in {
                    RequestedFactAnswerExpressionFamily.LIST_ROWS,
                    RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
                }
                else None
            ),
            ordering_basis=str(payload.get("ordering_basis") or ""),
            ordering_direction=(
                RequestedFactOrderingDirection(
                    str(payload.get("ordering_direction"))
                )
                if payload.get("ordering_direction")
                else None
            ),
            limit_input_ref=str(payload.get("limit_input_ref") or ""),
        ),
        answer_subject=RequestedFactAnswerSubject(
            subject_text=str(payload["subject_text"])
        ),
        answer_outputs=tuple(
            RequestedFactAnswerOutput(
                id=str(item["id"]),
                role=str(item["role"]),
                description=str(item.get("description") or item["id"]),
            )
            for item in payload["answer_outputs"]
        ),
    )


def _group_key(raw_value: object) -> RequestedFactGroupKey | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("group_key must be an object")
    return RequestedFactGroupKey(
        id=str(raw_value.get("id") or "group_key"),
        description=str(raw_value.get("description") or "group"),
        domain=GroupKeyDomainKind(str(raw_value.get("domain") or "")),
        source_kind=(
            GroupKeySourceKind(str(raw_value["source_kind"]))
            if raw_value.get("source_kind")
            else None
        ),
        temporal_grain=str(raw_value.get("grain") or ""),
        question_input_refs=tuple(
            str(item) for item in raw_value.get("question_input_refs") or ()
        ),
    )


def _catalog(payload: dict[str, Any]) -> RelationCatalog:
    return catalog_from_payload(payload)


def _fact_values(payload: object) -> tuple[FactValue, ...]:
    return tuple(
        FactValue.time(
            id=str(item["id"]),
            expression=str(item["expression"]),
            resolved_start=str(item["resolved_start"]),
            resolved_end=str(item["resolved_end"]),
            granularity=str(item["granularity"]),
            applies_to_requested_fact_ids=tuple(
                str(fact_id)
                for fact_id in item.get("applies_to_requested_fact_ids") or ()
            ),
        )
        for item in payload
        if isinstance(item, dict) and item.get("kind") == "time"
    )


def _source_candidates_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for context in fact_sources.get("source_contexts") or ():
            if not isinstance(context, dict):
                continue
            for candidate in context.get("source_options") or ():
                if not isinstance(candidate, dict):
                    continue
                candidate_id = str(candidate.get("source_candidate_id") or "")
                if candidate_id:
                    output[candidate_id] = candidate
    return output


def _evidence_summary(
    candidate: dict[str, Any],
    *,
    support_set_ids: tuple[str, ...],
) -> dict[str, Any]:
    selected_ids = set(support_set_ids)
    support_sets = [
        support_set
        for support_set in candidate.get("fulfillment_support_sets") or ()
        if isinstance(support_set, dict)
        and str(support_set.get("fulfillment_support_set_id") or "") in selected_ids
    ]
    return {
        "entity_evidence_by_id": _entity_evidence_by_id(
            support_sets,
        ),
        "metric_fields_by_field": _evidence_by_field(
            support_sets,
            evidence_key="metric_measure_evidence",
        ),
    }


def _entity_evidence_by_id(
    support_sets: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for support_set in support_sets:
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for item in slot.get("entity_evidence") or ():
                if not isinstance(item, dict):
                    continue
                key_id = str(item.get("key_id") or item.get("target_key_id") or "")
                if not key_id:
                    continue
                output[key_id] = {
                    "entity_kind": str(
                        item.get("entity_kind") or item.get("target_entity_kind") or ""
                    ),
                    "fields": [
                        str(component.get("field_id") or "")
                        for component in item.get("components") or ()
                        if isinstance(component, dict)
                        and str(component.get("field_id") or "")
                    ],
                    "row_path": str(item.get("row_path_id") or ""),
                }
    return output


def _operation_key(operation_evidence: tuple[OperationEvidence, ...]) -> str:
    field_ids = tuple(
        item.field_id
        for item in operation_evidence
        if item.field_id
    )
    return "__".join(field_ids)


def _evidence_by_field(
    support_sets: list[dict[str, Any]],
    *,
    evidence_key: str,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for support_set in support_sets:
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for item in slot.get(evidence_key) or ():
                if not isinstance(item, dict):
                    continue
                field_id = str(item.get("field_id") or "")
                if field_id:
                    output[field_id] = {
                        "field": field_id,
                        "row_path": str(item.get("row_path_id") or ""),
                        "type": str(item.get("type") or ""),
                    }
    return output
