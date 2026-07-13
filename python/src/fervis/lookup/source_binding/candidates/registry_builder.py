"""Parse final candidate cards into the typed runtime registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    SourceAppliedFilter,
)
from fervis.lookup.answer_program.relations import SourceKind
from fervis.lookup.source_binding.compiler_ir import DraftRelationSource
from fervis.lookup.source_binding.param_values import canonical_param_value

from .bindings import _bound_param_bindings
from .model import (
    CandidateBindingValue,
    CandidateNormalInstanceProfile,
    CandidateParameter,
    CandidateParamDecision,
    CandidatePopulationBinding,
    CandidateRowPredicate,
    SourceCandidate,
    SourceCandidateRegistry,
)
from fervis.lookup.source_binding.candidates.contracts import (
    JsonObject,
    JsonValue,
    parse_entity_target,
    parse_evidence_item,
    parse_fulfillment_support_set,
)


def parse_source_candidate_registry(payload: JsonObject) -> SourceCandidateRegistry:
    candidates_by_id = _source_candidates_from_cards(payload)
    return SourceCandidateRegistry(
        prompt_payload=payload,
        candidates_by_id=candidates_by_id,
        prompt_candidate_ids=tuple(candidates_by_id),
    )


def _source_candidates_from_cards(payload: JsonObject) -> dict[str, SourceCandidate]:
    candidates: dict[str, SourceCandidate] = {}
    for fact_sources in _objects(payload.get("requested_fact_sources")):
        requested_fact_id = _text(fact_sources.get("requested_fact_id"))
        for context in _objects(fact_sources.get("source_contexts")):
            for card in _objects(context.get("source_options")):
                candidate = _source_candidate(card, requested_fact_id=requested_fact_id)
                _add_candidate(candidates, candidate)
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for card in _objects(payload.get(key)):
            candidate = _source_candidate(card, requested_fact_id="")
            _add_candidate(candidates, candidate)
    return candidates


def _add_candidate(
    candidates: dict[str, SourceCandidate],
    candidate: SourceCandidate,
) -> None:
    existing = candidates.get(candidate.id)
    if existing is None:
        candidates[candidate.id] = candidate
        return
    candidate_contract = replace(candidate, applies_to_requested_fact_ids=())
    existing_contract = replace(existing, applies_to_requested_fact_ids=())
    if existing_contract != candidate_contract:
        raise ValueError("source candidate id identifies conflicting contracts")
    requested_fact_ids = tuple(
        dict.fromkeys(
            (
                *existing.applies_to_requested_fact_ids,
                *candidate.applies_to_requested_fact_ids,
            )
        )
    )
    candidates[candidate.id] = replace(
        existing,
        applies_to_requested_fact_ids=requested_fact_ids,
    )


def _source_candidate(card: JsonObject, *, requested_fact_id: str) -> SourceCandidate:
    kind = _text(card.get("kind"))
    candidate_id = _text(card.get("source_candidate_id"))
    source = _relation_source(card, kind=kind)
    params = tuple(_parameter(item) for item in _objects(card.get("params")))
    applied_binding_sets = _applied_param_binding_sets(card, kind=kind)
    applied_bindings = (
        applied_binding_sets[0]
        if kind == "same_scope_api_read" and applied_binding_sets
        else _bound_param_bindings(card.get("bound_params"))
    )
    evidence_items = tuple(
        parse_evidence_item(item) for item in _objects(card.get("evidence_items"))
    )
    if any(not item.evidence_id for item in evidence_items):
        raise ValueError("final source candidate evidence requires ids")
    fulfillment_support_sets = tuple(
        parse_fulfillment_support_set(item)
        for item in _objects(card.get("fulfillment_support_sets"))
    )
    population_bindings = tuple(
        _population_binding(item) for item in _objects(card.get("population_bindings"))
    )
    row_predicates = tuple(
        _row_predicate(item) for item in _objects(card.get("row_predicates"))
    )
    population_role_ids = tuple(
        role_id
        for item in _objects(card.get("population_roles"))
        for role_id in (_text(item.get("role_id")),)
        if role_id
    )
    return SourceCandidate(
        id=candidate_id,
        applies_to_requested_fact_ids=tuple(
            dict.fromkeys(
                (
                    *((requested_fact_id,) if requested_fact_id else ()),
                    *_texts(card.get("applies_to_requested_facts")),
                )
            )
        ),
        kind=kind,
        source=source,
        value_id=_text(card.get("value_id")),
        source_relation_id=_text(card.get("source_relation_id")),
        source_field_id=_text(card.get("source_field_id")),
        cardinality=_text(card.get("cardinality")),
        result_row_path_ids=_result_row_path_ids(card),
        params=params,
        applied_param_bindings=applied_bindings,
        applied_param_binding_sets=applied_binding_sets,
        applied_filters=_applied_filters(card),
        evidence_items=evidence_items,
        fulfillment_support_sets=fulfillment_support_sets,
        population_bindings=population_bindings,
        row_predicates=row_predicates,
        population_role_ids=population_role_ids,
    )


def _relation_source(card: JsonObject, *, kind: str) -> DraftRelationSource | None:
    if kind in {"new_api_read", "same_scope_api_read"}:
        return DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id=_text(card.get("read_id")),
            row_source_id=_text(card.get("row_source_id")),
        )
    if kind == "prior_answer_rows":
        return DraftRelationSource(
            kind=SourceKind.MEMORY_READ,
            memory_relation_id=_text(card.get("memory_relation_id")),
        )
    if kind in {"calendar", "generated_calendar"}:
        return DraftRelationSource(
            kind=SourceKind.GENERATED_CALENDAR,
            row_source_id=_text(card.get("row_source_id")),
            calendar_id=_text(card.get("calendar_id")),
        )
    if kind == "value":
        return None
    raise ValueError(f"unsupported source candidate kind: {kind}")


def _parameter(payload: JsonObject) -> CandidateParameter:
    population_contract = _mapping(payload.get("population_contract"))
    omission = _mapping(population_contract.get("omission_behavior"))
    entity_target_payload = _mapping(payload.get("entity_target"))
    entity_target = (
        parse_entity_target(entity_target_payload) if entity_target_payload else None
    )
    choices = tuple(
        value
        for item in _values(payload.get("choices"))
        for value in (canonical_param_value(item),)
        if value
    )
    return CandidateParameter(
        id=_text(payload.get("param_id")),
        type=_text(payload.get("type")),
        required=payload.get("required") is True,
        choices=choices,
        decision_options=tuple(
            _param_decision(item) for item in _objects(payload.get("decision_options"))
        ),
        binding_values=tuple(
            _binding_value(item) for item in _objects(payload.get("binding_values"))
        ),
        entity_target=entity_target,
        has_default="default" in payload and payload.get("default") is not None,
        default=payload.get("default"),
        finite_choice_review=bool(choices)
        and bool(population_contract)
        and (
            _text(omission.get("kind")) == "all_values"
            or payload.get("required") is True
        ),
        omission_kind=_text(omission.get("kind")),
        omission_default_value=canonical_param_value(omission.get("default_value")),
        normal_instance_profiles=tuple(
            _normal_instance_profile(item)
            for item in _objects(payload.get("normal_instance_role_profiles"))
        ),
        owned_membership_test_ids=_owned_membership_test_ids(population_contract),
    )


def _param_decision(payload: JsonObject) -> CandidateParamDecision:
    return CandidateParamDecision(
        id=_text(payload.get("param_decision_id")),
        decision=_text(payload.get("decision")),
        value=_text(payload.get("value")),
        value_component=_text(payload.get("value_component")),
    )


def _binding_value(payload: JsonObject) -> CandidateBindingValue:
    return CandidateBindingValue(
        value=_text(payload.get("value")),
        label=_text(payload.get("label")),
        source=_text(payload.get("source")),
        value_component=_text(payload.get("value_component")),
    )


def _normal_instance_profile(payload: JsonObject) -> CandidateNormalInstanceProfile:
    return CandidateNormalInstanceProfile(
        test_id=_text(payload.get("test_id")),
        excluded_role_ids=tuple(
            role_id
            for item in _objects(payload.get("excluded_state_roles"))
            for role_id in (_text(item.get("role")),)
            if role_id
        ),
    )


def _population_binding(payload: JsonObject) -> CandidatePopulationBinding:
    basis = _mapping(payload.get("basis"))
    return CandidatePopulationBinding(
        id=_text(payload.get("population_binding_id")),
        kind=_text(payload.get("kind")),
        memory_relation_id=_text(basis.get("memory_relation_id")),
        proof_refs=_texts(basis.get("proof_refs")),
    )


def _row_predicate(payload: JsonObject) -> CandidateRowPredicate:
    return CandidateRowPredicate(
        id=_text(payload.get("predicate_id")),
        field_id=_text(payload.get("field_id")),
        field_type=_text(payload.get("type")),
        operator=_text(payload.get("operator")) or "in",
        allowed_values=_texts(payload.get("allowed_values")),
        owned_membership_test_ids=_owned_membership_test_ids(payload),
    )


def _owned_membership_test_ids(payload: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return _texts(payload.get("owned_membership_test_ids"))


def _applied_param_binding_sets(
    card: JsonObject,
    *,
    kind: str,
) -> tuple[tuple[DraftEndpointParamBinding, ...], ...]:
    if kind != "same_scope_api_read":
        return ()
    invocations = _objects(card.get("source_invocations"))
    if invocations:
        return tuple(
            _bound_param_bindings(invocation.get("bound_params"))
            for invocation in invocations
        )
    bindings = _bound_param_bindings(card.get("bound_params"))
    return (bindings,) if bindings else ()


def _applied_filters(card: JsonObject) -> tuple[SourceAppliedFilter, ...]:
    return tuple(
        SourceAppliedFilter(
            known_input_id=_text(item.get("known_input_id")),
            predicate_field_ids=_texts(item.get("field_ids")),
            value_id=_text(item.get("value_id")),
            value_kind=_text(item.get("kind")),
            display_value=_text(item.get("display_value")),
            matched_field_ref=_text(item.get("matched_field_ref")),
            matched_field_path=_text(item.get("matched_field_path")),
            resolved_start=_text(item.get("resolved_start")),
            resolved_end=_text(item.get("resolved_end")),
            literal_type=_text(item.get("literal_type")),
            operator=_text(item.get("operator")) or "equals",
        )
        for item in _objects(card.get("applied_filters"))
    )


def _result_row_path_ids(card: JsonObject) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            row_path_id
            for grain in _objects(card.get("result_grains"))
            for row_path_id in (_text(grain.get("row_path_id")),)
            if row_path_id
        )
    )


def _mapping(value: JsonValue | None) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _objects(value: JsonValue | None) -> tuple[JsonObject, ...]:
    if isinstance(value, dict):
        return (value,)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _values(value: JsonValue | None) -> tuple[JsonValue, ...]:
    return tuple(value) if isinstance(value, list) else ()


def _texts(value: JsonValue | None) -> tuple[str, ...]:
    return tuple(item for item in _values(value) if isinstance(item, str) and item)


def _text(value: JsonValue | None) -> str:
    return value.strip() if isinstance(value, str) else ""
