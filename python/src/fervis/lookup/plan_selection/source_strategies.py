"""Backend-built plan selection source strategies."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from itertools import product
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.turn_prompts.projections import ApiReadResponseShapeProjector
from fervis.lookup.question_contract import (
    RequestedFactAnswerExpressionFamily,
    RequestedFact,
)
from fervis.lookup.plan_selection.family_specs import PlanSelectionShapeSpec
from fervis.lookup.plan_selection.model import (
    OperationEvidence,
    SourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.plan_selection.support_options import (
    PlanSelectionSupportOption,
    plan_selection_fulfillment_support_sets,
    plan_selection_support_options,
)
from fervis.lookup.source_binding.candidates.contracts import (
    EvidenceItem,
    FulfillmentSupportSet,
    evidence_field_ids,
    evidence_row_path_id,
)
from fervis.lookup.source_binding.candidates.model import (
    SourceCandidate,
    SourceCandidateRegistry,
)


def source_strategies_by_fact(
    source_catalog: SourceCandidateRegistry,
    *,
    requested_facts: tuple[RequestedFact, ...],
    relation_catalog: RelationCatalog,
    shape_specs_for_family: Callable[
        [RequestedFactAnswerExpressionFamily],
        tuple[PlanSelectionShapeSpec, ...],
    ],
) -> dict[str, tuple[SourceStrategy, ...]]:
    output: dict[str, tuple[SourceStrategy, ...]] = {}
    for fact in requested_facts:
        output[fact.id] = _source_strategies_for_fact(
            fact=fact,
            candidates=source_catalog.candidates_for(fact.id),
            relation_catalog=relation_catalog,
            shape_specs_for_family=shape_specs_for_family,
        )
    return output


def source_strategy_payload(
    source_strategy: SourceStrategy,
) -> dict[str, Any]:
    return {
        "source_strategy_id": source_strategy.source_strategy_id,
        "plan_shape": source_strategy.plan_shape,
        "required_answer_output_ids": list(source_strategy.required_answer_output_ids),
        "source_members": [
            _source_member_payload(member) for member in source_strategy.source_members
        ],
    }


def source_alignment_candidates_by_fact(
    source_strategies_by_fact: dict[str, tuple[SourceStrategy, ...]],
) -> dict[str, tuple[SourceStrategyMember, ...]]:
    output: dict[str, tuple[SourceStrategyMember, ...]] = {}
    for requested_fact_id, source_strategies in source_strategies_by_fact.items():
        members: dict[str, SourceStrategyMember] = {}
        for source_strategy in source_strategies:
            for member in source_strategy.source_members:
                members.setdefault(member.source_candidate_id, member)
        output[requested_fact_id] = tuple(members.values())
    return output


def source_alignment_candidate_payload(member: SourceStrategyMember) -> dict[str, Any]:
    output: dict[str, Any] = {"source_candidate_id": member.source_candidate_id}
    for key, value in (
        ("kind", member.kind),
        ("read_id", member.read_id),
        ("value_id", member.value_id),
        ("memory_relation_id", member.memory_relation_id),
        ("source_relation_id", member.source_relation_id),
        ("calendar_id", member.calendar_id),
    ):
        if value:
            output[key] = value
    if member.source_interface:
        input_params = member.source_interface.get("input_params")
        if input_params:
            output["input_params"] = input_params
        response_rows = member.source_interface.get("response_rows")
        if response_rows:
            output["response_rows"] = response_rows
    if member.applied_filters:
        output["applied_filters"] = [
            applied_filter.to_payload() for applied_filter in member.applied_filters
        ]
    return output


def source_candidate_ids_by_requested_fact_id(
    candidates_by_fact: dict[str, tuple[SourceStrategyMember, ...]],
) -> dict[str, tuple[str, ...]]:
    return {
        requested_fact_id: tuple(
            candidate.source_candidate_id for candidate in source_candidates
        )
        for requested_fact_id, source_candidates in candidates_by_fact.items()
    }


def _source_member_payload(member: SourceStrategyMember) -> dict[str, Any]:
    output: dict[str, Any] = {"source_candidate_id": member.source_candidate_id}
    if member.requirement_ids:
        output["requirement_ids"] = list(member.requirement_ids)
    for key, value in (
        ("kind", member.kind),
        ("read_id", member.read_id),
        ("value_id", member.value_id),
        ("memory_relation_id", member.memory_relation_id),
        ("source_relation_id", member.source_relation_id),
        ("calendar_id", member.calendar_id),
    ):
        if value:
            output[key] = value
    if member.field_ids:
        output["field_ids"] = list(member.field_ids)
    if member.operation_evidence:
        output["operation_evidence"] = [
            _operation_evidence_payload(item) for item in member.operation_evidence
        ]
    if member.source_interface:
        input_params = member.source_interface.get("input_params")
        if input_params:
            output["input_params"] = input_params
        response_rows = member.source_interface.get("response_rows")
        if response_rows:
            output["response_rows"] = response_rows
    return output


def _source_interface(
    candidate: SourceCandidate,
    support_sets: tuple[FulfillmentSupportSet, ...],
    *,
    relation_catalog: RelationCatalog,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    answer_output_ids = _support_set_answer_output_ids(support_sets)
    if answer_output_ids:
        summary["answer_output_ids"] = list(answer_output_ids)
    if candidate.kind in {"new_api_read", "same_scope_api_read"}:
        read_shape = ApiReadResponseShapeProjector(
            relation_catalog.read(candidate.read_id)
        )
        input_params = read_shape.input_params()
        if input_params:
            summary["input_params"] = input_params
        summary["response_rows"] = read_shape.response_rows(
            row_path_ids=candidate.result_row_path_ids,
        )
    return summary


def _support_set_answer_output_ids(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            answer_output_id
            for support_set in support_sets
            for answer_output_id in (support_set.answer_output_id,)
            if answer_output_id
        )
    )


def _source_strategies_for_fact(
    *,
    fact: RequestedFact,
    candidates: tuple[SourceCandidate, ...],
    relation_catalog: RelationCatalog,
    shape_specs_for_family: Callable[
        [RequestedFactAnswerExpressionFamily],
        tuple[PlanSelectionShapeSpec, ...],
    ],
) -> tuple[SourceStrategy, ...]:
    output: list[SourceStrategy] = []
    seen: set[tuple[Any, ...]] = set()
    answer_output_ids = tuple(output.id for output in fact.support_answer_outputs)
    for shape_spec in _plan_shape_specs_for_fact(
        fact,
        shape_specs_for_family=shape_specs_for_family,
    ):
        plan_shape = shape_spec.plan_shape
        member_options = tuple(
            _member_requirement_options(
                candidates,
                requirement_id=requirement_id,
                shape_spec=shape_spec,
                required_answer_output_ids=answer_output_ids,
            )
            for requirement_id in shape_spec.member_requirements
        )
        if any(not options for options in member_options):
            continue
        for combo in product(*member_options):
            if not shape_spec.supports_member_combo(
                source_candidate_ids=tuple(
                    candidate.id
                    for _, candidate, _ in combo
                )
            ):
                continue
            members = _source_strategy_members(
                combo,
                relation_catalog=relation_catalog,
            )
            if not members:
                continue
            if not _members_cover_required_answer_outputs(
                members,
                required_answer_output_ids=answer_output_ids,
                shape_spec=shape_spec,
            ):
                continue
            key = (
                plan_shape,
                tuple(
                    (
                        member.source_candidate_id,
                        member.requirement_ids,
                        member.fulfillment_support_set_ids,
                    )
                    for member in members
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(
                SourceStrategy(
                    source_strategy_id=(
                        f"source_strategy.{fact.id}.{plan_shape}."
                        f"{_source_strategy_member_id_suffix(members)}"
                    ),
                    plan_shape=plan_shape,
                    required_answer_output_ids=answer_output_ids,
                    source_members=members,
                )
            )
    return tuple(output)


def _plan_shape_specs_for_fact(
    fact: RequestedFact,
    *,
    shape_specs_for_family: Callable[
        [RequestedFactAnswerExpressionFamily],
        tuple[PlanSelectionShapeSpec, ...],
    ],
) -> tuple[PlanSelectionShapeSpec, ...]:
    if fact.answer_expression is None:
        return ()
    return shape_specs_for_family(fact.answer_expression.family)


def _member_requirement_options(
    candidates: tuple[SourceCandidate, ...],
    *,
    requirement_id: str,
    shape_spec: PlanSelectionShapeSpec,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[
    tuple[str, SourceCandidate, tuple[FulfillmentSupportSet, ...]], ...
]:
    output: list[
        tuple[str, SourceCandidate, tuple[FulfillmentSupportSet, ...]]
    ] = []
    for candidate in candidates:
        support_set_groups = _selected_support_set_groups_for_requirement(
            candidate,
            requirement_id=requirement_id,
            shape_spec=shape_spec,
            required_answer_output_ids=required_answer_output_ids,
        )
        output.extend(
            (requirement_id, candidate, support_sets)
            for support_sets in support_set_groups
        )
    return tuple(output)


def _selected_support_set_groups_for_requirement(
    candidate: SourceCandidate,
    *,
    requirement_id: str,
    shape_spec: PlanSelectionShapeSpec,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[tuple[FulfillmentSupportSet, ...], ...]:
    if shape_spec.support_set_grouper is not None:
        support_sets = _answer_output_fulfillment_support_sets(
            candidate,
            required_answer_output_ids=required_answer_output_ids,
        )
        return shape_spec.support_set_groups_for_requirement(
            support_sets,
            requirement_id=requirement_id,
            required_answer_output_ids=required_answer_output_ids,
            source_candidate_id=candidate.id,
        )
    support_options = plan_selection_support_options(candidate)
    validation_roles = shape_spec.validation_roles_for_requirement(requirement_id)
    requirement_has_validation_roles = validation_roles is not None
    validation_role_is_supported = True
    if validation_roles is not None:
        validation_role_is_supported = any(
            option.support_roles & validation_roles for option in support_options
        )
    if requirement_has_validation_roles and not validation_role_is_supported:
        return ()
    requirement_has_no_validation_roles = validation_roles is None
    candidate_has_no_support_options = not support_options
    if requirement_has_no_validation_roles and candidate_has_no_support_options:
        intrinsic_support_is_allowed = (
            shape_spec.allows_intrinsic_support_for_requirement(requirement_id)
        )
        candidate_has_intrinsic_source = _candidate_has_intrinsic_source(candidate)
        if intrinsic_support_is_allowed and candidate_has_intrinsic_source:
            return ((),)
        return ()
    binding_roles = shape_spec.binding_roles_for_requirement(
        requirement_id,
        support_options=support_options,
    )
    if binding_roles is None:
        support_sets = _answer_output_fulfillment_support_sets(
            candidate,
            required_answer_output_ids=required_answer_output_ids,
        )
        if support_sets:
            return (support_sets,)
        intrinsic_support_is_allowed = (
            shape_spec.allows_intrinsic_support_for_requirement(requirement_id)
        )
        candidate_has_intrinsic_source = _candidate_has_intrinsic_source(candidate)
        intrinsic_support_option_exists = _has_intrinsic_support_option(support_options)
        intrinsic_support_matches_requirement = _intrinsic_support_matches_requirement(
            support_options,
            requirement_id=requirement_id,
        )
        if (
            intrinsic_support_is_allowed
            and candidate_has_intrinsic_source
            and intrinsic_support_option_exists
            and intrinsic_support_matches_requirement
        ):
            return ((),)
        return ()
    if not binding_roles:
        return ((),)
    selected_ids = {
        option.binding_support_set_id or option.support_set_id
        for option in support_options
        if option.support_roles and option.support_roles <= binding_roles
    }
    support_sets = tuple(
        support_set
        for support_set in _raw_fulfillment_support_sets(candidate)
        if _support_set_binding_id(support_set) in selected_ids
    )
    no_binding_support_sets_selected = not support_sets
    intrinsic_support_is_allowed = shape_spec.allows_intrinsic_support_for_requirement(
        requirement_id
    )
    intrinsic_support_option_exists = _has_intrinsic_support_option(
        support_options,
        binding_roles=binding_roles,
    )
    intrinsic_support_matches_requirement = _intrinsic_support_matches_requirement(
        support_options,
        requirement_id=requirement_id,
    )
    if (
        no_binding_support_sets_selected
        and intrinsic_support_is_allowed
        and intrinsic_support_option_exists
        and intrinsic_support_matches_requirement
    ):
        return ((),)
    return shape_spec.support_set_groups_for_requirement(
        support_sets,
        requirement_id=requirement_id,
        required_answer_output_ids=required_answer_output_ids,
        source_candidate_id=candidate.id,
    )


def _has_intrinsic_support_option(
    support_options: tuple[PlanSelectionSupportOption, ...],
    *,
    binding_roles: frozenset[str] | None = None,
) -> bool:
    return any(
        _support_option_is_intrinsic(
            option,
            binding_roles=binding_roles,
        )
        for option in support_options
    )


def _support_option_is_intrinsic(
    option: PlanSelectionSupportOption,
    *,
    binding_roles: frozenset[str] | None,
) -> bool:
    option_has_no_binding_support_set = not option.binding_support_set_id
    support_roles = option.support_roles
    option_has_support_roles = bool(support_roles)
    option_roles_match_binding_roles = (
        binding_roles is None or support_roles <= binding_roles
    )
    return (
        option_has_no_binding_support_set
        and option_has_support_roles
        and option_roles_match_binding_roles
    )


def _intrinsic_support_matches_requirement(
    support_options: tuple[PlanSelectionSupportOption, ...],
    *,
    requirement_id: str,
) -> bool:
    if requirement_id in {"value_1", "value_2"}:
        return True
    return not any("VALUE_SOURCE" in option.support_roles for option in support_options)


def _candidate_has_intrinsic_source(candidate: SourceCandidate) -> bool:
    if candidate.value_id:
        return True
    if candidate.read_id or candidate.memory_relation_id:
        return bool(_candidate_field_ids(candidate, support_sets=()))
    return False


def _source_strategy_members(
    combo: tuple[
        tuple[str, SourceCandidate, tuple[FulfillmentSupportSet, ...]], ...
    ],
    *,
    relation_catalog: RelationCatalog,
) -> tuple[SourceStrategyMember, ...]:
    support_sets_by_candidate: dict[str, list[FulfillmentSupportSet]] = {}
    requirement_ids_by_candidate: dict[str, list[str]] = {}
    candidates_by_id: dict[str, SourceCandidate] = {}
    for requirement_id, candidate, support_sets in combo:
        source_candidate_id = candidate.id
        if not source_candidate_id:
            continue
        candidates_by_id[source_candidate_id] = candidate
        support_sets_by_candidate.setdefault(source_candidate_id, [])
        requirement_ids = requirement_ids_by_candidate.setdefault(
            source_candidate_id, []
        )
        if requirement_id and requirement_id not in requirement_ids:
            requirement_ids.append(requirement_id)
        for support_set in support_sets:
            support_set_id = _support_set_binding_id(support_set)
            support_set_has_binding_id = bool(support_set_id)
            support_set_already_selected = any(
                _support_set_binding_id(item) == support_set_id
                for item in support_sets_by_candidate[source_candidate_id]
            )
            if support_set_has_binding_id and not support_set_already_selected:
                support_sets_by_candidate[source_candidate_id].append(support_set)
    output: list[SourceStrategyMember] = []
    for source_candidate_id, selected_support_sets in support_sets_by_candidate.items():
        candidate = candidates_by_id[source_candidate_id]
        output.append(
            SourceStrategyMember(
                source_candidate_id=source_candidate_id,
                requirement_ids=tuple(
                    requirement_ids_by_candidate.get(source_candidate_id, ())
                ),
                fulfillment_support_set_ids=tuple(
                    _support_set_binding_id(support_set)
                    for support_set in selected_support_sets
                    if _support_set_binding_id(support_set)
                ),
                kind=candidate.kind,
                read_id=candidate.read_id,
                value_id=candidate.value_id,
                memory_relation_id=candidate.memory_relation_id,
                source_relation_id=candidate.source_relation_id,
                calendar_id=candidate.calendar_id,
                field_ids=_candidate_field_ids(
                    candidate,
                    support_sets=tuple(selected_support_sets),
                ),
                answer_output_ids=_support_set_answer_output_ids(
                    tuple(selected_support_sets)
                ),
                operation_evidence=_operation_evidence(tuple(selected_support_sets)),
                applied_filters=candidate.applied_filters,
                source_interface=_source_interface(
                    candidate,
                    tuple(selected_support_sets),
                    relation_catalog=relation_catalog,
                ),
            )
        )
    return tuple(output)


def _operation_evidence(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[OperationEvidence, ...]:
    output: list[OperationEvidence] = []
    seen: set[tuple[str, str]] = set()
    for support_set in support_sets:
        for slot in support_set.fulfillment_slots:
            for default_kind, items in (
                ("entity", slot.entity_evidence),
                ("metric", slot.metric_measure_evidence),
                ("value", slot.value_evidence),
                ("row_count", slot.row_count_basis_evidence),
            ):
                for evidence in items:
                    evidence_id = evidence.evidence_id
                    field_ids = _evidence_field_ids(evidence)
                    kind = evidence.type if default_kind == "entity" else default_kind
                    if not evidence_id or not field_ids:
                        continue
                    for field_id in field_ids:
                        dedupe_key = (kind, f"{evidence_id}:{field_id}")
                        if dedupe_key in seen:
                            continue
                        seen.add(dedupe_key)
                        output.append(
                            _operation_evidence_item(
                                evidence,
                                kind=kind,
                                evidence_id=evidence_id,
                                field_id=field_id,
                            )
                        )
    return tuple(output)


def _operation_evidence_item(
    evidence: EvidenceItem,
    *,
    kind: str,
    evidence_id: str,
    field_id: str,
) -> OperationEvidence:
    row_path_id = evidence_row_path_id(evidence)
    return OperationEvidence(
        kind=kind,
        evidence_id=evidence_id,
        field_id=field_id,
        row_path_id=row_path_id,
        evidence_type=evidence.type,
    )


def _operation_evidence_payload(item: OperationEvidence) -> dict[str, str]:
    output = {
        "kind": item.kind,
        "evidence_id": item.evidence_id,
        "field_id": item.field_id,
    }
    if item.row_path_id:
        output["row_path_id"] = item.row_path_id
    if item.evidence_type:
        output["type"] = item.evidence_type
    return output


def _candidate_field_ids(
    candidate: SourceCandidate,
    *,
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[str, ...]:
    output: list[str] = []
    for field_id in _support_set_field_ids(support_sets):
        _append_unique(output, field_id)
    for field_id in candidate.applied_filter_field_ids:
        _append_unique(output, field_id)
    return tuple(output)


def _support_set_binding_id(support_set: FulfillmentSupportSet) -> str:
    return support_set.fulfillment_support_set_id or support_set.fulfillment_choice_id


def _support_set_field_ids(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[str, ...]:
    output: list[str] = []
    for support_set in support_sets:
        for slot in support_set.fulfillment_slots:
            for evidence in (
                *slot.metric_measure_evidence,
                *slot.value_evidence,
                *slot.entity_evidence,
            ):
                for field_id in _evidence_field_ids(evidence):
                    _append_unique(output, field_id)
    return tuple(output)


def _evidence_field_ids(item: EvidenceItem) -> tuple[str, ...]:
    return evidence_field_ids(item)


def _members_cover_required_answer_outputs(
    members: tuple[SourceStrategyMember, ...],
    *,
    required_answer_output_ids: tuple[str, ...],
    shape_spec: PlanSelectionShapeSpec,
) -> bool:
    fulfillment_members = tuple(
        member
        for member in members
        if any(
            shape_spec.requires_complete_answer_fulfillment_for_requirement(
                requirement_id
            )
            for requirement_id in member.requirement_ids
        )
    )
    if not fulfillment_members:
        return True
    covered = tuple(
        str(answer_output_id)
        for member in fulfillment_members
        for answer_output_id in member.answer_output_ids
    )
    return set(required_answer_output_ids) <= set(covered)


def _source_strategy_member_id_suffix(
    members: tuple[SourceStrategyMember, ...],
) -> str:
    return "_and_".join(_source_strategy_member_id_part(member) for member in members)


def _source_strategy_member_id_part(member: SourceStrategyMember) -> str:
    if not member.fulfillment_support_set_ids:
        return member.source_candidate_id
    digest = hashlib.sha256(
        "\n".join(member.fulfillment_support_set_ids).encode("utf-8")
    ).hexdigest()[:12]
    return f"{member.source_candidate_id}_{digest}"


def _append_unique(output: list[str], value: str) -> None:
    if value and value not in output:
        output.append(value)


def _raw_fulfillment_support_sets(
    candidate: SourceCandidate,
) -> tuple[FulfillmentSupportSet, ...]:
    return tuple(
        item
        for item in plan_selection_fulfillment_support_sets(candidate)
        if _support_set_binding_id(item)
    )


def _answer_output_fulfillment_support_sets(
    candidate: SourceCandidate,
    *,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[FulfillmentSupportSet, ...]:
    required = set(required_answer_output_ids)
    return tuple(
        support_set
        for support_set in _raw_fulfillment_support_sets(candidate)
        if support_set.answer_output_id in required
    )
