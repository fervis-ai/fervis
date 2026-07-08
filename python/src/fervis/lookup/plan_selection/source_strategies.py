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
    SourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.plan_selection.support_options import (
    plan_selection_fulfillment_support_sets,
    plan_selection_support_options,
)


def source_strategies_by_fact(
    payload: dict[str, Any],
    *,
    requested_facts: tuple[RequestedFact, ...],
    relation_catalog: RelationCatalog,
    shape_specs_for_family: Callable[
        [RequestedFactAnswerExpressionFamily],
        tuple[PlanSelectionShapeSpec, ...],
    ],
) -> dict[str, tuple[SourceStrategy, ...]]:
    candidates_by_fact = _source_candidates_by_fact(
        payload,
        requested_fact_ids=tuple(fact.id for fact in requested_facts),
    )
    output: dict[str, tuple[SourceStrategy, ...]] = {}
    for fact in requested_facts:
        output[fact.id] = _source_strategies_for_fact(
            fact=fact,
            candidates=candidates_by_fact.get(fact.id, ()),
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
        output["operation_evidence"] = list(member.operation_evidence)
    if member.source_interface:
        input_params = member.source_interface.get("input_params")
        if input_params:
            output["input_params"] = input_params
        response_rows = member.source_interface.get("response_rows")
        if response_rows:
            output["response_rows"] = response_rows
    return output


def _source_interface(
    candidate: dict[str, Any],
    support_sets: tuple[dict[str, Any], ...],
    *,
    relation_catalog: RelationCatalog,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    answer_output_ids = _support_set_answer_output_ids(support_sets)
    if answer_output_ids:
        summary["answer_output_ids"] = list(answer_output_ids)
    if candidate.get("kind") in {"new_api_read", "same_scope_api_read"}:
        read_shape = ApiReadResponseShapeProjector(
            relation_catalog.read(str(candidate["read_id"]))
        )
        input_params = read_shape.input_params()
        if input_params:
            summary["input_params"] = input_params
        summary["response_rows"] = read_shape.response_rows(
            row_path_ids=_candidate_row_path_ids(candidate),
        )
    return summary


def _support_set_answer_output_ids(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            answer_output_id
            for support_set in support_sets
            for answer_output_id in (str(support_set.get("answer_output_id") or ""),)
            if answer_output_id
        )
    )


def _source_candidates_by_fact(
    payload: dict[str, Any],
    *,
    requested_fact_ids: tuple[str, ...],
) -> dict[str, tuple[dict[str, Any], ...]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        requested_fact_id = str(fact_sources.get("requested_fact_id") or "")
        if not requested_fact_id:
            continue
        for context in fact_sources.get("source_contexts") or ():
            if not isinstance(context, dict):
                continue
            for candidate in context.get("source_options") or ():
                if isinstance(candidate, dict):
                    _add_source_candidate(
                        output,
                        requested_fact_id=requested_fact_id,
                        candidate=candidate,
                    )
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if not isinstance(candidate, dict):
                continue
            applies_to = (
                candidate.get("applies_to_requested_facts") or requested_fact_ids
            )
            for requested_fact_id in applies_to:
                _add_source_candidate(
                    output,
                    requested_fact_id=str(requested_fact_id),
                    candidate=candidate,
                )
    return {key: tuple(value) for key, value in output.items()}


def _add_source_candidate(
    output: dict[str, list[dict[str, Any]]],
    *,
    requested_fact_id: str,
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not requested_fact_id or not source_candidate_id:
        return
    existing = output.setdefault(requested_fact_id, [])
    if not any(
        str(item.get("source_candidate_id") or "") == source_candidate_id
        for item in existing
    ):
        existing.append(candidate)


def _source_strategies_for_fact(
    *,
    fact: RequestedFact,
    candidates: tuple[dict[str, Any], ...],
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
                    str(candidate.get("source_candidate_id") or "")
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
    candidates: tuple[dict[str, Any], ...],
    *,
    requirement_id: str,
    shape_spec: PlanSelectionShapeSpec,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[tuple[str, dict[str, Any], tuple[dict[str, Any], ...]], ...]:
    output: list[tuple[str, dict[str, Any], tuple[dict[str, Any], ...]]] = []
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
    candidate: dict[str, Any],
    *,
    requirement_id: str,
    shape_spec: PlanSelectionShapeSpec,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    support_options = plan_selection_support_options(candidate)
    validation_roles = shape_spec.validation_roles_for_requirement(requirement_id)
    requirement_has_validation_roles = validation_roles is not None
    validation_role_is_supported = (
        any(
            set(option.get("support_roles") or ()) & validation_roles
            for option in support_options
        )
        if requirement_has_validation_roles
        else True
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
        if shape_spec.support_set_grouper is not None:
            return shape_spec.support_set_groups_for_requirement(
                support_sets,
                requirement_id=requirement_id,
                required_answer_output_ids=required_answer_output_ids,
                source_candidate_id=str(candidate.get("source_candidate_id") or ""),
            )
        if support_sets:
            return (support_sets,)
        intrinsic_support_is_allowed = (
            shape_spec.allows_intrinsic_support_for_requirement(requirement_id)
        )
        candidate_has_intrinsic_source = _candidate_has_intrinsic_source(candidate)
        intrinsic_support_option_exists = _has_intrinsic_support_option(
            support_options
        )
        intrinsic_support_matches_requirement = (
            _intrinsic_support_matches_requirement(
                support_options,
                requirement_id=requirement_id,
            )
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
        str(option.get("binding_support_set_id") or option.get("support_set_id") or "")
        for option in support_options
        if set(option.get("support_roles") or ())
        and set(option.get("support_roles") or ()) <= binding_roles
    }
    support_sets = tuple(
        support_set
        for support_set in _raw_fulfillment_support_sets(candidate)
        if _support_set_binding_id(support_set) in selected_ids
    )
    no_binding_support_sets_selected = not support_sets
    intrinsic_support_is_allowed = (
        shape_spec.allows_intrinsic_support_for_requirement(requirement_id)
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
        source_candidate_id=str(candidate.get("source_candidate_id") or ""),
    )


def _has_intrinsic_support_option(
    support_options: tuple[dict[str, object], ...],
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
    option: dict[str, object],
    *,
    binding_roles: frozenset[str] | None,
) -> bool:
    option_has_no_binding_support_set = not option.get("binding_support_set_id")
    support_roles = set(option.get("support_roles") or ())
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
    support_options: tuple[dict[str, object], ...],
    *,
    requirement_id: str,
) -> bool:
    if requirement_id in {"value_1", "value_2"}:
        return True
    return not any(
        "VALUE_SOURCE" in set(option.get("support_roles") or ())
        for option in support_options
    )


def _candidate_has_intrinsic_source(candidate: dict[str, Any]) -> bool:
    if str(candidate.get("value_id") or ""):
        return True
    if str(candidate.get("read_id") or "") or str(
        candidate.get("memory_relation_id") or ""
    ):
        return bool(_candidate_field_ids(candidate, support_sets=()))
    return False


def _source_strategy_members(
    combo: tuple[tuple[str, dict[str, Any], tuple[dict[str, Any], ...]], ...],
    *,
    relation_catalog: RelationCatalog,
) -> tuple[SourceStrategyMember, ...]:
    support_sets_by_candidate: dict[str, list[dict[str, Any]]] = {}
    requirement_ids_by_candidate: dict[str, list[str]] = {}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for requirement_id, candidate, support_sets in combo:
        source_candidate_id = str(candidate.get("source_candidate_id") or "")
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
    for source_candidate_id, support_sets in support_sets_by_candidate.items():
        candidate = candidates_by_id[source_candidate_id]
        output.append(
            SourceStrategyMember(
                source_candidate_id=source_candidate_id,
                requirement_ids=tuple(
                    requirement_ids_by_candidate.get(source_candidate_id, ())
                ),
                fulfillment_support_set_ids=tuple(
                    _support_set_binding_id(support_set)
                    for support_set in support_sets
                    if _support_set_binding_id(support_set)
                ),
                kind=str(candidate.get("kind") or ""),
                read_id=str(candidate.get("read_id") or ""),
                value_id=str(candidate.get("value_id") or ""),
                memory_relation_id=str(candidate.get("memory_relation_id") or ""),
                source_relation_id=str(candidate.get("source_relation_id") or ""),
                calendar_id=str(candidate.get("calendar_id") or ""),
                field_ids=_candidate_field_ids(
                    candidate,
                    support_sets=tuple(support_sets),
                ),
                operation_evidence=_operation_evidence(tuple(support_sets)),
                source_interface=_source_interface(
                    candidate,
                    tuple(support_sets),
                    relation_catalog=relation_catalog,
                ),
            )
        )
    return tuple(output)


def _operation_evidence(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for support_set in support_sets:
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for kind, key in (
                ("group_key", "group_key_evidence"),
                ("metric", "metric_measure_evidence"),
                ("row_count", "row_count_basis_evidence"),
            ):
                for item in slot.get(key) or ():
                    if not isinstance(item, dict):
                        continue
                    evidence_id = str(item.get("evidence_id") or "")
                    field_id = str(item.get("field_id") or "")
                    if not evidence_id or not field_id:
                        continue
                    dedupe_key = (kind, evidence_id)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    output.append(
                        _operation_evidence_item(
                            item,
                            kind=kind,
                            evidence_id=evidence_id,
                            field_id=field_id,
                        )
                    )
    return tuple(output)


def _operation_evidence_item(
    item: dict[str, Any],
    *,
    kind: str,
    evidence_id: str,
    field_id: str,
) -> dict[str, Any]:
    output = {
        "kind": kind,
        "evidence_id": evidence_id,
        "field_id": field_id,
    }
    for key in ("row_path_id", "type"):
        value = str(item.get(key) or "")
        if value:
            output[key] = value
    return output


def _candidate_row_path_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            row_path_id
            for grain in candidate.get("result_grains") or ()
            if isinstance(grain, dict)
            for row_path_id in (str(grain.get("row_path_id") or ""),)
            if row_path_id
        )
    )


def _candidate_field_ids(
    candidate: dict[str, Any],
    *,
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    output: list[str] = []
    for field_id in _support_set_field_ids(support_sets):
        _append_unique(output, field_id)
    for key in ("available_field_ids", "field_ids"):
        for field_id in candidate.get(key) or ():
            _append_unique(output, str(field_id or ""))
    for field in candidate.get("fields") or ():
        if isinstance(field, dict):
            _append_unique(
                output,
                str(field.get("field_id") or field.get("id") or ""),
            )
    binding_surface = candidate.get("binding_surface")
    if isinstance(binding_surface, dict):
        for field in binding_surface.get("fields") or ():
            if isinstance(field, dict):
                _append_unique(
                    output,
                    str(field.get("field_id") or field.get("id") or ""),
                )
        for field in binding_surface.get("evidence_items") or ():
            if isinstance(field, dict):
                _append_unique(output, str(field.get("field_id") or ""))
    for support_set in _raw_fulfillment_support_sets(candidate):
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for evidence_key in (
                "metric_measure_evidence",
                "scope_evidence",
                "group_key_evidence",
            ):
                for item in slot.get(evidence_key) or ():
                    if isinstance(item, dict):
                        _append_unique(output, str(item.get("field_id") or ""))
    return tuple(output)


def _support_set_binding_id(support_set: dict[str, Any]) -> str:
    return str(
        support_set.get("fulfillment_support_set_id")
        or support_set.get("fulfillment_choice_id")
        or ""
    )


def _support_set_field_ids(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    output: list[str] = []
    for support_set in support_sets:
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for evidence_key in (
                "metric_measure_evidence",
                "scope_evidence",
                "group_key_evidence",
            ):
                for item in slot.get(evidence_key) or ():
                    if isinstance(item, dict):
                        _append_unique(output, str(item.get("field_id") or ""))
    return tuple(output)


def _members_cover_required_answer_outputs(
    members: tuple[SourceStrategyMember, ...],
    *,
    required_answer_output_ids: tuple[str, ...],
) -> bool:
    covered = tuple(
        str(answer_output_id)
        for member in members
        if isinstance(member.source_interface, dict)
        for answer_output_id in member.source_interface.get("answer_output_ids") or ()
        if str(answer_output_id)
    )
    if not covered:
        return True
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
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for item in plan_selection_fulfillment_support_sets(candidate)
        if isinstance(item, dict) and _support_set_binding_id(item)
    )


def _answer_output_fulfillment_support_sets(
    candidate: dict[str, Any],
    *,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    required = set(required_answer_output_ids)
    return tuple(
        support_set
        for support_set in _raw_fulfillment_support_sets(candidate)
        if str(support_set.get("answer_output_id") or "") in required
    )
