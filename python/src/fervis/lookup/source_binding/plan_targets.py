"""Source-binding targets derived from selected plan members."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fervis.lookup.operation_families.plan_selection_registry import (
    plan_selection_shape_specs_for_family,
)
from fervis.lookup.plan_selection import (
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.plan_selection.family_specs import (
    PlanSelectionShapeSpec,
    SourceMemberConstraint,
)
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.source_binding.model import SourceBindingRequest


_DEFAULT_REQUIREMENT_ID = "source"
_FACT_BINDING_FIELD_PREFIX = "bindings_for_"


def source_binding_fact_field_id(requested_fact_id: str) -> str:
    return f"{_FACT_BINDING_FIELD_PREFIX}{requested_fact_id}"


def source_binding_fact_id_from_field(field_id: str) -> str | None:
    if not field_id.startswith(_FACT_BINDING_FIELD_PREFIX):
        return None
    requested_fact_id = field_id.removeprefix(_FACT_BINDING_FIELD_PREFIX)
    return requested_fact_id or None
@dataclass(frozen=True)
class SourceBindingTarget:
    binding_target_id: str
    requested_fact_id: str
    plan_shape: str
    source_candidate_id: str
    requirement_id: str
    answer_output_ids: tuple[str, ...] = ()
    required_answer_output_ids: tuple[str, ...] = ()

    @property
    def requires_answer_fulfillment(self) -> bool:
        return bool(self.answer_output_ids)

    def to_payload(self) -> dict[str, object]:
        return {
            "binding_target_id": self.binding_target_id,
            "requested_fact_id": self.requested_fact_id,
            "plan_shape": self.plan_shape,
            "source_candidate_id": self.source_candidate_id,
            "requirement_id": self.requirement_id,
            "answer_output_ids": list(self.answer_output_ids),
            "required_answer_output_ids": list(self.required_answer_output_ids),
        }


@dataclass(frozen=True)
class SourceBindingTargetCompatibility:
    binding_target_id: str
    plan_selection_id: str
    source_strategy_id: str
    requested_fact_id: str
    plan_shape: str
    source_candidate_id: str
    requirement_id: str


@dataclass(frozen=True)
class SourceBindingTargetIndex:
    targets: tuple[SourceBindingTarget, ...]
    compatibilities: tuple[SourceBindingTargetCompatibility, ...] = ()

    def __post_init__(self) -> None:
        target_ids = tuple(target.binding_target_id for target in self.targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("source binding target ids must be unique")
        known_target_ids = set(target_ids)
        for compatibility in self.compatibilities:
            if compatibility.binding_target_id not in known_target_ids:
                raise ValueError("source binding target compatibility is unowned")

    @property
    def by_id(self) -> dict[str, SourceBindingTarget]:
        return {target.binding_target_id: target for target in self.targets}

    def require(self, binding_target_id: str) -> SourceBindingTarget:
        target = self.by_id.get(binding_target_id)
        if target is None:
            raise ValueError("source binding references unknown binding target")
        return target

    def target_ids_by_plan(self) -> dict[str, frozenset[str]]:
        output: dict[str, set[str]] = {}
        for compatibility in self.compatibilities:
            output.setdefault(compatibility.plan_selection_id, set()).add(
                compatibility.binding_target_id
            )
        target_ids_by_plan = {
            plan_id: frozenset(target_ids) for plan_id, target_ids in output.items()
        }
        return target_ids_by_plan

    def payload(self) -> tuple[dict[str, object], ...]:
        target_payloads = tuple(target.to_payload() for target in self.targets)
        return target_payloads


@dataclass(frozen=True)
class SourceBindingPlanFamily:
    requested_fact_id: str
    plan_shape: str
    member_constraint: SourceMemberConstraint
    required_answer_output_ids: tuple[str, ...]
    role_targets: tuple[tuple[str, tuple[SourceBindingTarget, ...]], ...]

    def payload(
        self,
        *,
        target_payload: Callable[[SourceBindingTarget], dict[str, object]]
        | None = None,
    ) -> dict[str, object]:
        render_target = target_payload or SourceBindingTarget.to_payload
        role_targets = {
            role_id: [render_target(target) for target in targets]
            for role_id, targets in self.role_targets
        }
        return {
            "member_constraint": self.member_constraint.value,
            "required_answer_output_ids": list(self.required_answer_output_ids),
            "role_targets": role_targets,
        }


def source_binding_plan_families(
    request: SourceBindingRequest,
    *,
    target_index: SourceBindingTargetIndex | None = None,
    visible_targets: tuple[SourceBindingTarget, ...] | None = None,
) -> tuple[SourceBindingPlanFamily, ...]:
    index = target_index or source_binding_target_index(request)
    visible_targets = visible_targets if visible_targets is not None else index.targets
    visible_target_ids = frozenset(
        target.binding_target_id for target in visible_targets
    )
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    plans_by_family = _visible_plans_by_family(
        request.plan_selection,
        target_index=index,
        visible_target_ids=visible_target_ids,
    )
    families: list[SourceBindingPlanFamily] = []
    for family_key, plans in plans_by_family.items():
        family = _source_binding_plan_family(
            family_key,
            plans,
            target_index=index,
            visible_target_ids=visible_target_ids,
            facts_by_id=facts_by_id,
        )
        if family is not None:
            families.append(family)
    return tuple(families)


def _visible_plans_by_family(
    plan_selection: PlanSelectionSet,
    *,
    target_index: SourceBindingTargetIndex,
    visible_target_ids: frozenset[str],
) -> dict[tuple[str, str], tuple[SelectedSourceStrategy, ...]]:
    plans_by_family: dict[tuple[str, str], list[SelectedSourceStrategy]] = {}
    target_ids_by_plan = target_index.target_ids_by_plan()
    for plan in plan_selection.plan_selections:
        plan_target_ids = target_ids_by_plan.get(plan.plan_selection_id, frozenset())
        if plan_target_ids and plan_target_ids <= visible_target_ids:
            plans_by_family.setdefault(
                (plan.requested_fact_id, plan.plan_shape), []
            ).append(plan)
    frozen_families = {
        key: tuple(plans) for key, plans in plans_by_family.items()
    }
    return frozen_families


def _source_binding_plan_family(
    family_key: tuple[str, str],
    plans: tuple[SelectedSourceStrategy, ...],
    *,
    target_index: SourceBindingTargetIndex,
    visible_target_ids: frozenset[str],
    facts_by_id: dict[str, RequestedFact],
) -> SourceBindingPlanFamily | None:
    requested_fact_id, plan_shape = family_key
    shape_spec = _shape_spec(plans[0], fact=facts_by_id.get(requested_fact_id))
    requirement_ids = _family_requirement_ids(plans, shape_spec=shape_spec)
    role_targets = _family_role_targets(
        plans,
        requirement_ids=requirement_ids,
        target_index=target_index,
        visible_target_ids=visible_target_ids,
    )
    if any(not targets for _, targets in role_targets):
        return None
    return SourceBindingPlanFamily(
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
        member_constraint=(
            shape_spec.member_constraint
            if shape_spec is not None
            else SourceMemberConstraint.ANY
        ),
        required_answer_output_ids=_family_required_answer_output_ids(plans),
        role_targets=role_targets,
    )


def _family_requirement_ids(
    plans: tuple[SelectedSourceStrategy, ...],
    *,
    shape_spec: PlanSelectionShapeSpec | None,
) -> tuple[str, ...]:
    if shape_spec is not None:
        return shape_spec.member_requirements
    requirement_ids = dict.fromkeys(
        requirement_id
        for plan in plans
        for member in plan.source_members
        for requirement_id in source_binding_member_requirement_ids(member)
    )
    return tuple(requirement_ids)


def _family_role_targets(
    plans: tuple[SelectedSourceStrategy, ...],
    *,
    requirement_ids: tuple[str, ...],
    target_index: SourceBindingTargetIndex,
    visible_target_ids: frozenset[str],
) -> tuple[tuple[str, tuple[SourceBindingTarget, ...]], ...]:
    plan_ids = {plan.plan_selection_id for plan in plans}
    role_targets: list[tuple[str, tuple[SourceBindingTarget, ...]]] = []
    for requirement_id in requirement_ids:
        target_ids = dict.fromkeys(
            compatibility.binding_target_id
            for compatibility in target_index.compatibilities
            if compatibility.plan_selection_id in plan_ids
            and compatibility.requirement_id == requirement_id
            and compatibility.binding_target_id in visible_target_ids
        )
        targets = tuple(target_index.require(target_id) for target_id in target_ids)
        role_targets.append((requirement_id, targets))
    return tuple(role_targets)


def _family_required_answer_output_ids(
    plans: tuple[SelectedSourceStrategy, ...],
) -> tuple[str, ...]:
    output_ids = dict.fromkeys(
        output_id for plan in plans for output_id in plan.required_answer_output_ids
    )
    return tuple(output_ids)


def source_binding_target_index(
    request: SourceBindingRequest,
) -> SourceBindingTargetIndex:
    return source_binding_target_index_for_plan_selection(
        request.plan_selection,
        requested_facts=request.requested_facts,
    )


def source_binding_targets(
    request: SourceBindingRequest,
) -> tuple[SourceBindingTarget, ...]:
    return source_binding_target_index(request).targets


def source_binding_targets_for_plan_selection(
    plan_selection: PlanSelectionSet,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> tuple[SourceBindingTarget, ...]:
    return source_binding_target_index_for_plan_selection(
        plan_selection,
        requested_facts=requested_facts,
    ).targets


def source_binding_target_index_for_plan_selection(
    plan_selection: PlanSelectionSet,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> SourceBindingTargetIndex:
    facts_by_id = {fact.id: fact for fact in requested_facts}
    targets_by_id: dict[str, SourceBindingTarget] = {}
    compatibilities: list[SourceBindingTargetCompatibility] = []
    for target, compatibility in (
        pair
        for plan in plan_selection.plan_selections
        for pair in _targets_for_plan(
            plan,
            fact=facts_by_id.get(plan.requested_fact_id),
        )
    ):
        existing = targets_by_id.get(target.binding_target_id)
        if existing is None:
            targets_by_id[target.binding_target_id] = target
        elif existing != target:
            raise ValueError("compact source binding target has conflicting shape")
        compatibilities.append(compatibility)
    return SourceBindingTargetIndex(
        targets=tuple(targets_by_id.values()),
        compatibilities=tuple(compatibilities),
    )


def _targets_for_plan(
    plan: SelectedSourceStrategy,
    *,
    fact: RequestedFact | None,
) -> tuple[tuple[SourceBindingTarget, SourceBindingTargetCompatibility], ...]:
    shape_spec = _shape_spec(plan, fact=fact)
    return tuple(
        _target_for_requirement(
            plan,
            member=member,
            requirement_id=requirement_id,
            shape_spec=shape_spec,
        )
        for member in plan.source_members
        for requirement_id in source_binding_member_requirement_ids(member)
    )


def _target_for_requirement(
    plan: SelectedSourceStrategy,
    *,
    member: SourceStrategyMember,
    requirement_id: str,
    shape_spec: PlanSelectionShapeSpec | None,
) -> tuple[SourceBindingTarget, SourceBindingTargetCompatibility]:
    answer_output_ids = ()
    if _requires_answer_fulfillment(
        requirement_id,
        shape_spec=shape_spec,
    ):
        answer_output_ids = _member_answer_output_ids(member) or (
            plan.required_answer_output_ids
        )
    target = SourceBindingTarget(
        binding_target_id=_binding_target_id(
            requested_fact_id=plan.requested_fact_id,
            plan_shape=plan.plan_shape,
            source_candidate_id=member.source_candidate_id,
            requirement_id=requirement_id,
        ),
        requested_fact_id=plan.requested_fact_id,
        plan_shape=plan.plan_shape,
        source_candidate_id=member.source_candidate_id,
        requirement_id=requirement_id,
        answer_output_ids=answer_output_ids,
        required_answer_output_ids=(
            answer_output_ids
            if _requires_complete_answer_fulfillment(
                requirement_id,
                shape_spec=shape_spec,
            )
            else ()
        ),
    )
    return (
        target,
        SourceBindingTargetCompatibility(
            binding_target_id=target.binding_target_id,
            plan_selection_id=plan.plan_selection_id,
            requested_fact_id=plan.requested_fact_id,
            source_strategy_id=plan.source_strategy_id,
            plan_shape=plan.plan_shape,
            source_candidate_id=member.source_candidate_id,
            requirement_id=requirement_id,
        ),
    )


def _binding_target_id(
    *,
    requested_fact_id: str,
    plan_shape: str,
    source_candidate_id: str,
    requirement_id: str,
) -> str:
    return f"target.{requested_fact_id}.{plan_shape}.{source_candidate_id}.{requirement_id}"


def source_binding_member_requirement_ids(
    member: SourceStrategyMember,
) -> tuple[str, ...]:
    return member.requirement_ids or (_DEFAULT_REQUIREMENT_ID,)


def _member_answer_output_ids(
    member: SourceStrategyMember,
) -> tuple[str, ...]:
    source_interface = member.source_interface
    if not isinstance(source_interface, dict):
        return ()
    return tuple(
        dict.fromkeys(
            answer_output_id
            for raw in source_interface.get("answer_output_ids") or ()
            for answer_output_id in (str(raw or ""),)
            if answer_output_id
        )
    )


def _requires_answer_fulfillment(
    requirement_id: str,
    *,
    shape_spec: PlanSelectionShapeSpec | None,
) -> bool:
    if shape_spec is None or requirement_id == _DEFAULT_REQUIREMENT_ID:
        return True
    return shape_spec.requires_answer_fulfillment_for_requirement(requirement_id)


def _requires_complete_answer_fulfillment(
    requirement_id: str,
    *,
    shape_spec: PlanSelectionShapeSpec | None,
) -> bool:
    if shape_spec is None or requirement_id == _DEFAULT_REQUIREMENT_ID:
        return True
    return shape_spec.requires_complete_answer_fulfillment_for_requirement(
        requirement_id
    )


def _shape_spec(
    plan: SelectedSourceStrategy,
    *,
    fact: RequestedFact | None,
) -> PlanSelectionShapeSpec | None:
    if fact is None or fact.answer_expression is None:
        return None
    return next(
        (
            spec
            for spec in plan_selection_shape_specs_for_family(
                fact.answer_expression.family
            )
            if spec.plan_shape == plan.plan_shape
        ),
        None,
    )
