"""Source-binding targets derived from selected plan members."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.operation_families.plan_selection_registry import (
    plan_selection_shape_specs_for_family,
)
from fervis.lookup.plan_selection import (
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.plan_selection.family_specs import PlanSelectionShapeSpec
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.source_binding.model import SourceBindingRequest


_DEFAULT_REQUIREMENT_ID = "source"


@dataclass(frozen=True)
class SourceBindingTarget:
    binding_target_id: str
    requested_fact_id: str
    plan_shape: str
    source_candidate_id: str
    requirement_id: str
    answer_output_ids: tuple[str, ...] = ()

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
        return {plan_id: frozenset(target_ids) for plan_id, target_ids in output.items()}

    def payload(self) -> tuple[dict[str, object], ...]:
        return tuple(target.to_payload() for target in self.targets)


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
        answer_output_ids=(
            plan.required_answer_output_ids
            if _requires_answer_fulfillment(
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


def _requires_answer_fulfillment(
    requirement_id: str,
    *,
    shape_spec: PlanSelectionShapeSpec | None,
) -> bool:
    if shape_spec is None or requirement_id == _DEFAULT_REQUIREMENT_ID:
        return True
    return shape_spec.requires_answer_fulfillment_for_requirement(requirement_id)


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
