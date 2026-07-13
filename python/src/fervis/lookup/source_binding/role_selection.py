"""Role-bound source binding selection for compact target payloads."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.plan_selection import (
    BoundPlanSelectionSet,
    BoundRoleTarget,
    BoundSelectedSourceStrategy,
    BoundSourceStrategyMember,
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.source_binding.model import (
    AnswerPopulation,
    BoundSource,
    SourceBindingPlan,
)
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
    SourceBindingTargetIndex,
    source_binding_member_requirement_ids,
    source_binding_target_index_for_plan_selection,
)


@dataclass(frozen=True)
class RoleBoundSource:
    target: SourceBindingTarget
    source: BoundSource


@dataclass(frozen=True)
class SourceBindingRoleSelection:
    target_index: SourceBindingTargetIndex
    bound_sources: tuple[RoleBoundSource, ...]

    @classmethod
    def from_source_binding(
        cls,
        source_binding: SourceBindingPlan,
        *,
        plan_selection: PlanSelectionSet,
        requested_facts: tuple[RequestedFact, ...],
    ) -> "SourceBindingRoleSelection | None":
        target_index = source_binding_target_index_for_plan_selection(
            plan_selection,
            requested_facts=requested_facts,
        )
        bound_sources: list[RoleBoundSource] = []
        for source in source_binding.bound_sources:
            if source.is_auxiliary_value:
                continue
            target = target_index.by_id.get(source.binding_target_id)
            if target is None:
                return None
            if source.source_candidate_id != target.source_candidate_id:
                return None
            bound_sources.append(RoleBoundSource(target=target, source=source))
        return cls(target_index=target_index, bound_sources=tuple(bound_sources))

    @property
    def sources_by_id(self) -> dict[str, BoundSource]:
        return {item.source.id: item.source for item in self.bound_sources}

    @property
    def source_ids_by_target(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, list[str]] = {}
        for item in self.bound_sources:
            output.setdefault(item.target.binding_target_id, []).append(item.source.id)
        return {
            target_id: tuple(dict.fromkeys(ids)) for target_id, ids in output.items()
        }

    def target_ids_for_fact(self, requested_fact_id: str) -> frozenset[str]:
        return frozenset(
            item.target.binding_target_id
            for item in self.bound_sources
            if item.target.requested_fact_id == requested_fact_id
        )

    def targets_for_plan_member(
        self,
        *,
        plan_selection_id: str,
        source_candidate_id: str,
    ) -> tuple[SourceBindingTarget, ...]:
        target_ids = tuple(
            dict.fromkeys(
                compatibility.binding_target_id
                for compatibility in self.target_index.compatibilities
                if compatibility.plan_selection_id == plan_selection_id
                and compatibility.source_candidate_id == source_candidate_id
            )
        )
        return tuple(self.target_index.require(target_id) for target_id in target_ids)


def plan_selection_uses_only_values(plan_selection: PlanSelectionSet) -> bool:
    return all(
        member.value_id
        and not member.read_id
        and not member.memory_relation_id
        and not member.source_relation_id
        for plan in plan_selection.plan_selections
        for member in plan.source_members
    )


def value_only_source_binding_plan(
    plan_selection: PlanSelectionSet,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> SourceBindingPlan | None:
    target_index = source_binding_target_index_for_plan_selection(
        plan_selection,
        requested_facts=requested_facts,
    )
    bound_sources: list[BoundSource] = []
    selected_plans: list[SelectedSourceStrategy] = []
    for requested_fact_id in _requested_fact_ids(plan_selection):
        fact_plans = tuple(
            plan
            for plan in plan_selection.plan_selections
            if plan.requested_fact_id == requested_fact_id
        )
        selected_plan = _single_compatible_plan(fact_plans)
        if selected_plan is None:
            return None
        selected_plans.append(selected_plan)
    for plan in selected_plans:
        for member in plan.source_members:
            for target in _targets_for_plan_member(
                target_index,
                plan_selection_id=plan.plan_selection_id,
                source_candidate_id=member.source_candidate_id,
            ):
                bound_sources.append(
                    BoundSource(
                        id=f"sb_{len(bound_sources) + 1}",
                        requested_fact_id=plan.requested_fact_id,
                        binding_target_id=target.binding_target_id,
                        requirement_id=target.requirement_id,
                        answer_population=AnswerPopulation(
                            population_binding_id=(
                                f"pop.{member.source_candidate_id}.value"
                            ),
                            intent_text="selected known value",
                            match_basis_explanation=(
                                "The selected source strategy uses an already-known "
                                "value as an operation operand."
                            ),
                        ),
                        value_id=member.value_id,
                        source_candidate_id=member.source_candidate_id,
                        evidence_items=(),
                    )
                )
    return SourceBindingPlan(bound_sources=tuple(bound_sources))


def bound_plan_selection_for_source_binding(
    plan_selection: PlanSelectionSet,
    source_binding: SourceBindingPlan,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> BoundPlanSelectionSet | None:
    role_selection = SourceBindingRoleSelection.from_source_binding(
        source_binding,
        plan_selection=plan_selection,
        requested_facts=requested_facts,
    )
    if role_selection is None:
        return None
    selected_plan_ids = _selected_plan_ids(
        plan_selection,
        role_selection=role_selection,
    )
    if not selected_plan_ids:
        return None
    bound_plans = tuple(
        _bound_plan(plan, role_selection=role_selection)
        for plan in plan_selection.plan_selections
        if plan.plan_selection_id in selected_plan_ids
    )
    if not bound_plans:
        return None
    return BoundPlanSelectionSet(plan_selections=bound_plans)


def _selected_plan_ids(
    plan_selection: PlanSelectionSet,
    *,
    role_selection: SourceBindingRoleSelection,
) -> frozenset[str]:
    selected: list[str] = []
    for requested_fact_id in _requested_fact_ids(plan_selection):
        matches = tuple(
            plan
            for plan in plan_selection.plan_selections
            if plan.requested_fact_id == requested_fact_id
            and _plan_matches(plan, role_selection=role_selection)
        )
        if not matches:
            return frozenset()
        selected.extend(plan.plan_selection_id for plan in matches)
    return frozenset(selected)


def _plan_matches(
    plan: SelectedSourceStrategy,
    *,
    role_selection: SourceBindingRoleSelection,
) -> bool:
    target_ids = role_selection.target_index.target_ids_by_plan().get(
        plan.plan_selection_id
    )
    if not target_ids:
        return False
    if target_ids != role_selection.target_ids_for_fact(plan.requested_fact_id):
        return False
    return all(
        _member_matches(
            plan_selection_id=plan.plan_selection_id,
            member=member,
            role_selection=role_selection,
        )
        for member in plan.source_members
    )


def _member_matches(
    *,
    plan_selection_id: str,
    member: SourceStrategyMember,
    role_selection: SourceBindingRoleSelection,
) -> bool:
    bound_target_ids = frozenset(role_selection.source_ids_by_target)
    member_targets = role_selection.targets_for_plan_member(
        plan_selection_id=plan_selection_id,
        source_candidate_id=member.source_candidate_id,
    )
    targets_match = all(
        target.binding_target_id in bound_target_ids
        for target in member_targets
    )
    if not targets_match:
        return False
    requires_answer_fulfillment = any(
        target.required_answer_output_ids for target in member_targets
    )
    if not requires_answer_fulfillment:
        return True
    if member.fulfillment_support_set_ids:
        selected_support_set_ids = _member_selected_support_set_ids(
            plan_selection_id=plan_selection_id,
            member=member,
            role_selection=role_selection,
        )
        return bool(selected_support_set_ids) and selected_support_set_ids <= set(
            member.fulfillment_support_set_ids
        )
    expected_evidence_ids = {
        item.evidence_id
        for item in member.operation_evidence
        if item.evidence_id
    }
    if not expected_evidence_ids:
        return True
    bound_evidence_ids = {
        evidence_id
        for target in role_selection.targets_for_plan_member(
            plan_selection_id=plan_selection_id,
            source_candidate_id=member.source_candidate_id,
        )
        for source_id in role_selection.source_ids_by_target.get(
            target.binding_target_id,
            (),
        )
        for source in (role_selection.sources_by_id[source_id],)
        for fulfillment in source.fulfillments
        for evidence_id in fulfillment.all_evidence_ids()
    }
    return expected_evidence_ids <= bound_evidence_ids


def _member_selected_support_set_ids(
    *,
    plan_selection_id: str,
    member: SourceStrategyMember,
    role_selection: SourceBindingRoleSelection,
) -> set[str]:
    selected: set[str] = set()
    targets = role_selection.targets_for_plan_member(
        plan_selection_id=plan_selection_id,
        source_candidate_id=member.source_candidate_id,
    )
    for target in targets:
        source_ids = role_selection.source_ids_by_target.get(
            target.binding_target_id,
            (),
        )
        for source_id in source_ids:
            source = role_selection.sources_by_id[source_id]
            selected.update(_bound_source_fulfillment_support_set_ids(source))
    return selected


def _single_compatible_plan(
    matches: tuple[SelectedSourceStrategy, ...],
) -> SelectedSourceStrategy | None:
    if len(matches) == 1:
        return matches[0]
    signatures = {_plan_binding_signature(plan) for plan in matches}
    if len(signatures) == 1 and matches:
        return matches[0]
    return None


def _plan_binding_signature(plan: SelectedSourceStrategy) -> tuple[object, ...]:
    return (
        plan.requested_fact_id,
        plan.plan_shape,
        plan.required_answer_output_ids,
        tuple(
            (
                member.source_candidate_id,
                source_binding_member_requirement_ids(member),
                tuple(sorted(set(member.field_ids))),
            )
            for member in plan.source_members
        ),
    )


def _bound_plan(
    plan: SelectedSourceStrategy,
    *,
    role_selection: SourceBindingRoleSelection,
) -> BoundSelectedSourceStrategy:
    return BoundSelectedSourceStrategy(
        plan_selection_id=plan.plan_selection_id,
        requested_fact_id=plan.requested_fact_id,
        source_strategy_id=plan.source_strategy_id,
        plan_shape=plan.plan_shape,
        required_answer_output_ids=plan.required_answer_output_ids,
        source_members=tuple(
            _bound_source_strategy_member(
                plan=plan,
                member=member,
                role_selection=role_selection,
            )
            for member in plan.source_members
        ),
    )


def _bound_source_strategy_member(
    *,
    plan: SelectedSourceStrategy,
    member: SourceStrategyMember,
    role_selection: SourceBindingRoleSelection,
) -> BoundSourceStrategyMember:
    role_targets = tuple(
        BoundRoleTarget(
            requirement_id=target.requirement_id,
            source_candidate_id=member.source_candidate_id,
            source_binding_ids=role_selection.source_ids_by_target[
                target.binding_target_id
            ],
            fulfillment_support_set_ids=_selected_support_set_ids(
                role_selection.source_ids_by_target[target.binding_target_id],
                role_selection=role_selection,
            ),
            answer_output_ids=target.answer_output_ids,
        )
        for target in role_selection.targets_for_plan_member(
            plan_selection_id=plan.plan_selection_id,
            source_candidate_id=member.source_candidate_id,
        )
        if role_selection.source_ids_by_target.get(target.binding_target_id)
    )
    source_binding_ids = tuple(
        source_id
        for role_target in role_targets
        for source_id in role_target.source_binding_ids
    )
    return BoundSourceStrategyMember(
        source_candidate_id=member.source_candidate_id,
        role_targets=role_targets,
        field_ids=_bound_plan_member_field_ids(
            plan_shape=plan.plan_shape,
            member=member,
            requested_fact_id=plan.requested_fact_id,
            required_answer_output_ids=plan.required_answer_output_ids,
            role_selection=role_selection,
            source_binding_ids=source_binding_ids,
        ),
    )


def _selected_support_set_ids(
    source_ids: tuple[str, ...],
    *,
    role_selection: SourceBindingRoleSelection,
) -> tuple[str, ...]:
    sources_by_id = role_selection.sources_by_id
    return tuple(
        dict.fromkeys(
            support_set_id
            for source_id in source_ids
            for bound in (sources_by_id.get(source_id),)
            if bound is not None
            for support_set_id in _bound_source_fulfillment_support_set_ids(bound)
        )
    )


def _bound_source_fulfillment_support_set_ids(bound: BoundSource) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            fulfillment.fulfillment_support_set_id
            for fulfillment in bound.fulfillments
            if fulfillment.fulfillment_support_set_id
        )
    )


def _bound_plan_member_field_ids(
    *,
    plan_shape: str,
    member: SourceStrategyMember,
    requested_fact_id: str,
    required_answer_output_ids: tuple[str, ...],
    role_selection: SourceBindingRoleSelection,
    source_binding_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if plan_shape != "list_rows" or len(required_answer_output_ids) <= 1:
        return member.field_ids
    bound_field_ids = _list_rows_bound_fulfillment_field_ids(
        requested_fact_id=requested_fact_id,
        required_answer_output_ids=required_answer_output_ids,
        role_selection=role_selection,
        source_binding_ids=source_binding_ids,
    )
    if bound_field_ids:
        return bound_field_ids
    return member.field_ids


def _list_rows_bound_fulfillment_field_ids(
    *,
    requested_fact_id: str,
    required_answer_output_ids: tuple[str, ...],
    role_selection: SourceBindingRoleSelection,
    source_binding_ids: tuple[str, ...],
) -> tuple[str, ...]:
    sources_by_id = role_selection.sources_by_id
    selected: list[str] = []
    for answer_output_id in required_answer_output_ids:
        for source_binding_id in source_binding_ids:
            bound = sources_by_id.get(source_binding_id)
            if bound is None:
                continue
            field_id_by_evidence_id = {
                item.evidence_id: item.field_id for item in bound.evidence_items
            }
            for fulfillment in bound.fulfillments:
                if fulfillment.requested_fact_id != requested_fact_id:
                    continue
                if fulfillment.answer_output_id != answer_output_id:
                    continue
                for evidence_id in (
                    *fulfillment.metric_measure_evidence_ids,
                    *fulfillment.value_evidence_ids,
                    *(
                        tuple(
                            component.field_evidence_id
                            for component in fulfillment.entity_evidence.components
                        )
                        if fulfillment.entity_evidence is not None
                        else ()
                    ),
                ):
                    field_id = field_id_by_evidence_id.get(evidence_id, "")
                    if field_id and field_id not in selected:
                        selected.append(field_id)
    return tuple(selected)


def _targets_for_plan_member(
    target_index: SourceBindingTargetIndex,
    *,
    plan_selection_id: str,
    source_candidate_id: str,
) -> tuple[SourceBindingTarget, ...]:
    target_ids = tuple(
        dict.fromkeys(
            compatibility.binding_target_id
            for compatibility in target_index.compatibilities
            if compatibility.plan_selection_id == plan_selection_id
            and compatibility.source_candidate_id == source_candidate_id
        )
    )
    return tuple(target_index.require(target_id) for target_id in target_ids)


def _requested_fact_ids(plan_selection: PlanSelectionSet) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(plan.requested_fact_id for plan in plan_selection.plan_selections)
    )
