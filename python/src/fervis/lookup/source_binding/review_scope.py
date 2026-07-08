"""Membership-test review scope for source binding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fervis.lookup.fact_plan.relations import EndpointParamBinding
from fervis.lookup.fact_plan.values import FactValue, known_input_id_for_value
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    RequestedFact,
    RequestedFactAnswerPopulationMembershipTest,
)
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.membership_tests import membership_test_key
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
    SourceBindingTargetIndex,
    source_binding_target_index,
)
from fervis.lookup.source_binding.review_surface import (
    SourceBindingReviewAxisKind,
    source_binding_review_surface,
)


class ReviewScopeDecisionKind(StrEnum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class ReviewScopeDecision:
    membership_test_id: str
    decision: ReviewScopeDecisionKind
    axis_kind: SourceBindingReviewAxisKind
    axis_id: str
    is_normal_instance: bool = False
    owner_surface_id: str = ""
    owner_surface_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewAxisScope:
    binding_target_id: str
    axis_kind: SourceBindingReviewAxisKind
    axis_id: str
    test_scope_decisions: tuple[ReviewScopeDecision, ...]

    @property
    def in_scope_test_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.membership_test_id
            for decision in self.test_scope_decisions
            if decision.decision == ReviewScopeDecisionKind.IN_SCOPE
        )

    @property
    def normal_instance_test_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.membership_test_id
            for decision in self.test_scope_decisions
            if decision.decision == ReviewScopeDecisionKind.IN_SCOPE
            and decision.is_normal_instance
        )

    @property
    def out_of_scope_proof_refs(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                ref
                for decision in self.test_scope_decisions
                if decision.decision == ReviewScopeDecisionKind.OUT_OF_SCOPE
                for ref in decision.proof_refs
            )
        )

    @property
    def out_of_scope_decisions(self) -> tuple[ReviewScopeDecision, ...]:
        return tuple(
            decision
            for decision in self.test_scope_decisions
            if decision.decision == ReviewScopeDecisionKind.OUT_OF_SCOPE
        )


@dataclass(frozen=True)
class SourceBindingReviewScope:
    axes: dict[tuple[str, SourceBindingReviewAxisKind, str], ReviewAxisScope]

    def finite_choice_param_test_ids(
        self,
        binding_target_id: str,
        param_id: str,
    ) -> tuple[str, ...]:
        return self._axis_scope(
            binding_target_id,
            SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
            param_id,
        ).in_scope_test_ids

    def finite_choice_param_normal_instance_test_ids(
        self,
        binding_target_id: str,
        param_id: str,
    ) -> tuple[str, ...]:
        return self._axis_scope(
            binding_target_id,
            SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
            param_id,
        ).normal_instance_test_ids

    def finite_choice_param_out_of_scope_proof_refs(
        self,
        binding_target_id: str,
        param_id: str,
    ) -> tuple[str, ...]:
        return self._axis_scope(
            binding_target_id,
            SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
            param_id,
        ).out_of_scope_proof_refs

    def finite_choice_param_out_of_scope_decisions(
        self,
        binding_target_id: str,
        param_id: str,
    ) -> tuple[ReviewScopeDecision, ...]:
        return self._axis_scope(
            binding_target_id,
            SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
            param_id,
        ).out_of_scope_decisions

    def row_predicate_test_ids(
        self,
        binding_target_id: str,
        predicate_id: str,
    ) -> tuple[str, ...]:
        return self._axis_scope(
            binding_target_id,
            SourceBindingReviewAxisKind.ROW_PREDICATE,
            predicate_id,
        ).in_scope_test_ids

    def row_predicate_out_of_scope_proof_refs(
        self,
        binding_target_id: str,
        predicate_id: str,
    ) -> tuple[str, ...]:
        return self._axis_scope(
            binding_target_id,
            SourceBindingReviewAxisKind.ROW_PREDICATE,
            predicate_id,
        ).out_of_scope_proof_refs

    def row_predicate_out_of_scope_decisions(
        self,
        binding_target_id: str,
        predicate_id: str,
    ) -> tuple[ReviewScopeDecision, ...]:
        return self._axis_scope(
            binding_target_id,
            SourceBindingReviewAxisKind.ROW_PREDICATE,
            predicate_id,
        ).out_of_scope_decisions

    def axis_scope(
        self,
        binding_target_id: str,
        axis_kind: SourceBindingReviewAxisKind,
        axis_id: str,
    ) -> ReviewAxisScope:
        return self._axis_scope(binding_target_id, axis_kind, axis_id)

    def _axis_scope(
        self,
        binding_target_id: str,
        axis_kind: SourceBindingReviewAxisKind,
        axis_id: str,
    ) -> ReviewAxisScope:
        return self.axes.get(
            (binding_target_id, axis_kind, axis_id),
            ReviewAxisScope(
                binding_target_id=binding_target_id,
                axis_kind=axis_kind,
                axis_id=axis_id,
                test_scope_decisions=(),
            ),
        )


def source_binding_review_scope(
    request: SourceBindingRequest,
    *,
    candidates_by_id: dict[str, SourceCandidate],
    target_index: SourceBindingTargetIndex | None = None,
) -> SourceBindingReviewScope:
    targets = (target_index or source_binding_target_index(request)).targets
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    values_by_input_ref = _applicable_values_by_input_ref(request.available_values)
    axes: dict[tuple[str, SourceBindingReviewAxisKind, str], ReviewAxisScope] = {}
    for target in targets:
        fact = facts_by_id.get(target.requested_fact_id)
        candidate = candidates_by_id.get(target.source_candidate_id)
        for scoped in _target_axis_scopes(
            target,
            fact=fact,
            candidate=candidate,
            values_by_input_ref=values_by_input_ref,
        ):
            axes[(target.binding_target_id, scoped.axis_kind, scoped.axis_id)] = scoped
    return SourceBindingReviewScope(axes=axes)


def _target_axis_scopes(
    target: SourceBindingTarget,
    *,
    fact: RequestedFact | None,
    candidate: SourceCandidate | None,
    values_by_input_ref: dict[str, tuple[FactValue, ...]],
) -> tuple[ReviewAxisScope, ...]:
    if fact is None or fact.answer_population is None or candidate is None:
        return ()
    surface = source_binding_review_surface(candidate)
    axis_owners = _axis_owners(surface)
    input_owner_surfaces = _input_owner_surfaces(
        candidate,
        requested_fact_id=target.requested_fact_id,
        values_by_input_ref=values_by_input_ref,
    )
    output: list[ReviewAxisScope] = []
    for axis_id in surface.finite_choice_params:
        output.append(
            _axis_scope(
                target,
                fact=fact,
                values_by_input_ref=values_by_input_ref,
                axis_kind=SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
                axis_id=axis_id,
                axis_owners=axis_owners,
                input_owner_surfaces=input_owner_surfaces,
            )
        )
    for axis_id in surface.row_predicates:
        output.append(
            _axis_scope(
                target,
                fact=fact,
                values_by_input_ref=values_by_input_ref,
                axis_kind=SourceBindingReviewAxisKind.ROW_PREDICATE,
                axis_id=axis_id,
                axis_owners=axis_owners,
                input_owner_surfaces=input_owner_surfaces,
            )
        )
    return tuple(output)


def _axis_scope(
    target: SourceBindingTarget,
    *,
    fact: RequestedFact,
    values_by_input_ref: dict[str, tuple[FactValue, ...]],
    axis_kind: SourceBindingReviewAxisKind,
    axis_id: str,
    axis_owners: dict[str, tuple[SourceBindingReviewAxisKind, str]],
    input_owner_surfaces: dict[str, tuple[_OwnerSurfaceProof, ...]],
) -> ReviewAxisScope:
    decisions: list[ReviewScopeDecision] = []
    for test in fact.answer_population.membership_tests:
        test_id = membership_test_key(test)
        is_normal_instance = (
            test.kind == AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
        )
        owner_surface_proofs = _owner_surface_proofs_for_test(
            test,
            input_owner_surfaces=input_owner_surfaces,
        )
        if owner_surface_proofs:
            decisions.append(
                _out_of_scope_decision(
                    test_id,
                    axis_kind=axis_kind,
                    axis_id=axis_id,
                    is_normal_instance=is_normal_instance,
                    owner_surface_ids=_owner_surface_ids(owner_surface_proofs),
                    proof_refs=_test_owner_proof_refs(
                        test,
                        requested_fact_id=target.requested_fact_id,
                        owner_proofs=owner_surface_proofs,
                        values_by_input_ref=values_by_input_ref,
                    ),
                )
            )
            continue
        owner_axis = axis_owners.get(test_id)
        if owner_axis is None or owner_axis == (axis_kind, axis_id):
            decisions.append(
                _in_scope_decision(
                    test_id,
                    axis_kind=axis_kind,
                    axis_id=axis_id,
                    is_normal_instance=is_normal_instance,
                )
            )
            continue
        decisions.append(
            _out_of_scope_decision(
                test_id,
                axis_kind=axis_kind,
                axis_id=axis_id,
                is_normal_instance=is_normal_instance,
                owner_surface_ids=(_axis_owner_surface_id(owner_axis),),
                proof_refs=(
                    f"membership_test:{test_id}",
                    f"review_axis:{owner_axis[0].value}:{owner_axis[1]}",
                ),
            )
        )
    return ReviewAxisScope(
        binding_target_id=target.binding_target_id,
        axis_kind=axis_kind,
        axis_id=axis_id,
        test_scope_decisions=tuple(decisions),
    )


def _axis_owners(
    surface: object,
) -> dict[str, tuple[SourceBindingReviewAxisKind, str]]:
    output: dict[str, tuple[SourceBindingReviewAxisKind, str]] = {}
    for axis in surface.finite_choice_params.values():
        for test_id in axis.owned_membership_test_ids:
            output[test_id] = (
                SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
                axis.axis_id,
            )
    for axis in surface.row_predicates.values():
        for test_id in axis.owned_membership_test_ids:
            output[test_id] = (
                SourceBindingReviewAxisKind.ROW_PREDICATE,
                axis.axis_id,
            )
    return output


@dataclass(frozen=True)
class _OwnerSurfaceProof:
    owner_surface_id: str
    proof_refs: tuple[str, ...]


def _owner_surface_proofs_for_test(
    test: RequestedFactAnswerPopulationMembershipTest,
    *,
    input_owner_surfaces: dict[str, tuple[_OwnerSurfaceProof, ...]],
) -> tuple[_OwnerSurfaceProof, ...]:
    if not test.owned_question_input_refs:
        return ()
    owner_proofs: list[_OwnerSurfaceProof] = []
    for input_ref in test.owned_question_input_refs:
        proofs = input_owner_surfaces.get(input_ref, ())
        if not proofs:
            return ()
        owner_proofs.extend(proofs)
    return tuple(owner_proofs)


def _input_owner_surfaces(
    candidate: SourceCandidate,
    *,
    requested_fact_id: str,
    values_by_input_ref: dict[str, tuple[FactValue, ...]],
) -> dict[str, tuple[_OwnerSurfaceProof, ...]]:
    output: dict[str, tuple[_OwnerSurfaceProof, ...]] = {}
    for input_ref, values in values_by_input_ref.items():
        applicable_values = tuple(
            value for value in values if _value_applies_to_fact(value, requested_fact_id)
        )
        if not applicable_values:
            continue
        proofs = _candidate_owner_surface_proofs(
            candidate,
            input_ref=input_ref,
            values=applicable_values,
        )
        if proofs:
            output[input_ref] = proofs
    return output


def _test_owner_proof_refs(
    test: RequestedFactAnswerPopulationMembershipTest,
    *,
    requested_fact_id: str,
    owner_proofs: tuple[_OwnerSurfaceProof, ...],
    values_by_input_ref: dict[str, tuple[FactValue, ...]],
) -> tuple[str, ...]:
    refs: list[str] = [f"membership_test:{membership_test_key(test)}"]
    for input_ref in test.owned_question_input_refs:
        refs.append(f"known_input:{input_ref}")
    for proof in owner_proofs:
        refs.append(proof.owner_surface_id)
        refs.extend(proof.proof_refs)
    for input_ref in test.owned_question_input_refs:
        refs.extend(
            ref
            for value in values_by_input_ref.get(input_ref, ())
            if _value_applies_to_fact(value, requested_fact_id)
            for ref in value.proof_refs
        )
    return tuple(dict.fromkeys(refs))


def _in_scope_decision(
    membership_test_id: str,
    *,
    axis_kind: SourceBindingReviewAxisKind,
    axis_id: str,
    is_normal_instance: bool,
) -> ReviewScopeDecision:
    return ReviewScopeDecision(
        membership_test_id=membership_test_id,
        decision=ReviewScopeDecisionKind.IN_SCOPE,
        axis_kind=axis_kind,
        axis_id=axis_id,
        is_normal_instance=is_normal_instance,
    )


def _out_of_scope_decision(
    membership_test_id: str,
    *,
    axis_kind: SourceBindingReviewAxisKind,
    axis_id: str,
    is_normal_instance: bool,
    owner_surface_ids: tuple[str, ...],
    proof_refs: tuple[str, ...],
) -> ReviewScopeDecision:
    normalized_owner_surface_ids = tuple(dict.fromkeys(owner_surface_ids))
    return ReviewScopeDecision(
        membership_test_id=membership_test_id,
        decision=ReviewScopeDecisionKind.OUT_OF_SCOPE,
        axis_kind=axis_kind,
        axis_id=axis_id,
        is_normal_instance=is_normal_instance,
        owner_surface_id=(
            normalized_owner_surface_ids[0] if len(normalized_owner_surface_ids) == 1 else ""
        ),
        owner_surface_ids=normalized_owner_surface_ids,
        proof_refs=proof_refs,
    )


def _axis_owner_surface_id(
    owner_axis: tuple[SourceBindingReviewAxisKind, str],
) -> str:
    return f"review_axis:{owner_axis[0].value}:{owner_axis[1]}"


def _owner_surface_ids(
    proofs: tuple[_OwnerSurfaceProof, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(proof.owner_surface_id for proof in proofs))


def _candidate_owner_surface_proofs(
    candidate: SourceCandidate,
    *,
    input_ref: str,
    values: tuple[FactValue, ...],
) -> tuple[_OwnerSurfaceProof, ...]:
    known_ref = f"known_input:{input_ref}"
    proofs = list(_candidate_applied_param_binding_proofs(candidate, known_ref))
    proofs.extend(
        _OwnerSurfaceProof(
            owner_surface_id=f"applied_filter:{field_id}:{input_ref}",
            proof_refs=(known_ref,),
        )
        for item in candidate.applied_filters
        if item.known_input_id == input_ref
        for field_id in item.predicate_field_ids
    )
    proofs.extend(_candidate_selectable_param_decision_proofs(candidate, values))
    return _dedupe_owner_surface_proofs(tuple(proofs))


def _candidate_selectable_param_decision_proofs(
    candidate: SourceCandidate,
    values: tuple[FactValue, ...],
) -> tuple[_OwnerSurfaceProof, ...]:
    value_ids = frozenset(value.id for value in values)
    if not value_ids:
        return ()
    return tuple(
        _OwnerSurfaceProof(
            owner_surface_id=f"source_param:{param_id}",
            proof_refs=(f"param_decision:{decision_id}",),
        )
        for param in candidate.params
        if isinstance(param, dict)
        for param_id in (str(param.get("param_id") or ""),)
        if param_id
        for binding_key in _available_binding_value_keys(param, value_ids=value_ids)
        for decision_id in _matching_bind_decision_ids(param, binding_key=binding_key)
    )


def _available_binding_value_keys(
    param: dict[str, object],
    *,
    value_ids: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (value_id, str(item.get("value_component") or ""))
        for item in param.get("binding_values") or ()
        if isinstance(item, dict)
        for value_id in (str(item.get("value") or ""),)
        if value_id in value_ids and str(item.get("source") or "") == "available_value"
    )


def _matching_bind_decision_ids(
    param: dict[str, object],
    *,
    binding_key: tuple[str, str],
) -> tuple[str, ...]:
    value_id, value_component = binding_key
    return tuple(
        decision_id
        for option in param.get("decision_options") or ()
        if isinstance(option, dict)
        and str(option.get("decision") or "") == "bind"
        and str(option.get("value") or "") == value_id
        and str(option.get("value_component") or "") == value_component
        for decision_id in (str(option.get("param_decision_id") or ""),)
        if decision_id
    )


def _dedupe_owner_surface_proofs(
    proofs: tuple[_OwnerSurfaceProof, ...],
) -> tuple[_OwnerSurfaceProof, ...]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    output: list[_OwnerSurfaceProof] = []
    for proof in proofs:
        key = (proof.owner_surface_id, proof.proof_refs)
        if key in seen:
            continue
        seen.add(key)
        output.append(proof)
    return tuple(output)


def _candidate_applied_param_binding_proofs(
    candidate: SourceCandidate,
    known_ref: str,
) -> tuple[_OwnerSurfaceProof, ...]:
    binding_sets = candidate.applied_param_binding_sets
    if binding_sets:
        refs_by_set = tuple(
            _matching_binding_proofs(binding_set, known_ref)
            for binding_set in binding_sets
        )
        if all(refs_by_set):
            return tuple(proof for proofs in refs_by_set for proof in proofs)
        return ()
    return _matching_binding_proofs(candidate.applied_param_bindings, known_ref)


def _matching_binding_proofs(
    bindings: tuple[EndpointParamBinding, ...],
    known_ref: str,
) -> tuple[_OwnerSurfaceProof, ...]:
    return tuple(
        _OwnerSurfaceProof(
            owner_surface_id=f"source_param:{binding.param_id}",
            proof_refs=binding.proof_refs,
        )
        for binding in bindings
        if known_ref in binding.proof_refs
    )


def _applicable_values_by_input_ref(
    values: tuple[FactValue, ...],
) -> dict[str, tuple[FactValue, ...]]:
    grouped: dict[str, list[FactValue]] = {}
    for value in values:
        input_ref = known_input_id_for_value(value)
        if input_ref:
            grouped.setdefault(input_ref, []).append(value)
    return {input_ref: tuple(items) for input_ref, items in grouped.items()}


def _value_applies_to_fact(value: FactValue, requested_fact_id: str) -> bool:
    return (
        not value.applies_to_requested_fact_ids
        or requested_fact_id in value.applies_to_requested_fact_ids
    )
