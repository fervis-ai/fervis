"""Membership-test review scope for source binding."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
)
from fervis.lookup.answer_program.values import FactValue, known_input_id_for_value
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    RequestedFact,
    RequestedFactAnswerPopulationMembershipTest,
)
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.closed_key_params import (
    closed_key_param_binding_index,
)
from fervis.lookup.source_binding.input_applications import (
    ResolvedInputApplicationSurface,
    resolved_input_application_surfaces,
)
from fervis.lookup.source_binding.membership_tests import membership_test_key
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
    SourceBindingTargetIndex,
    source_binding_target_index,
)
from fervis.lookup.source_binding.review_surface import (
    SourceBindingReviewSurface,
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
    def out_of_scope_decisions(self) -> tuple[ReviewScopeDecision, ...]:
        return tuple(
            decision
            for decision in self.test_scope_decisions
            if decision.decision == ReviewScopeDecisionKind.OUT_OF_SCOPE
        )


@dataclass(frozen=True)
class SourceBindingReviewScope:
    axes: dict[tuple[str, SourceBindingReviewAxisKind, str], ReviewAxisScope]
    input_owned_test_ids_by_binding_target_id: dict[str, tuple[str, ...]]
    population_binding_test_ids_by_binding_target_id: dict[str, tuple[str, ...]]

    def input_owned_test_ids(self, binding_target_id: str) -> tuple[str, ...]:
        return self.input_owned_test_ids_by_binding_target_id.get(
            binding_target_id,
            (),
        )

    def population_binding_test_ids(
        self,
        binding_target_id: str,
    ) -> tuple[str, ...]:
        return self.population_binding_test_ids_by_binding_target_id.get(
            binding_target_id,
            (),
        )

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
    backend_bindings = closed_key_param_binding_index(
        request,
        targets=targets,
        candidates_by_id=candidates_by_id,
    )
    input_application_surfaces = resolved_input_application_surfaces(
        request,
        targets=targets,
        candidates_by_id=candidates_by_id,
        closed_key_bindings=backend_bindings,
    )
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    values_by_input_ref = _applicable_values_by_input_ref(request.available_values)
    axes: dict[tuple[str, SourceBindingReviewAxisKind, str], ReviewAxisScope] = {}
    input_owned_test_ids_by_target: dict[str, tuple[str, ...]] = {}
    population_binding_test_ids_by_target: dict[str, tuple[str, ...]] = {}
    for target in targets:
        fact = facts_by_id.get(target.requested_fact_id)
        candidate = candidates_by_id.get(target.source_candidate_id)
        input_owner_surfaces = _target_input_owner_surfaces(
            target,
            fact=fact,
            candidate=candidate,
            backend_param_bindings=tuple(
                binding
                for binding_set in backend_bindings.backend_param_binding_sets(
                    target.binding_target_id
                )
                for binding in binding_set
            ),
            input_application_surface=input_application_surfaces.get(
                target.binding_target_id
            ),
            values_by_input_ref=values_by_input_ref,
        )
        input_owned_test_ids = _input_owned_membership_test_ids(
            fact,
            input_owner_surfaces=input_owner_surfaces,
        )
        input_owned_test_ids_by_target[target.binding_target_id] = input_owned_test_ids
        target_axis_scopes = _target_axis_scopes(
            target,
            fact=fact,
            candidate=candidate,
            input_owner_surfaces=input_owner_surfaces,
            values_by_input_ref=values_by_input_ref,
        )
        for scoped in target_axis_scopes:
            axes[(target.binding_target_id, scoped.axis_kind, scoped.axis_id)] = scoped
        population_binding_test_ids_by_target[target.binding_target_id] = (
            _population_binding_membership_test_ids(
                fact,
                input_owned_test_ids=input_owned_test_ids,
                axis_scopes=target_axis_scopes,
            )
        )
    return SourceBindingReviewScope(
        axes=axes,
        input_owned_test_ids_by_binding_target_id=(
            input_owned_test_ids_by_target
        ),
        population_binding_test_ids_by_binding_target_id=(
            population_binding_test_ids_by_target
        ),
    )


def _population_binding_membership_test_ids(
    fact: RequestedFact | None,
    *,
    input_owned_test_ids: tuple[str, ...],
    axis_scopes: tuple[ReviewAxisScope, ...],
) -> tuple[str, ...]:
    population = fact.answer_population if fact is not None else None
    if population is None:
        return ()
    axis_owned_test_ids = {
        test_id
        for axis_scope in axis_scopes
        for test_id in axis_scope.in_scope_test_ids
    }
    tests_owned_elsewhere = set(input_owned_test_ids) | axis_owned_test_ids
    return tuple(
        membership_test_key(test)
        for test in population.membership_tests
        if test.kind is not AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY
        and membership_test_key(test) not in tests_owned_elsewhere
    )


def _target_input_owner_surfaces(
    target: SourceBindingTarget,
    *,
    fact: RequestedFact | None,
    candidate: SourceCandidate | None,
    backend_param_bindings: tuple[DraftEndpointParamBinding, ...],
    input_application_surface: ResolvedInputApplicationSurface | None,
    values_by_input_ref: dict[str, tuple[FactValue, ...]],
) -> dict[str, tuple[_OwnerSurfaceProof, ...]]:
    if fact is None or candidate is None:
        return {}
    return _input_owner_surfaces(
        candidate,
        requested_fact_id=target.requested_fact_id,
        backend_param_bindings=backend_param_bindings,
        input_application_surface=input_application_surface,
        values_by_input_ref=values_by_input_ref,
    )


def _input_owned_membership_test_ids(
    fact: RequestedFact | None,
    *,
    input_owner_surfaces: dict[str, tuple[_OwnerSurfaceProof, ...]],
) -> tuple[str, ...]:
    population = fact.answer_population if fact is not None else None
    if population is None:
        return ()
    return tuple(
        membership_test_key(test)
        for test in population.membership_tests
        if test.kind == AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT
        and test.owned_question_input_refs
        and all(
            input_owner_surfaces.get(input_ref)
            for input_ref in test.owned_question_input_refs
        )
    )


def _target_axis_scopes(
    target: SourceBindingTarget,
    *,
    fact: RequestedFact | None,
    candidate: SourceCandidate | None,
    input_owner_surfaces: dict[str, tuple[_OwnerSurfaceProof, ...]],
    values_by_input_ref: dict[str, tuple[FactValue, ...]],
) -> tuple[ReviewAxisScope, ...]:
    if fact is None or fact.answer_population is None or candidate is None:
        return ()
    surface = source_binding_review_surface(candidate)
    axis_owners = _axis_owners(surface)
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
    answer_population = fact.answer_population
    if answer_population is None:
        raise ValueError("review axis requires answer population")
    for test in answer_population.membership_tests:
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
    surface: SourceBindingReviewSurface,
) -> dict[str, tuple[SourceBindingReviewAxisKind, str]]:
    output: dict[str, tuple[SourceBindingReviewAxisKind, str]] = {}
    for finite_axis in surface.finite_choice_params.values():
        for test_id in finite_axis.owned_membership_test_ids:
            output[test_id] = (
                SourceBindingReviewAxisKind.FINITE_CHOICE_PARAM,
                finite_axis.axis_id,
            )
    for predicate_axis in surface.row_predicates.values():
        for test_id in predicate_axis.owned_membership_test_ids:
            output[test_id] = (
                SourceBindingReviewAxisKind.ROW_PREDICATE,
                predicate_axis.axis_id,
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
    backend_param_bindings: tuple[DraftEndpointParamBinding, ...],
    input_application_surface: ResolvedInputApplicationSurface | None,
    values_by_input_ref: dict[str, tuple[FactValue, ...]],
) -> dict[str, tuple[_OwnerSurfaceProof, ...]]:
    output: dict[str, tuple[_OwnerSurfaceProof, ...]] = {}
    for input_ref, values in values_by_input_ref.items():
        applicable_values = tuple(
            value
            for value in values
            if _value_applies_to_fact(value, requested_fact_id)
        )
        if not applicable_values:
            continue
        proofs = _candidate_owner_surface_proofs(
            candidate,
            input_ref=input_ref,
            backend_param_bindings=backend_param_bindings,
            input_application_surface=input_application_surface,
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
            normalized_owner_surface_ids[0]
            if len(normalized_owner_surface_ids) == 1
            else ""
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
    backend_param_bindings: tuple[DraftEndpointParamBinding, ...],
    input_application_surface: ResolvedInputApplicationSurface | None,
    values: tuple[FactValue, ...],
) -> tuple[_OwnerSurfaceProof, ...]:
    known_ref = f"known_input:{input_ref}"
    proofs = list(_candidate_applied_param_binding_proofs(candidate, known_ref))
    proofs.extend(_matching_binding_proofs(backend_param_bindings, known_ref))
    proofs.extend(
        _OwnerSurfaceProof(
            owner_surface_id=f"applied_filter:{field_id}:{input_ref}",
            proof_refs=(known_ref,),
        )
        for item in candidate.applied_filters
        if item.known_input_id == input_ref
        for field_id in item.predicate_field_ids
    )
    if input_application_surface is not None:
        proofs.extend(
            _OwnerSurfaceProof(
                owner_surface_id=owner.owner_surface_id,
                proof_refs=owner.proof_refs,
            )
            for owner in input_application_surface.owners()
            if owner.known_input_id == input_ref
        )
    return _dedupe_owner_surface_proofs(tuple(proofs))


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
    bindings: tuple[DraftEndpointParamBinding, ...],
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
