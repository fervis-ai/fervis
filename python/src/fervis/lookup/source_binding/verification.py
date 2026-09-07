"""Mechanical verification of one semantic source strategy."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.relation_catalog.model import requires_caller_supplied_input

from fervis.lookup.qualification import (
    BooleanAtomRef,
    BooleanRequirementUseSite,
    QualificationAtomProof,
    QualificationGuarantee,
    SubjectGuarantee,
)
from fervis.lookup.question_contract import (
    FactLocalKind,
    FactLocalRef,
    FactTerm,
    SubjectRows,
)
from fervis.lookup.question_contract.analysis import infer_expression_types
from fervis.lookup.relation_catalog.row_sources import (
    row_source_type_supports_semantic_type,
    semantic_type_for_row_source_type,
)
from fervis.lookup.semantic_types import IdentifierType
from fervis.lookup.source_binding.model import (
    AssociationRealizationKind,
    SemanticSourceBindingRequest,
    SourceBindingPlan,
    SourceMechanic,
    SourceMechanicKind,
)
from fervis.types.enums import StrEnum


class SourceStrategyVerificationFailureReason(StrEnum):
    INVALID_BINDING = "INVALID_BINDING"
    INCOMPATIBLE_GRAIN = "INCOMPATIBLE_GRAIN"
    INCOMPLETE_SUBJECT_IDENTITY = "INCOMPLETE_SUBJECT_IDENTITY"
    INCONSISTENT_IDENTIFIER = "INCONSISTENT_IDENTIFIER"
    INCOMPLETE_ASSOCIATION = "INCOMPLETE_ASSOCIATION"
    INSUFFICIENT_COMPLETENESS = "INSUFFICIENT_COMPLETENESS"
    INCOMPLETE_INVOCATION = "INCOMPLETE_INVOCATION"
    INCOMPLETE_QUALIFICATION = "INCOMPLETE_QUALIFICATION"
    INCOMPLETE_OUTPUT_DEPENDENCY = "INCOMPLETE_OUTPUT_DEPENDENCY"
    INVALID_AUTHORITY = "INVALID_AUTHORITY"


@dataclass(frozen=True)
class VerifiedSourceStrategy:
    request: SemanticSourceBindingRequest
    binding_plan: SourceBindingPlan
    qualification_guarantee: QualificationGuarantee
    subject_guarantee: SubjectGuarantee
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class SourceStrategyVerificationFailure:
    reason: SourceStrategyVerificationFailureReason
    failed_requirement_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]


SourceStrategyVerificationOutcome = (
    VerifiedSourceStrategy | SourceStrategyVerificationFailure
)


def verify_source_strategy(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> SourceStrategyVerificationOutcome:
    try:
        from fervis.lookup.source_binding.membership import validate_membership
        for realizations in plan.set_bindings.values():
            for realization in realizations:
                if realization.membership is not None:
                    validate_membership(realization.membership,
                        source=request.source_catalog.source(realization.source_ref), request=request)
        request = request.for_bindings(
            plan.set_bindings,
            plan.fact_bindings,
            plan.association_bindings,
        )
    except ValueError:
        return _failure(SourceStrategyVerificationFailureReason.INVALID_BINDING)
    if plan.strategy != request.strategy:
        return _failure(SourceStrategyVerificationFailureReason.INVALID_BINDING)
    output_failure = _verify_output_dependencies(plan, request=request)
    if output_failure is not None:
        return output_failure
    expression_failure = _verify_concrete_expression_types(plan, request=request)
    if expression_failure is not None:
        return expression_failure
    identity_failure = _verify_set_identities(plan, request=request)
    if identity_failure is not None:
        return identity_failure
    association_failure = _verify_associations(plan, request=request)
    if association_failure is not None:
        return association_failure
    invocation_failure = _verify_invocation_completeness(plan, request=request)
    if invocation_failure is not None:
        return invocation_failure
    qualification = _qualification_guarantee(plan, request=request)
    if isinstance(qualification, SourceStrategyVerificationFailure):
        return qualification
    subject = _subject_guarantee(plan, request=request)
    if isinstance(subject, SourceStrategyVerificationFailure):
        return subject
    evidence = tuple(
        dict.fromkeys(
            (
                request.source_catalog.contract_snapshot.ref,
                *(
                    ref
                    for proof in qualification.atom_proofs
                    for ref in proof.proof_refs
                ),
                *subject.proof_refs,
            )
        )
    )
    return VerifiedSourceStrategy(
        request=request,
        binding_plan=plan,
        qualification_guarantee=qualification,
        subject_guarantee=subject,
        evidence_refs=evidence,
    )


def _verify_set_identities(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> SourceStrategyVerificationFailure | None:
    failed: list[str] = []
    if isinstance(request.index.result_grain, SubjectRows):
        subject_ref = request.index.subject_obligation.subject_set_ref.token
        for realization in plan.set_bindings.get(subject_ref, ()):
            if realization.identity_ref is None or not realization.identity_field_refs:
                failed.append(subject_ref)
        if failed:
            return _failure(
                SourceStrategyVerificationFailureReason.INCOMPLETE_SUBJECT_IDENTITY,
                *failed,
            )
    for fact_ref, realizations in plan.fact_bindings.items():
        fact = request.index.term_by_ref[FactLocalRef.from_token(fact_ref)]
        if not isinstance(fact, FactTerm) or not isinstance(
            fact.value_type, IdentifierType
        ):
            continue
        identified_set_ref = request.index.fact_local_ref_by_local_id[
            fact.value_type.set_ref
        ].token
        set_realizations = plan.set_bindings.get(identified_set_ref, ())
        for fact_realization in realizations:
            fact_contract = _identity_contract(
                fact_realization.identity_ref, request=request
            )
            set_contracts = {
                contract
                for item in set_realizations
                if item.branch_id == fact_realization.branch_id
                for contract in (
                    _identity_contract(item.identity_ref, request=request),
                )
                if contract is not None
            }
            if fact_contract is None or fact_contract not in set_contracts:
                failed.extend((fact_ref, identified_set_ref))
    if failed:
        return _failure(
            SourceStrategyVerificationFailureReason.INCONSISTENT_IDENTIFIER,
            *failed,
        )
    return None


def _verify_concrete_expression_types(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> SourceStrategyVerificationFailure | None:
    for branch in request.strategy.branches:
        concrete_types = {}
        for ref, term in request.index.term_by_ref.items():
            if not isinstance(term, FactTerm) or isinstance(
                term.value_type, IdentifierType
            ):
                continue
            realizations = tuple(
                item
                for item in plan.fact_bindings.get(ref.token, ())
                if item.branch_id == branch.branch_id
            )
            if not realizations:
                continue
            if len(realizations) != 1 or len(realizations[0].field_refs) != 1:
                return _failure(
                    SourceStrategyVerificationFailureReason.INVALID_BINDING,
                    ref.token,
                )
            realization = realizations[0]
            source = request.source_catalog.source(realization.source_ref)
            field = next(
                (
                    item
                    for item in source.fields
                    if item.field_ref == realization.field_refs[0]
                ),
                None,
            )
            if field is None:
                return _failure(
                    SourceStrategyVerificationFailureReason.INVALID_BINDING,
                    ref.token,
                )
            try:
                concrete_types[ref] = semantic_type_for_row_source_type(field.type)
            except ValueError:
                return _failure(
                    SourceStrategyVerificationFailureReason.INVALID_BINDING,
                    ref.token,
                )
        try:
            infer_expression_types(
                request.index,
                fact_type_by_ref=concrete_types,
            )
        except ValueError:
            return _failure(
                SourceStrategyVerificationFailureReason.INVALID_BINDING,
                *(ref.token for ref in concrete_types),
            )
    return None


def _identity_contract(
    identity_ref: str | None,
    *,
    request: SemanticSourceBindingRequest,
) -> tuple[str, str] | None:
    if identity_ref is None:
        return None
    identity = request.source_catalog.identity(identity_ref)
    return identity.entity_kind, identity.key_id


def _verify_output_dependencies(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> SourceStrategyVerificationFailure | None:
    branch_ids = {item.branch_id for item in request.strategy.branches}
    required_fact_branches = request.required_fact_branches(
        plan.invocation_applications, plan.subject_binding
    )
    failed: list[str] = []
    for requirement_ref, realizations in plan.fact_bindings.items():
        term = request.index.term_by_ref[FactLocalRef.from_token(requirement_ref)]
        if not isinstance(term, FactTerm) or isinstance(
            term.value_type, IdentifierType
        ):
            continue
        for realization in realizations:
            source = request.source_catalog.source(realization.source_ref)
            fields = {field.field_ref: field for field in source.fields}
            if any(
                field_ref not in fields
                or not row_source_type_supports_semantic_type(
                    fields[field_ref].type,
                    term.value_type,
                )
                for field_ref in realization.field_refs
            ):
                failed.append(requirement_ref)
    for semantic_ref in request.index.source_requirement_refs:
        token = semantic_ref.token
        if semantic_ref.kind is FactLocalKind.SET:
            realized_branch_ids = {
                item.branch_id for item in plan.set_bindings.get(token, ())
            }
        elif semantic_ref.kind is FactLocalKind.FACT:
            realized_branch_ids = {
                item.branch_id for item in plan.fact_bindings.get(token, ())
            }
        elif semantic_ref.kind is FactLocalKind.ASSOCIATION:
            realized_branch_ids = {
                item.branch_id for item in plan.association_bindings.get(token, ())
            }
        else:
            continue
        expected_branch_ids = (
            set(required_fact_branches.get(token, ()))
            if semantic_ref.kind is FactLocalKind.FACT
            else branch_ids
        )
        if not expected_branch_ids <= realized_branch_ids:
            failed.append(token)
    if failed:
        return _failure(
            SourceStrategyVerificationFailureReason.INCOMPLETE_OUTPUT_DEPENDENCY,
            *failed,
        )
    return None


def _verify_associations(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> SourceStrategyVerificationFailure | None:
    branches = {item.branch_id: item for item in request.strategy.branches}
    failed: list[str] = []
    for requirement_ref, realizations in plan.association_bindings.items():
        for realization in realizations:
            branch = branches[realization.branch_id]
            if not set(realization.source_refs) <= set(branch.source_refs):
                failed.append(requirement_ref)
            if (
                realization.kind is AssociationRealizationKind.DECLARED_RELATION
                and realization.relation_evidence_ref
                not in branch.relation_evidence_refs
            ):
                failed.append(requirement_ref)
    if failed:
        return _failure(
            SourceStrategyVerificationFailureReason.INCOMPLETE_ASSOCIATION,
            *failed,
        )
    return None


def _verify_invocation_completeness(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> SourceStrategyVerificationFailure | None:
    failed: list[str] = []
    for owner_ref, realizations in plan.boolean_bindings.items():
        if any(
            not request.invocation_preserves_population(
                owner_ref, branch_id=realization.branch_id
            )
            for realization in realizations
        ) and any(
            mechanic.kind is SourceMechanicKind.INVOCATION_PREDICATE
            for realization in realizations
            for mechanic in realization.mechanics
        ):
            failed.append(owner_ref)
    for application in plan.invocation_applications:
        if (
            application.owner_ref is not None
            and not request.invocation_preserves_population(
                application.owner_ref, branch_id=application.branch_id
            )
        ):
            failed.append(application.owner_ref)
        if application.owner_ref in {item.requirement_ref for item in request.index.boolean_requirements} and any(
            not request.invocation_target_matches_owner(application.source_ref, target.target_ref,
                owner_ref=application.owner_ref, branch_id=application.branch_id, projection=target.projection, value_ref=target.value_ref)
            for target in application.target_applications
        ):
            failed.append(application.owner_ref)
        choice = next((item for item in request.source_catalog.choice_values if item.value_ref == application.value_ref), None)
        if choice is not None and application.owner_ref in {item.requirement_ref for item in request.index.boolean_requirements}:
            if application.owner_ref not in request.choice_value_requirement_refs(choice, branch_id=application.branch_id):
                failed.append(application.owner_ref)
    from fervis.lookup.source_binding.population_values import contradictory_request_predicates
    failed.extend(contradictory_request_predicates(plan, request=request))
    from fervis.lookup.source_binding.occurrences import occurrence_scope

    for branch in request.strategy.branches:
        scope = occurrence_scope(request, plan, branch.branch_id)
        for occurrence in scope.occurrences:
            applications = scope.applications_for(
                request, plan, branch_id=branch.branch_id, occurrence=occurrence
            )
            failed.extend(
                application.owner_ref
                for application in applications
                if application.owner_ref is not None
                and not request.invocation_preserves_population(
                    application.owner_ref,
                    branch_id=branch.branch_id,
                    affected_set_refs=occurrence.set_refs,
                )
            )
            applied = {
                target.target_ref
                for application in applications
                for target in application.target_applications
            }
            source = request.source_catalog.source(occurrence.source_ref)
            from fervis.lookup.source_binding.invocation_bindings import (
                invocation_binding_sets,
            )

            try:
                alternatives = invocation_binding_sets(
                    request=request,
                    plan=plan,
                    scope=scope,
                    occurrence=occurrence,
                    branch_id=branch.branch_id,
                )
                if len(alternatives) > 1 and not source.stable_grain_field_refs:
                    failed.append(f"invocation_union_identity:{occurrence.id}")
            except ValueError:
                failed.append(f"invocation_alternatives:{occurrence.id}")
            missing = {
                param.param_ref
                for param in source.params
                if requires_caller_supplied_input(param) and not request.access_supplies(source.id,param.param_ref)
            } - applied
            failed.extend(sorted(missing))
    if failed:
        return _failure(
            SourceStrategyVerificationFailureReason.INCOMPLETE_INVOCATION,
            *failed,
        )
    return None


def _qualification_guarantee(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> QualificationGuarantee | SourceStrategyVerificationFailure:
    proofs: dict[BooleanAtomRef, tuple[str, ...]] = {}
    failed: list[str] = []
    for requirement in request.index.boolean_requirements:
        if requirement.use_site is not BooleanRequirementUseSite.POPULATION:
            continue
        realizations = plan.boolean_bindings.get(requirement.requirement_ref, ())
        if not realizations or any(not item.mechanics for item in realizations):
            failed.append(requirement.requirement_ref)
            continue
        proofs[requirement.atom_ref] = tuple(
            dict.fromkeys(
                ref
                for realization in realizations
                for mechanic in realization.mechanics
                for ref in (
                    mechanic.source_ref,
                    *mechanic.contract_evidence_refs,
                )
            )
        )
    if failed:
        return _failure(
            SourceStrategyVerificationFailureReason.INCOMPLETE_QUALIFICATION,
            *failed,
        )
    return QualificationGuarantee(
        requested_fact_id=request.index.requested_fact_id,
        formula=request.index.qualification,
        atom_proofs=tuple(
            QualificationAtomProof(atom_ref=atom, proof_refs=refs)
            for atom, refs in sorted(proofs.items())
        ),
    )


def _subject_guarantee(
    plan: SourceBindingPlan,
    *,
    request: SemanticSourceBindingRequest,
) -> SubjectGuarantee | SourceStrategyVerificationFailure:
    binding = plan.subject_binding
    obligation = request.index.subject_obligation
    failed: list[str] = []
    proof_refs: list[str] = [request.source_catalog.contract_snapshot.ref]
    branch_ids = {branch.branch_id for branch in request.strategy.branches}
    realizations_by_branch = {
        realization.branch_id: realization
        for realization in binding.branch_realizations
    }
    if (
        binding.subject_ref != obligation.subject_set_ref.token
        or len(realizations_by_branch) != len(binding.branch_realizations)
        or set(realizations_by_branch) != branch_ids
    ):
        return _failure(
            SourceStrategyVerificationFailureReason.INVALID_BINDING,
            obligation.subject_set_ref.token,
        )
    for branch in request.strategy.branches:
        realization = realizations_by_branch[branch.branch_id]
        from fervis.lookup.source_binding.choice_requirements import (
            requirement_choice_surfaces,
        )

        expected = {
            surface.surface_ref
            for surface in requirement_choice_surfaces(request, branch.branch_id)
        }
        reviewed = [review.surface_ref for review in realization.surface_reviews]
        if len(set(reviewed)) != len(reviewed) or set(reviewed) != expected:
            failed.append(branch.branch_id)
            continue
        for review in realization.surface_reviews:
            surface = request.source_catalog.choice_surface(review.surface_ref)
            if review.owner_set_ref is not None:
                failed.append(review.surface_ref)
            if {c.choice_ref for c in review.choice_reviews} != {
                c.value_ref for c in surface.values
            }:
                failed.append(review.surface_ref)
            for choice in review.choice_reviews:
                allowed = request.explicit_subject_requirement_refs(
                    request.source_catalog.choice_value(choice.choice_ref),
                    branch_id=branch.branch_id,
                )
                if not set(choice.selection_requirement_refs) <= set(allowed):
                    failed.append(choice.choice_ref)
            if set(review.included_choice_refs) != {
                c.choice_ref
                for c in review.choice_reviews
                if c.selection_requirement_refs
            }:
                failed.append(review.surface_ref)
            proof_refs.append(review.surface_ref)
            proof_refs.extend(
                ref
                for mechanic in review.mechanics
                for ref in mechanic.contract_evidence_refs
            )
    from fervis.lookup.source_binding.parameter_coverage import (
        uncovered_parameter_scopes,
    )

    failed.extend(uncovered_parameter_scopes(plan, request=request))
    if failed:
        return _failure(
            SourceStrategyVerificationFailureReason.INSUFFICIENT_COMPLETENESS,
            *failed,
        )
    return SubjectGuarantee(
        requested_fact_id=request.index.requested_fact_id,
        subject_set_ref=obligation.subject_set_ref.token,
        interpretation=type(obligation).__name__,
        proof_refs=tuple(dict.fromkeys(proof_refs)),
    )


def _branch_mechanics(
    plan: SourceBindingPlan,
    branch_id: str,
) -> tuple[SourceMechanic, ...]:
    return tuple(
        mechanic
        for realizations in plan.boolean_bindings.values()
        for realization in realizations
        if realization.branch_id == branch_id
        for mechanic in realization.mechanics
    ) + tuple(
        mechanic
        for realization in plan.subject_binding.branch_realizations
        if realization.branch_id == branch_id
        for review in realization.surface_reviews
        for mechanic in review.mechanics
    )


def _failure(
    reason: SourceStrategyVerificationFailureReason,
    *requirement_refs: str,
) -> SourceStrategyVerificationFailure:
    return SourceStrategyVerificationFailure(
        reason=reason,
        failed_requirement_refs=tuple(dict.fromkeys(requirement_refs)),
        evidence_refs=(),
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
