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
    RawDataRecord,
    SubjectRows,
)
from fervis.lookup.relation_catalog.row_sources import (
    row_source_type_supports_semantic_type,
)
from fervis.lookup.semantic_types import IdentifierType
from fervis.lookup.source_binding.model import (
    AssociationRealizationKind,
    SemanticSourceBindingRequest,
    SourceBindingPlan,
    SourceMechanic,
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
    if plan.strategy != request.strategy:
        return _failure(SourceStrategyVerificationFailureReason.INVALID_BINDING)
    output_failure = _verify_output_dependencies(plan, request=request)
    if output_failure is not None:
        return output_failure
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
                for contract in (_identity_contract(item.identity_ref, request=request),)
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
        plan.invocation_applications
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
    for branch in request.strategy.branches:
        applied = {
            target.target_ref
            for application in plan.invocation_applications
            if application.branch_id == branch.branch_id
            for target in application.target_applications
        }
        for source_ref in branch.source_refs:
            source = request.source_catalog.source(source_ref)
            missing = {
                param.param_ref
                for param in source.params
                if requires_caller_supplied_input(param)
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
        expected_surfaces = {
            surface.surface_ref
            for source_ref in branch.source_refs
            for surface in request.source_catalog.choice_surfaces
            if surface.source_ref == source_ref
        }
        reviewed_surfaces = {
            review.surface_ref for review in realization.surface_reviews
        }
        if isinstance(obligation, RawDataRecord):
            if realization.surface_reviews:
                failed.append(realization.branch_id)
            continue
        if (
            len(reviewed_surfaces) != len(realization.surface_reviews)
            or reviewed_surfaces != expected_surfaces
        ):
            failed.append(realization.branch_id)
            continue
        for review in realization.surface_reviews:
            excluded_choice_count = sum(
                not choice.included
                for choice in review.choice_reviews
            )
            if excluded_choice_count and not review.mechanics:
                failed.append(review.surface_ref)
            proof_refs.append(review.surface_ref)
            proof_refs.extend(
                ref
                for mechanic in review.mechanics
                for ref in mechanic.contract_evidence_refs
            )
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
