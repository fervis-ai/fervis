"""Validate model output and compile the canonical source-binding plan."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from fervis.lookup.available_sources import SourceChoiceSurfaceKind
from fervis.lookup.answer_program.values import ValueProjectionKind
from fervis.lookup.source_binding.param_binding_sets import (
    finite_choice_parameter_is_omittable,
)
from fervis.lookup.source_binding import semantic_provider_contract as output
from fervis.lookup.source_binding.semantic import (
    AssociationRealization,
    AssociationRealizationKind,
    BooleanRequirementRealization,
    FactRealization,
    FactRealizationKind,
    InvocationTargetApplication,
    InvocationValueApplication,
    SemanticSourceBindingRequest,
    SetRealization,
    SourceBindingPlan,
    SourceMechanic,
    SourceMechanicKind,
    SubjectObligationBinding,
    SubjectObligationRealization,
    SubjectChoiceReview,
    SubjectSurfaceReview,
)
from fervis.lookup.question_contract import FactLocalRef, FactTerm, RawDataRecord
from fervis.lookup.relation_catalog.row_sources import RowSourceIdentityEvidence
from fervis.lookup.semantic_types import IdentifierType


def compile_source_binding_plan(
    payload: dict[str, object],
    *,
    request: SemanticSourceBindingRequest,
) -> SourceBindingPlan:
    parsed = output.SemanticSourceBindingOutput.parse(payload)
    branch_ids = tuple(item.branch_id for item in request.strategy.branches)
    _exact_keys(
        parsed.set_bindings,
        expected=_required_term_refs(request, kind="set"),
        label="set binding",
    )
    _exact_keys(
        parsed.fact_bindings,
        expected=_required_term_refs(request, kind="fact"),
        label="fact binding",
    )
    _exact_keys(
        parsed.association_bindings,
        expected=tuple(
            item.token for item in request.index.association_requirement_refs
        ),
        label="association binding",
    )
    set_bindings = {
        ref: tuple(
            _set_realization(item, request=request)
            for item in _exact_branch_realizations(
                values, branch_ids=branch_ids, label="set realization"
            )
        )
        for ref, values in parsed.set_bindings.items()
    }
    association_bindings = {
        ref: tuple(
            _association_realization(item, request=request)
            for item in _exact_branch_realizations(
                values, branch_ids=branch_ids, label="association realization"
            )
        )
        for ref, values in parsed.association_bindings.items()
    }
    _exact_keys(
        parsed.resolved_input_applications,
        expected=branch_ids,
        label="resolved input application branch",
    )
    ordinary_applications = tuple(
        _resolved_input_application(
            item,
            branch_id=branch_id,
            application_ref=f"application_{index}",
            request=request,
        )
        for index, (branch_id, item) in enumerate(
            (
                (branch_id, item)
                for branch_id, applications in (
                    parsed.resolved_input_applications.items()
                )
                for item in applications
            ),
            start=1,
        )
    )
    subject_binding, subject_applications = _subject_binding(
        parsed.subject_binding,
        request=request,
    )
    _validate_choice_requirement_ownership(subject_binding)
    invocation_applications = (
        *ordinary_applications,
        *subject_applications,
    )
    _applications_by_ref(invocation_applications)
    _validate_unique_resolved_input_targets(ordinary_applications)
    _validate_required_invocation_targets(
        invocation_applications,
        request=request,
    )
    required_fact_branches = request.required_fact_branches(invocation_applications)
    fact_bindings = {
        ref: tuple(
            _fact_realization(
                item,
                fact_ref=ref,
                set_bindings=set_bindings,
                request=request,
            )
            for item in (
                _required_branch_realizations(
                    values,
                    branch_ids=required_fact_branches.get(ref, ()),
                    label="fact realization",
                )
            )
        )
        for ref, values in parsed.fact_bindings.items()
    }
    boolean_bindings = _derive_boolean_bindings(
        invocation_applications,
        fact_bindings=fact_bindings,
        subject_binding=subject_binding,
        request=request,
    )
    return SourceBindingPlan(
        strategy=request.strategy,
        set_bindings=set_bindings,
        fact_bindings=fact_bindings,
        association_bindings=association_bindings,
        invocation_applications=invocation_applications,
        boolean_bindings=boolean_bindings,
        subject_binding=subject_binding,
    )


def _subject_binding(
    item: output.SubjectObligationBindingOutput,
    *,
    request: SemanticSourceBindingRequest,
) -> tuple[SubjectObligationBinding, tuple[InvocationValueApplication, ...]]:
    expected_subject_ref = request.index.subject_obligation.subject_set_ref.token
    if item.subject_ref != expected_subject_ref:
        raise ValueError("subject obligation references the wrong subject")
    branch_ids = tuple(branch.branch_id for branch in request.strategy.branches)
    parsed_realizations = tuple(
        _subject_realization(
            value,
            request=request,
        )
        for value in _exact_branch_realizations(
            item.branch_realizations,
            branch_ids=branch_ids,
            label="subject obligation realization",
        )
    )
    return (
        SubjectObligationBinding(
            subject_ref=expected_subject_ref,
            branch_realizations=tuple(item[0] for item in parsed_realizations),
        ),
        tuple(
            application
            for _, applications in parsed_realizations
            for application in applications
        ),
    )


def _validate_choice_requirement_ownership(
    binding: SubjectObligationBinding,
) -> None:
    """Require one finite-choice surface per requirement in each branch."""

    for realization in binding.branch_realizations:
        surfaces_by_requirement: dict[str, set[str]] = {}
        for surface in realization.surface_reviews:
            for review in surface.choice_reviews:
                if review.requirement_ref is not None:
                    surfaces_by_requirement.setdefault(
                        review.requirement_ref, set()
                    ).add(surface.surface_ref)
        if any(len(surface_refs) != 1 for surface_refs in surfaces_by_requirement.values()):
            raise ValueError(
                "Boolean requirement is mapped by more than one finite-choice surface"
            )


def _subject_realization(
    item: output.SubjectObligationRealizationOutput,
    *,
    request: SemanticSourceBindingRequest,
) -> tuple[SubjectObligationRealization, tuple[InvocationValueApplication, ...]]:
    expected_surfaces = _branch_subject_surfaces(item.branch_id, request=request)
    if isinstance(request.index.subject_obligation, RawDataRecord):
        if item.finite_choice_reviews:
            raise ValueError("raw-record subjects suppress ordinary-instance reviews")
        parsed_reviews: tuple[
            tuple[SubjectSurfaceReview, tuple[InvocationValueApplication, ...]], ...
        ] = ()
    else:
        _exact_keys(
            item.finite_choice_reviews,
            expected=expected_surfaces,
            label="finite choice review",
        )
        parsed_reviews = tuple(
            _subject_surface_review(
                review,
                surface_ref,
                item.branch_id,
                request=request,
            )
            for surface_ref, review in item.finite_choice_reviews.items()
        )
    return (
        SubjectObligationRealization(
            branch_id=item.branch_id,
            surface_reviews=tuple(review for review, _ in parsed_reviews),
        ),
        tuple(
            application
            for _, applications in parsed_reviews
            for application in applications
        ),
    )


def _subject_surface_review(
    item: output.SubjectSurfaceReviewOutput,
    surface_ref: str,
    branch_id: str,
    *,
    request: SemanticSourceBindingRequest,
) -> tuple[SubjectSurfaceReview, tuple[InvocationValueApplication, ...]]:
    surface = _subject_surface(surface_ref, branch_id, request=request)
    choice_refs_by_value = {value.value: value.value_ref for value in surface.values}
    expected_choices = tuple(choice_refs_by_value)
    _exact_keys(
        item.choice_reviews,
        expected=expected_choices,
        label="subject choice review",
    )
    boolean_requirement_refs = {
        requirement.requirement_ref
        for requirement in request.index.boolean_requirements
    }

    def _review(choice_value, review) -> SubjectChoiceReview:
        choice_ref = choice_refs_by_value[choice_value]
        compatible_requirement_refs = set(
            request.choice_value_requirement_refs(
                request.source_catalog.choice_value(choice_ref),
                branch_id=branch_id,
            )
        )
        requirement_ref = review.requirement_ref
        if requirement_ref is not None and (
            requirement_ref not in compatible_requirement_refs
        ):
            raise ValueError(
                "choice review selects an incompatible Boolean requirement"
            )
        if (review.requirement_mapping_basis is None) != (
            requirement_ref is None
        ):
            raise ValueError(
                "choice requirement mapping basis and ref must be selected together"
            )
        matched_excluded_role = (
            None
            if review.matched_excluded_role == "NONE"
            else _exclusion_reason(review.matched_excluded_role)
        )
        explicit_override_applies = bool(
            matched_excluded_role is not None
            and requirement_ref in boolean_requirement_refs
        )
        return SubjectChoiceReview(
            choice_ref=choice_ref,
            choice_domain_meaning=_text(review.choice_domain_meaning),
            role_match_basis=_text(review.role_match_basis),
            matched_excluded_role=matched_excluded_role,
            explicit_user_override_basis=(
                _text(review.requirement_mapping_basis)
                if explicit_override_applies
                else "No explicit requirement application selects this choice."
            ),
            choice_inclusion_basis=_text(review.choice_inclusion_basis),
            choice_included=review.choice_inclusion == "INCLUDE",
            requirement_mapping_basis=(
                _text(review.requirement_mapping_basis)
                if review.requirement_mapping_basis is not None
                else None
            ),
            requirement_ref=requirement_ref,
            authored_explicit_user_override_applies=explicit_override_applies,
        )

    choice_reviews = tuple(
        _review(choice_value, review)
        for choice_value, review in item.choice_reviews.items()
    )
    expected_choice_refs = tuple(choice_refs_by_value.values())
    retained_reviews = tuple(review for review in choice_reviews if review.included)
    retained_values = tuple(review.choice_ref for review in retained_reviews)
    excluded_choices = tuple(
        review.choice_ref for review in choice_reviews if not review.included
    )
    applications: tuple[InvocationValueApplication, ...]
    mechanics: tuple[SourceMechanic, ...]
    if excluded_choices and not retained_values:
        raise ValueError("subject surface excludes every shown choice")
    mapped_reviews = tuple(
        review for review in choice_reviews if review.requirement_ref is not None
    )
    if any(not review.included for review in mapped_reviews):
        raise ValueError(
            "choice requirement selects a choice excluded by subject review"
        )
    selected_reviews = mapped_reviews or retained_reviews
    if surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER:
        selected_values = tuple(
            request.source_catalog.choice_value(review.choice_ref).value
            for review in selected_reviews
        )
        source = request.source_catalog.source(surface.source_ref)
        param = next(
            item for item in source.params if item.param_ref == surface.surface_ref
        )
        omittable = finite_choice_parameter_is_omittable(
            required=param.required,
            default=param.default,
            choices=param.choices,
            included_values=selected_values,
        )
        if omittable:
            applications = ()
        else:
            applications = tuple(
                _subject_choice_application(
                    branch_id=branch_id,
                    source_ref=surface.source_ref,
                    target_ref=surface.surface_ref,
                    value_ref=review.choice_ref,
                    owner_ref=review.requirement_ref,
                    ordinal=ordinal,
                    mapping_basis=(
                        review.requirement_mapping_basis
                        or "The reviewed finite-choice parameter retains this "
                        "subject choice."
                    ),
                    request=request,
                )
                for ordinal, review in enumerate(
                    (
                        selected_reviews
                        if mapped_reviews
                        or excluded_choices
                        or request.choice_surface_requires_application(surface)
                        else ()
                    ),
                    start=1,
                )
            )
        if excluded_choices or mapped_reviews:
            mechanics = (
                SourceMechanic(
                    mapping_basis=(
                        "The reviewed finite-choice parameter retains exactly the "
                        "choices inside the requested subject population."
                    ),
                    source_ref=surface.source_ref,
                    application_refs=tuple(
                        application.application_ref
                        for application in applications
                    ),
                    contract_evidence_refs=(
                        request.source_catalog.contract_snapshot.ref,
                        surface.surface_ref,
                        *expected_choice_refs,
                    ),
                    kind=SourceMechanicKind.INVOCATION_PREDICATE,
                ),
            )
        else:
            mechanics = ()
    else:
        applications = ()
        if excluded_choices or mapped_reviews:
            mechanics = (
                SourceMechanic(
                    mapping_basis=(
                        "The reviewed returned field retains exactly the choices "
                        "inside the requested subject population."
                    ),
                    source_ref=surface.source_ref,
                    application_refs=(),
                    contract_evidence_refs=(
                        request.source_catalog.contract_snapshot.ref,
                        surface.surface_ref,
                        *expected_choice_refs,
                    ),
                    kind=SourceMechanicKind.RETURNED_ROW_PREDICATE,
                ),
            )
        else:
            mechanics = ()
    return (
        SubjectSurfaceReview(
            surface_ref=surface_ref,
            surface_mapping_basis=_text(item.surface_mapping_basis),
            choice_reviews=choice_reviews,
            included_choice_refs=retained_values,
            mechanics=mechanics,
        ),
        applications,
    )


def _subject_choice_application(
    *,
    branch_id: str,
    source_ref: str,
    target_ref: str,
    value_ref: str,
    owner_ref: str | None,
    ordinal: int,
    mapping_basis: str,
    request: SemanticSourceBindingRequest,
) -> InvocationValueApplication:
    matches = tuple(
        option
        for option in request.invocation_projection_options
        if option.source_ref == source_ref
        and option.target_ref == target_ref
        and option.value_ref == value_ref
        and option.projection is ValueProjectionKind.WHOLE_VALUE
        and option.component_ref is None
    )
    if len(matches) != 1:
        raise ValueError("ordinary subject choice lacks one parameter projection")
    [option] = matches
    if owner_ref is not None:
        _validate_application_owner(owner_ref, option=option, request=request)
    application_owner = owner_ref or "subject_obligation"
    application_ref = (
        f"choice_application:{branch_id}:{target_ref}:{application_owner}:{ordinal}"
    )
    return InvocationValueApplication(
        application_ref=application_ref,
        branch_id=branch_id,
        source_ref=source_ref,
        value_ref=value_ref,
        owner_ref=owner_ref,
        target_applications=(
            InvocationTargetApplication(
                mapping_basis=mapping_basis,
                target_ref=target_ref,
                value_ref=value_ref,
                projection=option.projection,
                component_ref=option.component_ref,
            ),
        ),
    )


def _exclusion_reason(value: str) -> str:
    allowed = {
        "NOT_REALIZED",
        "CANCELED_OR_VOIDED",
        "FAILED_OR_REJECTED_BEFORE_EFFECT",
        "REVERSED_OR_CORRECTION_ARTIFACT",
        "TEST_PLACEHOLDER_OR_DEMO",
        "SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT",
    }
    if value not in allowed:
        raise ValueError("subject choice review uses an unknown exclusion reason")
    return value


def _branch_subject_surfaces(
    branch_id: str,
    *,
    request: SemanticSourceBindingRequest,
) -> tuple[str, ...]:
    branch = next(
        item for item in request.strategy.branches if item.branch_id == branch_id
    )
    return tuple(
        surface.surface_ref
        for source_ref in branch.source_refs
        for surface in request.source_catalog.choice_surfaces
        if surface.source_ref == source_ref
    )


def _subject_surface(
    surface_ref: str,
    branch_id: str,
    *,
    request: SemanticSourceBindingRequest,
):
    branch = next(
        item for item in request.strategy.branches if item.branch_id == branch_id
    )
    for surface in request.source_catalog.choice_surfaces:
        if (
            surface.source_ref in branch.source_refs
            and surface.surface_ref == surface_ref
        ):
            return surface
    raise ValueError("subject surface does not belong to its strategy branch")


def _set_realization(
    item: output.SetRealizationOutput,
    *,
    request: SemanticSourceBindingRequest,
) -> SetRealization:
    source = _branch_source(item.branch_id, item.source_ref, request=request)
    identity_ref = item.identity_ref
    if identity_ref is None:
        fields: tuple[str, ...] = ()
    else:
        identity = request.source_catalog.identity(identity_ref)
        if identity.source_ref != source.id:
            raise ValueError("set identity belongs to another source")
        fields = identity.field_refs
    return SetRealization(
        branch_id=item.branch_id,
        mapping_basis=_text(item.mapping_basis),
        source_ref=item.source_ref,
        identity_ref=identity_ref,
        identity_field_refs=fields,
        contract_evidence_refs=_local_evidence(
            request.source_catalog.contract_snapshot.ref,
            item.source_ref,
            identity_ref,
            *fields,
        ),
    )


def _fact_realization(
    item: output.IdentifierFactRealizationOutput | output.ReturnedFactRealizationOutput,
    *,
    fact_ref: str,
    set_bindings: dict[str, tuple[SetRealization, ...]],
    request: SemanticSourceBindingRequest,
) -> FactRealization:
    source = _branch_source(item.branch_id, item.source_ref, request=request)
    if isinstance(item, output.IdentifierFactRealizationOutput):
        identity = _identifier_fact_identity(
            fact_ref=fact_ref,
            branch_id=item.branch_id,
            source_ref=source.id,
            set_bindings=set_bindings,
            request=request,
        )
        kind = FactRealizationKind.ENTITY_KEY
        fields = identity.field_refs
    else:
        fields = _source_fields((item.field_ref,), source=source, label="fact field")
        kind = FactRealizationKind.RETURNED_FIELD
    return FactRealization(
        branch_id=item.branch_id,
        mapping_basis=_text(item.mapping_basis),
        source_ref=item.source_ref,
        kind=kind,
        identity_ref=(
            identity.identity_ref
            if isinstance(item, output.IdentifierFactRealizationOutput)
            else None
        ),
        field_refs=fields,
        contract_evidence_refs=_local_evidence(
            request.source_catalog.contract_snapshot.ref,
            item.source_ref,
            (
                identity.identity_ref
                if isinstance(item, output.IdentifierFactRealizationOutput)
                else None
            ),
            *fields,
        ),
    )


def _identifier_fact_identity(
    *,
    fact_ref: str,
    branch_id: str,
    source_ref: str,
    set_bindings: dict[str, tuple[SetRealization, ...]],
    request: SemanticSourceBindingRequest,
) -> RowSourceIdentityEvidence:
    semantic_ref = next(
        ref
        for ref in request.index.inferred_type_by_ref
        if isinstance(ref, FactLocalRef) and ref.token == fact_ref
    )
    term = request.index.term_by_ref[semantic_ref]
    if not isinstance(term, FactTerm) or not isinstance(
        term.value_type, IdentifierType
    ):
        raise ValueError("identifier fact realization requires an Identifier fact")
    identified_set_ref = request.index.fact_local_ref_by_local_id[
        term.value_type.set_ref
    ].token
    set_realization = next(
        (
            item
            for item in set_bindings[identified_set_ref]
            if item.branch_id == branch_id
        ),
        None,
    )
    if set_realization is None or set_realization.identity_ref is None:
        raise ValueError("identifier fact requires an identified-set identity contract")
    selected_identity = request.source_catalog.identity(set_realization.identity_ref)
    matches = tuple(
        identity
        for identity in request.source_catalog.identity_evidence
        if identity.source_ref == source_ref
        and identity.entity_kind == selected_identity.entity_kind
        and identity.key_id == selected_identity.key_id
    )
    if len(matches) != 1:
        raise ValueError(
            "identifier fact source must declare exactly one matching identity contract"
        )
    return matches[0]


def _association_realization(
    item: output.AssociationRealizationOutput,
    *,
    request: SemanticSourceBindingRequest,
) -> AssociationRealization:
    sources: tuple[str, ...]
    relation = next(
        (
            value
            for value in request.source_catalog.relation_evidence
            if value.evidence_ref == item.realization_ref
        ),
        None,
    )
    if relation is None:
        _branch_source(item.branch_id, item.realization_ref, request=request)
        kind = AssociationRealizationKind.CO_RESIDENT
        sources = (item.realization_ref,)
        relation_ref = None
    else:
        sources = (relation.left_source_ref, relation.right_source_ref)
        for source_ref in sources:
            _branch_source(item.branch_id, source_ref, request=request)
        kind = AssociationRealizationKind.DECLARED_RELATION
        relation_ref = relation.evidence_ref
    return AssociationRealization(
        branch_id=item.branch_id,
        mapping_basis=_text(item.mapping_basis),
        kind=kind,
        source_refs=sources,
        relation_evidence_ref=relation_ref,
        contract_evidence_refs=_association_evidence(
            request=request,
            source_refs=sources,
            relation_evidence_ref=relation_ref,
        ),
    )


def _resolved_input_application(
    item: output.ResolvedInputApplicationOutput,
    *,
    branch_id: str,
    application_ref: str,
    request: SemanticSourceBindingRequest,
) -> InvocationValueApplication:
    branch = next(
        branch for branch in request.strategy.branches if branch.branch_id == branch_id
    )
    option = next(
        (
            value
            for value in request.authored_invocation_projection_options
            if value.value_ref == item.value_ref
            and value.target_ref == item.target_ref
            and (value.component_ref or value.projection.value) == item.value_component
        ),
        None,
    )
    if option is None or option.source_ref not in branch.source_refs:
        raise ValueError("invocation application selects an unavailable projection")
    _validate_application_owner(item.owner_ref, option=option, request=request)
    _branch_source(branch_id, option.source_ref, request=request)
    return InvocationValueApplication(
        application_ref=application_ref,
        branch_id=branch_id,
        source_ref=option.source_ref,
        value_ref=option.value_ref,
        owner_ref=item.owner_ref,
        target_applications=(
            InvocationTargetApplication(
                mapping_basis=_text(item.mapping_basis),
                target_ref=option.target_ref,
                value_ref=option.value_ref,
                projection=option.projection,
                component_ref=option.component_ref,
            ),
        ),
    )


def _validate_unique_resolved_input_targets(
    applications: tuple[InvocationValueApplication, ...],
) -> None:
    selected_targets: set[tuple[str, str]] = set()
    for application in applications:
        [target] = application.target_applications
        target_key = (application.branch_id, target.target_ref)
        if target_key in selected_targets:
            raise ValueError("resolved input applications repeat a target")
        selected_targets.add(target_key)


def _validate_required_invocation_targets(
    applications: tuple[InvocationValueApplication, ...],
    *,
    request: SemanticSourceBindingRequest,
) -> None:
    selected: set[tuple[str, str, str]] = set()
    for application in applications:
        [target] = application.target_applications
        key = (application.branch_id, application.owner_ref or "", target.target_ref)
        selected.add(key)
    for owner_ref in request.invocation_application_owner_refs:
        for branch in request.strategy.branches:
            required = {
                (branch.branch_id, owner_ref, target_ref)
                for target_ref in request.required_invocation_target_refs(
                    owner_ref,
                    branch_id=branch.branch_id,
                )
            }
            if not required <= selected:
                raise ValueError("required invocation target is not applied")


def _validate_application_owner(owner_ref: str, *, option, request) -> None:
    requirement_refs = {
        item.requirement_ref for item in request.index.boolean_requirements
    }
    if owner_ref in requirement_refs:
        choice_refs = {
            value.value_ref for value in request.source_catalog.choice_values
        }
        if (
            option.value_ref not in choice_refs
            and option.value_ref not in request.requirement_value_refs(owner_ref)
        ):
            raise ValueError(
                "invocation projection value is unrelated to its requirement"
            )
        return
    if owner_ref != f"source_required:{option.target_ref}":
        raise ValueError("invocation projection has an invalid owner")
    source = request.source_catalog.source(option.source_ref)
    param = next(
        (item for item in source.params if item.param_ref == option.target_ref),
        None,
    )
    if param is None or not param.required or param.default is not None:
        raise ValueError("source-required application targets an optional parameter")


def _applications_by_ref(
    values: tuple[InvocationValueApplication, ...],
) -> dict[str, InvocationValueApplication]:
    output = {item.application_ref: item for item in values}
    if len(output) != len(values):
        raise ValueError("invocation application refs must be unique")
    return output


def _derive_boolean_bindings(
    applications: tuple[InvocationValueApplication, ...],
    *,
    fact_bindings: dict[str, tuple[FactRealization, ...]],
    subject_binding: SubjectObligationBinding,
    request: SemanticSourceBindingRequest,
) -> dict[str, tuple[BooleanRequirementRealization, ...]]:
    output_bindings: dict[str, tuple[BooleanRequirementRealization, ...]] = {}
    for requirement in request.index.boolean_requirements:
        realizations: list[BooleanRequirementRealization] = []
        fact_refs = request.requirement_fact_refs(requirement.requirement_ref)
        for branch in request.strategy.branches:
            owned = tuple(
                application
                for application in applications
                if application.branch_id == branch.branch_id
                and application.owner_ref == requirement.requirement_ref
            )
            choice_mechanics = tuple(
                mechanic
                for realization in subject_binding.branch_realizations
                if realization.branch_id == branch.branch_id
                for surface in realization.surface_reviews
                if any(
                    review.requirement_ref == requirement.requirement_ref
                    for review in surface.choice_reviews
                )
                for mechanic in surface.mechanics
            )
            mechanics = (
                _invocation_mechanics(
                    owned, requirement_ref=requirement.requirement_ref
                )
                if owned
                else choice_mechanics
                if choice_mechanics
                else _returned_mechanics(
                    branch.branch_id,
                    fact_refs=fact_refs,
                    fact_bindings=fact_bindings,
                    requirement_ref=requirement.requirement_ref,
                )
            )
            if not mechanics:
                raise ValueError("Boolean requirement lacks source mechanics")
            realizations.append(
                BooleanRequirementRealization(
                    branch_id=branch.branch_id,
                    mechanics=mechanics,
                )
            )
        output_bindings[requirement.requirement_ref] = tuple(realizations)
    return output_bindings


def _invocation_mechanics(
    applications: tuple[InvocationValueApplication, ...],
    *,
    requirement_ref: str,
) -> tuple[SourceMechanic, ...]:
    source_refs = tuple(dict.fromkeys(item.source_ref for item in applications))
    return tuple(
        SourceMechanic(
            mapping_basis=f"Invocation applications enforce {requirement_ref}.",
            source_ref=source_ref,
            application_refs=tuple(
                item.application_ref
                for item in applications
                if item.source_ref == source_ref
            ),
            contract_evidence_refs=tuple(
                dict.fromkeys(
                    evidence_ref
                    for item in applications
                    if item.source_ref == source_ref
                    for target in item.target_applications
                    for evidence_ref in (
                        item.source_ref,
                        target.target_ref,
                        target.value_ref,
                    )
                )
            ),
            kind=SourceMechanicKind.INVOCATION_PREDICATE,
        )
        for source_ref in source_refs
    )


def _returned_mechanics(
    branch_id: str,
    *,
    fact_refs: tuple[str, ...],
    fact_bindings: dict[str, tuple[FactRealization, ...]],
    requirement_ref: str,
) -> tuple[SourceMechanic, ...]:
    realizations = tuple(
        realization
        for fact_ref in fact_refs
        for realization in fact_bindings.get(fact_ref, ())
        if realization.branch_id == branch_id
    )
    source_refs = tuple(dict.fromkeys(item.source_ref for item in realizations))
    return tuple(
        SourceMechanic(
            mapping_basis=f"Returned fact bindings evaluate {requirement_ref}.",
            source_ref=source_ref,
            application_refs=(),
            contract_evidence_refs=tuple(
                dict.fromkeys(
                    evidence_ref
                    for realization in realizations
                    if realization.source_ref == source_ref
                    for evidence_ref in realization.contract_evidence_refs
                )
            ),
            kind=SourceMechanicKind.RETURNED_ROW_PREDICATE,
        )
        for source_ref in source_refs
    )


def _required_term_refs(
    request: SemanticSourceBindingRequest, *, kind: str
) -> tuple[str, ...]:
    return tuple(
        ref.token
        for ref in request.index.source_requirement_refs
        if ref.kind.value == kind
    )


def _exact_keys(
    values: Mapping[str, object], *, expected: tuple[str, ...], label: str
) -> None:
    if set(values) != set(expected):
        raise ValueError(f"{label} requires exact coverage")


def _exact_branch_realizations(values, *, branch_ids: tuple[str, ...], label: str):
    actual = tuple(item.branch_id for item in values)
    if len(actual) != len(set(actual)) or set(actual) != set(branch_ids):
        raise ValueError(f"{label} requires exact branch coverage")
    return values


def _required_branch_realizations(values, *, branch_ids: tuple[str, ...], label: str):
    actual = tuple(item.branch_id for item in values)
    if len(actual) != len(set(actual)) or not set(branch_ids) <= set(actual):
        raise ValueError(f"{label} has incorrect branch coverage")
    return values


def _branch_source(branch_id: str, source_ref: str, *, request):
    branch = next(
        item for item in request.strategy.branches if item.branch_id == branch_id
    )
    if source_ref not in branch.source_refs:
        raise ValueError("binding source does not belong to its strategy branch")
    return request.source_catalog.source(source_ref)


def _source_fields(values: tuple[str, ...], *, source, label: str) -> tuple[str, ...]:
    selected = _unique(values, label=label)
    if not set(selected) <= {field.field_ref for field in source.fields}:
        raise ValueError(f"{label} does not belong to its source")
    return selected


def _association_evidence(
    *,
    request: SemanticSourceBindingRequest,
    source_refs: tuple[str, ...],
    relation_evidence_ref: str | None,
) -> tuple[str, ...]:
    if relation_evidence_ref is None:
        return _local_evidence(
            request.source_catalog.contract_snapshot.ref,
            *source_refs,
        )
    relation = next(
        item
        for item in request.source_catalog.relation_evidence
        if item.evidence_ref == relation_evidence_ref
    )
    return _local_evidence(
        request.source_catalog.contract_snapshot.ref,
        relation.evidence_ref,
        *source_refs,
        *relation.left_field_refs,
        *relation.right_field_refs,
    )


def _local_evidence(*refs: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref is not None))


def _unique(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    selected = tuple(values)
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label} contains duplicates")
    return selected


def _text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("source binding requires a non-empty basis")
    return text


__all__ = ["compile_source_binding_plan"]
