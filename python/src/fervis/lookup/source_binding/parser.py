"""Validate provider output and compile the canonical source-binding plan."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from fervis.lookup.available_sources import SourceChoiceSurfaceKind
from fervis.lookup.source_binding.occurrences import OccurrenceScope, occurrence_scope
from fervis.lookup.answer_program.values import ValueProjectionKind
from fervis.lookup.source_binding.param_binding_sets import (
    finite_choice_parameter_is_omittable,
)
from fervis.lookup.source_binding.subject_obligations import (
    explicit_override_from_application_owners,
)
from fervis.lookup.source_binding import provider_contract as output
from fervis.lookup.source_binding.membership import (
    SourceMembership,
    membership_scopes,
    requirement_choice_surfaces,
)
from fervis.lookup.source_binding.model import (
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
    SourceRealization,
    SourceMechanic,
    SourceMechanicKind,
    SubjectObligationBinding,
    SubjectObligationRealization,
    SubjectChoiceReview,
    SubjectSurfaceReview,
)
from fervis.lookup.question_contract import (
    AssociationTerm,
    FactLocalRef,
    FactTerm,
    Quantify,
    RelatedRow,
    Coverage,
)
from fervis.lookup.question_contract.analysis import RowDomain
from fervis.lookup.semantic_types import IdentifierType


def compile_source_realization(
    payload: dict[str, object],
    *,
    request: SemanticSourceBindingRequest,
) -> SourceRealization:
    parsed = output.SourceRealizationOutput.parse(payload)
    from fervis.lookup.source_binding.association_choices import (
        AssociationChoice, association_choices, association_endpoints,
    )
    _exact_keys(
        parsed.association_bindings,
        expected=tuple(
            item.token for item in request.index.association_requirement_refs
        ),
        label="association binding",
    )
    connected_sets = {
        endpoint
        for ref in request.index.association_requirement_refs
        for endpoint in association_endpoints(request, ref.token)
    }
    _exact_keys(
        parsed.set_bindings,
        expected=tuple(ref for ref in _required_term_refs(request, kind="set") if ref not in connected_sets),
        label="isolated set binding",
    )
    authored_sets = {ref: list(values) for ref, values in parsed.set_bindings.items()}
    for ref, association_values in parsed.association_bindings.items():
        allowed_associations = frozenset(association_choices(request, ref))
        endpoints = association_endpoints(request, ref)
        for item in association_values:
            if AssociationChoice(item.from_rows_ref, item.to_rows_ref,
                                 item.realization_ref, item.reference_from_set_ref) not in allowed_associations:
                raise ValueError("association realization has incompatible endpoint rows")
            for endpoint, rows_ref in zip(endpoints, (item.from_rows_ref, item.to_rows_ref)):
                candidates = authored_sets.setdefault(endpoint, [])
                existing = next((candidate for candidate in candidates if candidate.branch_id == item.branch_id), None)
                if existing is not None:
                    if existing.rows_ref != rows_ref:
                        raise ValueError("connected associations disagree on their shared set rows")
                else:
                    candidates.append(output.SetRealizationOutput(item.branch_id, item.mapping_basis, rows_ref))
    branch_ids = tuple(item.branch_id for item in request.strategy.branches)
    _exact_keys(
        authored_sets,
        expected=_required_term_refs(request, kind="set"),
        label="set binding",
    )
    _exact_keys(
        parsed.fact_bindings,
        expected=request.model_authored_fact_refs,
        label="fact binding",
    )
    set_bindings = {
        ref: tuple(
            _set_realization(item, request=request, set_ref=ref)
            for item in _exact_branch_realizations(
                values, branch_ids=branch_ids, label="set realization"
            )
        )
        for ref, values in authored_sets.items()
    }
    association_bindings = {
        ref: tuple(
            _association_realization(item, request=request, association_ref=ref)
            for item in _exact_branch_realizations(
                values, branch_ids=branch_ids, label="association realization"
            )
        )
        for ref, values in parsed.association_bindings.items()
    }
    for ref, values in parsed.fact_bindings.items():
        allowed = frozenset(request.returned_field_refs_for_fact(ref))
        if any(value.field_ref not in allowed for value in values):
            raise ValueError("fact realization violates its declared value type")
    returned_fact_bindings = {
        ref: tuple(_fact_realization(item, request=request) for item in values)
        for ref, values in parsed.fact_bindings.items()
    }
    for observed_ref in request.index.observed_fact_refs:
        if isinstance(request.index.value_type(observed_ref), IdentifierType):
            continue
        _exact_branch_realizations(
            returned_fact_bindings.get(observed_ref.token, ()),
            branch_ids=branch_ids,
            label="observed fact realization",
        )
    for ref, realizations in returned_fact_bindings.items():
        term = request.index.term_by_ref[FactLocalRef.from_token(ref)]
        if not isinstance(term, FactTerm):
            raise ValueError("fact binding must reference a fact term")
        owner_ref = request.index.fact_local_ref_by_local_id[term.owner_ref].token
        for value in realizations:
            owners = set_bindings.get(owner_ref, ())
            associations = association_bindings.get(owner_ref, ())
            owner_sources = {
                owner.source_ref
                for owner in owners
                if owner.branch_id == value.branch_id
            }
            owner_sources.update(
                source
                for owner in associations
                if owner.branch_id == value.branch_id
                for source in owner.source_refs
            )
            if value.source_ref not in owner_sources:
                raise ValueError("fact realization must preserve its owning rows")
            from fervis.lookup.relation_catalog.row_sources.model import (
                RowSourceIdentityKind,
            )

            if any(
                owner.identity_ref is not None
                and request.source_catalog.identity(owner.identity_ref).kind
                is RowSourceIdentityKind.ENTITY_REFERENCE
                for owner in owners
                if owner.branch_id == value.branch_id
            ):
                raise ValueError(
                    "an entity reference supplies identity, not the referenced entity's scalar row fields"
                )
    request = request.for_bindings(
        set_bindings, returned_fact_bindings, association_bindings
    )
    realization = SourceRealization(
        request, set_bindings, returned_fact_bindings, association_bindings
    )
    for branch in request.strategy.branches:
        occurrence_scope(request, realization, branch.branch_id)
    return realization


def compile_source_binding_plan(
    payload: dict[str, object],
    *,
    membership: SourceMembership,
) -> SourceBindingPlan:
    realization = membership.realization
    parsed = output.SemanticSourceBindingOutput.parse(payload)
    request = realization.request
    set_bindings = realization.set_bindings
    returned_fact_bindings = realization.fact_bindings
    association_bindings = realization.association_bindings
    branch_ids = tuple(branch.branch_id for branch in request.strategy.branches)

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
    finite_choice_applications = _finite_choice_applications(
        parsed.finite_choice_applications,
        request=request,
    )
    invocation_applications = (
        *ordinary_applications,
        *finite_choice_applications,
    )
    subject_binding, subject_applications = _subject_binding(
        parsed.choice_requirement_applications,
        membership=membership,
        requirement_applications=finite_choice_applications,
        request=request,
    )
    invocation_applications = (
        *invocation_applications,
        *subject_applications,
    )
    _applications_by_ref(invocation_applications)
    _validate_required_invocation_targets(
        invocation_applications,
        request=request,
    )
    model_authored_fact_bindings = returned_fact_bindings
    identifier_fact_bindings = {
        fact_ref: _identifier_fact_realizations(
            fact_ref=fact_ref,
            branch_ids=branch_ids,
            set_bindings=set_bindings,
            request=request,
        )
        for fact_ref in request.identifier_fact_refs
    }
    fact_bindings = {
        **model_authored_fact_bindings,
        **identifier_fact_bindings,
    }
    required_facts = request.required_fact_branches(
        invocation_applications, subject_binding
    )
    fact_bindings = {
        ref: tuple(
            value for value in values if value.branch_id in required_facts.get(ref, ())
        )
        for ref, values in fact_bindings.items()
    }
    fact_bindings = {ref: values for ref, values in fact_bindings.items() if values}
    boolean_bindings = _derive_boolean_bindings(
        invocation_applications,
        fact_bindings=fact_bindings,
        set_bindings=set_bindings,
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
    returned_applications,
    *,
    membership: SourceMembership,
    requirement_applications: tuple[InvocationValueApplication, ...],
    request: SemanticSourceBindingRequest,
) -> tuple[SubjectObligationBinding, tuple[InvocationValueApplication, ...]]:
    branch_ids = tuple(branch.branch_id for branch in request.strategy.branches)
    _exact_keys(
        returned_applications,
        expected=branch_ids,
        label="returned choice application branch",
    )
    branches = []
    applications: list[InvocationValueApplication] = []
    for branch in request.strategy.branches:
        returned = returned_applications[branch.branch_id]
        surfaces = requirement_choice_surfaces(request, branch.branch_id)
        _exact_keys(
            returned,
            expected=tuple(s.surface_ref for s in surfaces),
            label="returned choice application surface",
        )
        reviews = []
        scopes = membership_scopes(membership.realization)[branch.branch_id]
        occurrences = occurrence_scope(request, membership.realization, branch.branch_id)
        for surface_ref in _branch_subject_surfaces(branch.branch_id, request=request):
            surface = request.source_catalog.choice_surface(surface_ref)
            baselines = tuple((scope.owner_set_ref, membership.reviews[branch.branch_id][scope.owner_set_ref][surface_ref])
                              for scope in scopes if surface in scope.surfaces)
            if not baselines:
                if surface_ref not in returned:
                    continue
                baselines = ((None, {
                    "surface_mapping_basis": "Explicit predicate over declared source values.",
                    "choice_reviews": {
                        choice.value: {
                            "choice_domain_meaning": choice.label,
                            "decision_basis": "No ordinary-state restriction is applied to this source value.",
                            "baseline_decision": "INCLUDE",
                        } for choice in surface.values
                    },
                }),)
            for owner_set_ref, baseline in baselines:
                review, extra = _subject_surface_review(
                    returned.get(surface_ref, {}), surface_ref, branch.branch_id,
                    owner_set_ref=owner_set_ref, baseline=baseline, occurrences=occurrences,
                    requirement_applications=requirement_applications, request=request,
                )
                reviews.append(review)
                applications.extend(extra)
        branches.append(SubjectObligationRealization(branch.branch_id, tuple(reviews)))
    return (
        SubjectObligationBinding(
            request.index.subject_obligation.subject_set_ref.token, tuple(branches)
        ),
        tuple(applications),
    )


def _subject_surface_review(
    choice_applications: Mapping[str, output.ChoiceRequirementApplicationOutput],
    surface_ref: str,
    branch_id: str,
    *,
    baseline,
    owner_set_ref: str | None,
    occurrences: OccurrenceScope,
    requirement_applications: tuple[InvocationValueApplication, ...],
    request: SemanticSourceBindingRequest,
) -> tuple[SubjectSurfaceReview, tuple[InvocationValueApplication, ...]]:
    surface = _subject_surface(surface_ref, branch_id, request=request)

    def applies(owner_ref):
        return owner_set_ref is None or occurrences.owner_applies(
            request, owner_ref, source_ref=surface.source_ref,
            occurrence_ref=occurrences.for_set(owner_set_ref).id,
        )

    choice_refs_by_value = {value.value: value.value_ref for value in surface.values}
    expected_choices = tuple(choice_refs_by_value)
    _exact_keys(
        choice_applications,
        expected=expected_choices
        if surface in requirement_choice_surfaces(request, branch_id)
        else (),
        label="subject choice review",
    )
    surface_applications = tuple(
        application
        for application in requirement_applications
        if application.branch_id == branch_id
        and application.source_ref == surface.source_ref
        and application.target_applications[0].target_ref == surface.target_ref
        and applies(application.owner_ref)
    )
    boolean_requirement_refs = frozenset(
        requirement.requirement_ref
        for requirement in request.index.boolean_requirements
    )
    explicit_requirement_applications = tuple(
        application
        for application in surface_applications
        if application.owner_ref in boolean_requirement_refs
    )
    explicitly_selected_choice_refs = {
        application.value_ref for application in explicit_requirement_applications
    }

    for choice_value, review in choice_applications.items():
        choice = request.source_catalog.choice_value(choice_refs_by_value[choice_value])
        allowed = request.explicit_subject_requirement_refs(choice, branch_id=branch_id)
        if len(set(review.selected_by_requirements)) != len(
            review.selected_by_requirements
        ) or not set(review.selected_by_requirements) <= set(allowed):
            raise ValueError(
                "returned choice selects an incompatible explicit requirement"
            )
        if any(applies(owner) for owner in review.selected_by_requirements):
            explicitly_selected_choice_refs.add(choice.value_ref)

    def _review(choice_value, review) -> SubjectChoiceReview:
        choice_ref = choice_refs_by_value[choice_value]
        declared = baseline["choice_reviews"][choice_value]
        decision = declared["baseline_decision"]
        if decision not in {"INCLUDE", "EXCLUDE"}:
            raise ValueError("subject choice review requires one membership decision")
        baseline_included = decision == "INCLUDE"
        selection_refs = tuple(
            dict.fromkeys(
                (
                    *(owner for owner in (review.selected_by_requirements if review is not None else ()) if applies(owner)),
                    *(
                        application.owner_ref
                        for application in explicit_requirement_applications
                        if application.value_ref == choice_ref
                        and application.owner_ref is not None
                    ),
                )
            )
        )
        explicit_override_applies = explicit_override_from_application_owners(
            application_owner_refs=selection_refs,
            boolean_requirement_refs=boolean_requirement_refs,
            baseline_included=baseline_included,
        )
        return SubjectChoiceReview(
            choice_ref=choice_ref,
            choice_domain_meaning=_text(declared["choice_domain_meaning"]),
            decision_basis=_text(declared["decision_basis"]),
            baseline_included=baseline_included,
            explicit_user_override_applies=explicit_override_applies,
            selection_requirement_refs=selection_refs,
        )

    choice_reviews = tuple(
        _review(choice_value, choice_applications.get(choice_value))
        for choice_value in expected_choices
    )
    expected_choice_refs = tuple(choice_refs_by_value.values())
    retained_reviews = (
        tuple(
            review
            for review in choice_reviews
            if review.choice_ref in explicitly_selected_choice_refs
        )
        if explicit_requirement_applications
        and surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
        else tuple(review for review in choice_reviews if review.included)
    )
    retained_values = tuple(review.choice_ref for review in retained_reviews)
    excluded_choices = tuple(
        review.choice_ref for review in choice_reviews if not review.included
    )
    applications: tuple[InvocationValueApplication, ...]
    mechanics: tuple[SourceMechanic, ...]
    if (
        excluded_choices
        and not retained_values
        and surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
    ):
        raise ValueError("subject parameter has no admitted retrieval choice")
    source_required_choice_refs = {
        application.value_ref
        for application in surface_applications
        if application.owner_ref is not None
        and application.owner_ref.startswith("source_required:")
    }
    if not source_required_choice_refs <= set(retained_values):
        raise ValueError("source-required choice selects an excluded subject state")
    if surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER:
        selected_values = tuple(
            request.source_catalog.choice_value(review.choice_ref).value
            for review in retained_reviews
        )
        source = request.source_catalog.source(surface.source_ref)
        param = next(
            item for item in source.params if item.param_ref == surface.target_ref
        )
        omittable = finite_choice_parameter_is_omittable(
            required=param.required,
            default=param.default,
            choices=param.choices,
            included_values=selected_values,
        )
        if surface_applications:
            applications = ()
        elif omittable:
            applications = ()
        else:
            applications = tuple(
                _subject_choice_application(
                    branch_id=branch_id,
                    source_ref=surface.source_ref,
                    target_ref=surface.target_ref,
                    value_ref=review.choice_ref,
                    owner_ref=None,
                    membership_owner_ref=owner_set_ref,
                    ordinal=ordinal,
                    mapping_basis=(
                        "The reviewed finite-choice parameter retains this "
                        "ordinary subject choice."
                    ),
                    request=request,
                )
                for ordinal, review in enumerate(
                    retained_reviews,
                    start=1,
                )
            )
        if excluded_choices or explicit_requirement_applications or applications:
            mechanics = (
                SourceMechanic(
                    mapping_basis=(
                        "The reviewed finite-choice parameter retains exactly the "
                        "choices inside the requested subject population."
                    ),
                    source_ref=surface.source_ref,
                    application_refs=tuple(
                        application.application_ref
                        for application in (surface_applications or applications)
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
        if excluded_choices or explicitly_selected_choice_refs:
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
                        *(
                            ref
                            for review in choice_reviews
                            for ref in review.selection_requirement_refs
                        ),
                    ),
                    kind=SourceMechanicKind.RETURNED_CHOICE_PREDICATE,
                ),
            )
        else:
            mechanics = ()
    return (
        SubjectSurfaceReview(
            owner_set_ref=owner_set_ref,
            surface_ref=surface_ref,
            surface_mapping_basis=_text(baseline["surface_mapping_basis"]),
            choice_reviews=choice_reviews,
            included_choice_refs=retained_values,
            mechanics=mechanics,
        ),
        applications,
    )


def _finite_choice_applications(
    values: dict[str, dict[str, output.FiniteChoiceApplicationOutput | None]],
    *,
    request: SemanticSourceBindingRequest,
) -> tuple[InvocationValueApplication, ...]:
    branch_ids = tuple(branch.branch_id for branch in request.strategy.branches)
    _exact_keys(
        values,
        expected=branch_ids,
        label="finite-choice application branch",
    )
    applications: list[InvocationValueApplication] = []
    for branch_id, owners in values.items():
        expected_options = {
            owner_ref: options
            for owner_ref in request.invocation_application_owner_refs
            if (
                options := request.finite_choice_options_for_owner(
                    owner_ref,
                    branch_id=branch_id,
                )
            )
        }
        _exact_keys(
            owners,
            expected=tuple(expected_options),
            label="finite-choice application owner",
        )
        for owner_ref, item in owners.items():
            if item is None:
                if owner_ref not in {
                    requirement.requirement_ref
                    for requirement in request.index.boolean_requirements
                }:
                    raise ValueError(
                        "source configuration requires a finite-choice application"
                    )
                continue
            options_by_surface = {
                surface.surface_ref: (surface, choices)
                for surface, choices in expected_options[owner_ref]
            }
            if item.surface_ref not in options_by_surface:
                raise ValueError(
                    "finite-choice application selects an incompatible surface"
                )
            surface, choices = options_by_surface[item.surface_ref]
            choice_refs_by_value = {
                choice.value: choice.value_ref for choice in choices
            }
            selected_values = tuple(item.selected_choice_values)
            if len(set(selected_values)) != len(selected_values):
                raise ValueError("finite-choice application repeats a choice")
            if not selected_values or any(
                value not in choice_refs_by_value for value in selected_values
            ):
                raise ValueError(
                    "finite-choice application selects an incompatible choice"
                )
            applications.extend(
                _subject_choice_application(
                    branch_id=branch_id,
                    source_ref=surface.source_ref,
                    target_ref=surface.target_ref,
                    value_ref=choice_refs_by_value[value],
                    owner_ref=owner_ref,
                    ordinal=ordinal,
                    mapping_basis=_text(item.application_basis),
                    request=request,
                )
                for ordinal, value in enumerate(selected_values, start=1)
            )
    return tuple(applications)


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
    membership_owner_ref: str | None = None,
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
    if membership_owner_ref is not None:
        owner_ref = f"membership:{membership_owner_ref}"
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
    set_ref: str,
) -> SetRealization:
    if item.rows_ref not in request.row_references_for_set(set_ref):
        raise ValueError("set realization references rows outside its declared options")
    identity = next(
        (
            value
            for value in request.source_catalog.identity_evidence
            if value.identity_ref == item.rows_ref
        ),
        None,
    )
    source_ref = identity.source_ref if identity is not None else item.rows_ref
    source = _branch_source(item.branch_id, source_ref, request=request)
    identity_ref = identity.identity_ref if identity is not None else None
    fields = identity.field_refs if identity is not None else ()
    return SetRealization(
        branch_id=item.branch_id,
        mapping_basis=_text(item.mapping_basis),
        source_ref=source.id,
        identity_ref=identity_ref,
        identity_field_refs=fields,
        contract_evidence_refs=_local_evidence(
            request.source_catalog.contract_snapshot.ref,
            source.id,
            identity_ref,
            *fields,
        ),
    )


def _fact_realization(
    item: output.ReturnedFactRealizationOutput,
    *,
    request: SemanticSourceBindingRequest,
) -> FactRealization:
    binding = request.source_catalog.field_binding(item.field_ref)
    source = _branch_source(item.branch_id, binding.source_ref, request=request)
    fields = (binding.field.field_ref,)
    return FactRealization(
        branch_id=item.branch_id,
        mapping_basis=_text(item.mapping_basis),
        source_ref=source.id,
        kind=FactRealizationKind.RETURNED_FIELD,
        identity_ref=None,
        field_refs=fields,
        contract_evidence_refs=_local_evidence(
            request.source_catalog.contract_snapshot.ref,
            source.id,
            *fields,
        ),
    )


def _identifier_fact_realizations(
    *,
    fact_ref: str,
    branch_ids: tuple[str, ...],
    set_bindings: dict[str, tuple[SetRealization, ...]],
    request: SemanticSourceBindingRequest,
) -> tuple[FactRealization, ...]:
    semantic_ref = next(
        ref
        for ref in request.index.inferred_type_by_ref
        if isinstance(ref, FactLocalRef) and ref.token == fact_ref
    )
    term = request.index.term_by_ref[semantic_ref]
    if not isinstance(term, FactTerm) or not isinstance(
        term.value_type, IdentifierType
    ):
        raise ValueError("derived identifier realization requires an Identifier fact")
    identified_set_ref = request.index.fact_local_ref_by_local_id[
        term.value_type.set_ref
    ].token
    realizations_by_branch = {
        item.branch_id: item for item in set_bindings[identified_set_ref]
    }
    if len(realizations_by_branch) != len(set_bindings[identified_set_ref]):
        raise ValueError("identified set repeats a strategy branch")
    output: list[FactRealization] = []
    for branch_id in branch_ids:
        set_realization = realizations_by_branch.get(branch_id)
        if set_realization is None or set_realization.identity_ref is None:
            raise ValueError(
                "identifier fact requires an identified-set identity contract"
            )
        output.append(
            FactRealization(
                branch_id=branch_id,
                mapping_basis=(
                    "The selected set identity realizes its identifier fact."
                ),
                source_ref=set_realization.source_ref,
                kind=FactRealizationKind.ENTITY_KEY,
                identity_ref=set_realization.identity_ref,
                field_refs=set_realization.identity_field_refs,
                contract_evidence_refs=set_realization.contract_evidence_refs,
            )
        )
    return tuple(output)


def _association_realization(
    item: output.AssociationRealizationOutput,
    *,
    request: SemanticSourceBindingRequest,
    association_ref: str,
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
    if relation is not None and relation.left_source_ref == relation.right_source_ref:
        association = request.index.term_by_ref[
            FactLocalRef.from_token(association_ref)
        ]
        assert isinstance(association, AssociationTerm)
        if item.reference_from_set_ref not in {
            request.index.fact_local_ref_by_local_id[ref].token
            for ref in (association.from_set_ref, association.to_set_ref)
        }:
            raise ValueError(
                "self-reference requires an explicit logical source-end role"
            )
    elif item.reference_from_set_ref is not None:
        raise ValueError("source-end role is only needed for self-references")
    return AssociationRealization(
        branch_id=item.branch_id,
        mapping_basis=_text(item.mapping_basis),
        kind=kind,
        source_refs=sources,
        relation_evidence_ref=relation_ref,
        reference_from_set_ref=item.reference_from_set_ref,
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
    scope_surface = request.raw_subject_scope_surfaces.get(owner_ref)
    if scope_surface is not None:
        if (
            option.source_ref != scope_surface.source_ref
            or option.target_ref != scope_surface.target_ref
        ):
            raise ValueError("subject-scope application targets another source surface")
        return
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
    set_bindings: dict[str, tuple[SetRealization, ...]],
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
            choice_mechanics = subject_binding.choice_mechanics(
                branch.branch_id, requirement.requirement_ref
            )
            mechanics = (
                (
                    *_invocation_mechanics(
                        owned, requirement_ref=requirement.requirement_ref
                    ),
                    *choice_mechanics,
                )
                if owned
                else choice_mechanics
                or _returned_mechanics(
                    branch.branch_id,
                    fact_refs=fact_refs,
                    fact_bindings=fact_bindings,
                    requirement_ref=requirement.requirement_ref,
                )
            )
            if not mechanics:
                mechanics = _relational_boolean_mechanics(
                    request=request,
                    branch_id=branch.branch_id,
                    requirement_ref=requirement.requirement_ref,
                    set_bindings=set_bindings,
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


def _relational_boolean_mechanics(
    *,
    request: SemanticSourceBindingRequest,
    branch_id: str,
    requirement_ref: str,
    set_bindings: dict[str, tuple[SetRealization, ...]],
) -> tuple[SourceMechanic, ...]:
    index = request.index
    ref = request._requirement_value_ref(requirement_ref)
    set_refs = set()
    domain = index.evaluation_domain_by_ref[ref]
    if isinstance(domain, RowDomain):
        set_refs.add(domain.owner_ref.token)
    for dependency in (ref, *index.transitive_dependencies_by_ref.get(ref, ())):
        if not isinstance(dependency, FactLocalRef):
            continue
        node = index.expression_by_ref.get(dependency)
        if isinstance(node, Quantify):
            set_refs.add(index.fact_local_ref_by_local_id[node.over_set_ref].token)
        elif isinstance(node, RelatedRow):
            set_refs.add(index.fact_local_ref_by_local_id[node.set_ref].token)
        elif isinstance(node, Coverage):
            set_refs.update(
                index.fact_local_ref_by_local_id[item].token
                for item in (
                    node.candidate_set_ref,
                    node.required_dimension_set_ref,
                    node.observation_set_ref,
                )
            )
    if not set_refs:
        set_refs.add(index.subject_obligation.subject_set_ref.token)
    realizations = tuple(
        value
        for ref in sorted(set_refs)
        for value in set_bindings.get(ref, ())
        if value.branch_id == branch_id
    )
    return tuple(
        SourceMechanic(
            f"Declared row domains evaluate {requirement_ref}.",
            source_ref,
            (),
            tuple(
                dict.fromkeys(
                    (
                        requirement_ref,
                        *(
                            evidence
                            for value in realizations
                            if value.source_ref == source_ref
                            for evidence in value.contract_evidence_refs
                        ),
                    )
                )
            ),
            SourceMechanicKind.RETURNED_ROW_PREDICATE,
        )
        for source_ref in dict.fromkeys(value.source_ref for value in realizations)
    )


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


__all__ = ["compile_source_binding_plan", "compile_source_realization"]


def empty_input_binding_payload(
    request: SemanticSourceBindingRequest,
) -> dict[str, object] | None:
    """Materialize the sole binding when there is no input decision to author."""
    if request.index.boolean_requirements:
        return None
    for branch in request.strategy.branches:
        for owner in request.invocation_application_owner_refs:
            if request.direct_value_options_for_owner(
                owner, branch_id=branch.branch_id
            ) or request.finite_choice_options_for_owner(
                owner, branch_id=branch.branch_id
            ):
                return None
    branches = tuple(branch.branch_id for branch in request.strategy.branches)
    return {
        "resolved_input_applications": {branch: [] for branch in branches},
        "finite_choice_applications": {branch: {} for branch in branches},
        "choice_requirement_applications": {branch: {} for branch in branches},
    }
