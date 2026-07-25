from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import ValidationError, validate

from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceContractSnapshot,
)
from fervis.lookup.plan_selection.semantic import (
    CandidateSourceStrategy,
    SourceAlignment,
    SourceAlignmentAssessment,
    SourceStrategyBranch,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    AllResults,
    BooleanComposition,
    BooleanCompositionOperator,
    FactTerm,
    InstanceInterpretation,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
)
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceKind,
    RowSourceParam,
    RowSourceValueType,
)
from fervis.lookup.semantic_types import (
    BooleanType,
    SourceOrigin,
    SourceOriginKind,
)
from fervis.lookup.source_binding.parser import compile_source_binding_plan
from fervis.lookup.source_binding.model import (
    SemanticSourceBindingRequest,
    SourceMechanicKind,
)
from fervis.lookup.source_binding.schema import (
    build_semantic_source_binding_schema,
)


def test_direct_boolean_requirement_owns_matching_truth_choice_application() -> None:
    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        "count of canceled sales",
    )
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(FactTerm("f1", "s1", BooleanType(), origin),),
        expressions=(
            Aggregate(
                id="e1",
                function=AggregateFunction.COUNT,
                argument_ref="s1",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref="f1",
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e1", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = RowSource(
        id="source_sales",
        kind=RowSourceKind.API_READ,
        label="sales",
        params=(
            RowSourceParam(
                id="is_canceled",
                param_ref="source_sales.is_canceled",
                name="is_canceled",
                type=RowSourceValueType.BOOLEAN,
                choices=("false", "true"),
            ),
        ),
    )
    branch = SourceStrategyBranch(
        branch_id="fact_1:source_branch:1",
        source_refs=(source.id,),
        relation_evidence_refs=(),
        qualification_clause_refs=tuple(
            clause.clause_ref for clause in index.qualification.clauses
        ),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=CandidateSourceStrategy(
            requested_fact_id=fact.id,
            source_assessments=(
                SourceAlignmentAssessment(
                    source_ref=source.id,
                    basis="The source returns sales and supports canceled filtering.",
                    alignment=SourceAlignment.DIRECT,
                ),
            ),
            strategy_basis="One direct source.",
            branches=(branch,),
        ),
        source_catalog=AvailableSourceCatalog(
            contract_snapshot=SourceContractSnapshot.from_content("{}"),
            sources=(source,),
            relation_evidence=(),
        ),
        canonical_values=(),
    )
    [requirement] = index.boolean_requirements
    set_ref = next(
        ref.token
        for ref in index.source_requirement_refs
        if ref.kind.value == "set"
    )
    payload = {
        "set_bindings": {
            set_ref: [
                {
                    "branch_id": branch.branch_id,
                    "mapping_basis": "Sale rows realize the requested sale set.",
                    "source_ref": source.id,
                    "identity_ref": None,
                }
            ]
        },
        "resolved_input_applications": {branch.branch_id: []},
        "finite_choice_applications": {
            branch.branch_id: {
                requirement.requirement_ref: {
                    "application_basis": (
                        "The true choice realizes the canceled-sale requirement."
                    ),
                    "surface_ref": "source_sales.is_canceled",
                    "selected_choice_values": ["true"],
                }
            }
        },
        "fact_bindings": {"fact_1:fact:f1": []},
        "association_bindings": {},
        "subject_binding": {
            "subject_ref": set_ref,
            "branch_realizations": [
                {
                    "branch_id": branch.branch_id,
                    "finite_choice_reviews": {
                        "source_sales.is_canceled": {
                            "surface_mapping_basis": (
                                "The parameter controls canceled sale membership."
                            ),
                            "choice_reviews": {
                                "false": {
                                    "choice_domain_meaning": (
                                        "The sale is not canceled."
                                    ),
                                    "role_match_basis": (
                                        "A non-canceled sale is an ordinary sale."
                                    ),
                                    "matched_excluded_role": "NONE",
                                    "choice_inclusion_basis": (
                                        "Ordinary sales include non-canceled sales."
                                    ),
                                    "choice_inclusion": "INCLUDE",
                                },
                                "true": {
                                    "choice_domain_meaning": "The sale is canceled.",
                                    "role_match_basis": (
                                        "Canceled matches CANCELED_OR_VOIDED."
                                    ),
                                    "matched_excluded_role": "CANCELED_OR_VOIDED",
                                    "choice_inclusion_basis": (
                                        "Canceled sales are outside the ordinary "
                                        "sale population."
                                    ),
                                    "choice_inclusion": "EXCLUDE",
                                },
                            },
                        }
                    },
                }
            ],
        },
    }

    validate(payload, build_semantic_source_binding_schema(request))
    plan = compile_source_binding_plan(payload, request=request)

    [application] = plan.invocation_applications
    assert application.owner_ref == requirement.requirement_ref
    assert request.source_catalog.choice_value(application.value_ref).value == "true"
    [surface_review] = plan.subject_binding.branch_realizations[0].surface_reviews
    canceled_review = surface_review.choice_reviews[1]
    assert canceled_review.explicit_user_override_applies
    assert canceled_review.included
    [realization] = plan.boolean_bindings[requirement.requirement_ref]
    [mechanic] = realization.mechanics
    assert mechanic.kind is SourceMechanicKind.INVOCATION_PREDICATE
    assert mechanic.application_refs == (application.application_ref,)

    negative_fact = replace(
        fact,
        expressions=(
            *fact.expressions,
            BooleanComposition(
                id="not_f1",
                operator=BooleanCompositionOperator.NOT,
                argument_refs=("f1",),
                origin=origin,
            ),
        ),
        qualification_ref="not_f1",
    )
    negative_index = analyze_requested_fact(
        negative_fact,
        inputs={},
        input_denotations={},
    )
    [negative_requirement] = negative_index.boolean_requirements
    negative_request = replace(request, index=negative_index)
    [surface] = negative_request.source_catalog.choice_surfaces
    false_choice, true_choice = surface.values
    assert negative_request.choice_value_requirement_refs(
        false_choice,
        branch_id=branch.branch_id,
    ) == (negative_requirement.requirement_ref,)
    assert (
        negative_request.choice_value_requirement_refs(
            true_choice,
            branch_id=branch.branch_id,
        )
        == ()
    )

    categorical_source = replace(
        source,
        params=(
            replace(
                source.params[0],
                type=RowSourceValueType.CHOICE,
                choices=("draft", "completed"),
            ),
        ),
    )
    categorical_request = replace(
        request,
        source_catalog=replace(
            request.source_catalog,
            sources=(categorical_source,),
        ),
    )
    [categorical_surface] = categorical_request.source_catalog.choice_surfaces
    categorical_requirement_refs = categorical_request.choice_requirement_refs(
        categorical_surface,
        branch_id=branch.branch_id,
    )
    assert all(
        categorical_request.choice_value_requirement_refs(
            choice,
            branch_id=branch.branch_id,
        )
        == categorical_requirement_refs
        for choice in categorical_surface.values
    )

    opposite_payload = deepcopy(payload)
    opposite_payload["finite_choice_applications"][branch.branch_id][
        requirement.requirement_ref
    ]["selected_choice_values"] = ["false"]

    with pytest.raises(ValidationError):
        validate(opposite_payload, build_semantic_source_binding_schema(request))
    with pytest.raises(
        ValueError,
        match="finite-choice application selects an incompatible choice",
    ):
        compile_source_binding_plan(opposite_payload, request=request)


def test_source_required_choice_cannot_override_an_excluded_subject_state() -> None:
    request, branch_id, set_ref = _source_required_choice_request(
        choices=("active", "deleted")
    )
    payload = _source_required_choice_payload(
        request,
        branch_id=branch_id,
        set_ref=set_ref,
        selected_choice="deleted",
        choice_reviews={
            "active": {
                "choice_domain_meaning": "Active rows are current sales.",
                "role_match_basis": "Active matches no excluded role.",
                "matched_excluded_role": "NONE",
                "choice_inclusion_basis": (
                    "Active sales belong to the ordinary population."
                ),
                "choice_inclusion": "INCLUDE",
            },
            "deleted": {
                "choice_domain_meaning": "Deleted rows are non-current sales.",
                "role_match_basis": "Deleted is a non-current artifact.",
                "matched_excluded_role": (
                    "SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT"
                ),
                "choice_inclusion_basis": (
                    "Deleted sales do not belong to the ordinary population."
                ),
                "choice_inclusion": "EXCLUDE",
            },
        },
    )

    validate(payload, build_semantic_source_binding_schema(request))
    with pytest.raises(
        ValueError,
        match="source-required choice selects an excluded subject state",
    ):
        compile_source_binding_plan(payload, request=request)


def test_source_required_presentation_choice_does_not_change_subject_membership() -> (
    None
):
    request, branch_id, set_ref = _source_required_choice_request(
        choices=("summary", "detailed")
    )
    payload = _source_required_choice_payload(
        request,
        branch_id=branch_id,
        set_ref=set_ref,
        selected_choice="summary",
        choice_reviews={
            choice: {
                "choice_domain_meaning": (
                    f"{choice.title()} changes response presentation only."
                ),
                "role_match_basis": "Presentation matches no excluded role.",
                "matched_excluded_role": "NONE",
                "choice_inclusion_basis": (
                    "Presentation does not alter subject membership."
                ),
                "choice_inclusion": "INCLUDE",
            }
            for choice in ("summary", "detailed")
        },
    )

    validate(payload, build_semantic_source_binding_schema(request))
    plan = compile_source_binding_plan(payload, request=request)

    [surface] = plan.subject_binding.branch_realizations[0].surface_reviews
    assert not any(review.explicit_user_override_applies for review in surface.choice_reviews)
    assert surface.included_choice_refs == tuple(
        value.value_ref for value in request.source_catalog.choice_values
    )
    assert surface.mechanics == ()


def _source_required_choice_request(
    *,
    choices: tuple[str, ...],
) -> tuple[SemanticSourceBindingRequest, str, str]:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "sale occurrences")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(),
        expressions=(
            Aggregate(
                id="e1",
                function=AggregateFunction.COUNT,
                argument_ref="s1",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e1", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = RowSource(
        id="source_sales",
        kind=RowSourceKind.API_READ,
        label="sales",
        params=(
            RowSourceParam(
                id="mode",
                param_ref="source_sales.mode",
                name="mode",
                type=RowSourceValueType.CHOICE,
                required=True,
                choices=choices,
            ),
        ),
    )
    branch_id = "fact_1:source_branch:1"
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=CandidateSourceStrategy(
            requested_fact_id=fact.id,
            source_assessments=(
                SourceAlignmentAssessment(
                    source_ref=source.id,
                    basis="The source returns sale occurrences.",
                    alignment=SourceAlignment.DIRECT,
                ),
            ),
            strategy_basis="One direct source.",
            branches=(
                SourceStrategyBranch(
                    branch_id=branch_id,
                    source_refs=(source.id,),
                    relation_evidence_refs=(),
                    qualification_clause_refs=(),
                ),
            ),
        ),
        source_catalog=AvailableSourceCatalog(
            contract_snapshot=SourceContractSnapshot.from_content("{}"),
            sources=(source,),
            relation_evidence=(),
        ),
        canonical_values=(),
    )
    set_ref = next(
        ref.token
        for ref in index.source_requirement_refs
        if ref.kind.value == "set"
    )
    return request, branch_id, set_ref


def _source_required_choice_payload(
    request: SemanticSourceBindingRequest,
    *,
    branch_id: str,
    set_ref: str,
    selected_choice: str,
    choice_reviews: dict[str, dict[str, object]],
) -> dict[str, object]:
    [source] = request.source_catalog.sources
    [surface] = request.source_catalog.choice_surfaces
    owner_ref = f"source_required:{surface.surface_ref}"
    return {
        "set_bindings": {
            set_ref: [
                {
                    "branch_id": branch_id,
                    "mapping_basis": "Sale rows realize the requested set.",
                    "source_ref": source.id,
                    "identity_ref": None,
                }
            ]
        },
        "resolved_input_applications": {branch_id: []},
        "finite_choice_applications": {
            branch_id: {
                owner_ref: {
                    "application_basis": (
                        "The selected choice supplies the required parameter."
                    ),
                    "surface_ref": surface.surface_ref,
                    "selected_choice_values": [selected_choice],
                }
            }
        },
        "fact_bindings": {},
        "association_bindings": {},
        "subject_binding": {
            "subject_ref": set_ref,
            "branch_realizations": [
                {
                    "branch_id": branch_id,
                    "finite_choice_reviews": {
                        surface.surface_ref: {
                            "surface_mapping_basis": (
                                "Review whether this surface changes membership."
                            ),
                            "choice_reviews": choice_reviews,
                        }
                    },
                }
            ],
        },
    }
