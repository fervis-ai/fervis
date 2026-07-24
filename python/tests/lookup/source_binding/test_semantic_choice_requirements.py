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
from fervis.lookup.question_contract.semantic_analysis import analyze_requested_fact
from fervis.lookup.question_contract.semantic_model import (
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
from fervis.lookup.source_binding.binding_plan_compilation import (
    compile_source_binding_plan,
)
from fervis.lookup.source_binding.semantic import (
    SemanticSourceBindingRequest,
    SourceMechanicKind,
)
from fervis.lookup.source_binding.semantic_schema import (
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
                                    "requirement_mapping_basis": None,
                                    "requirement_ref": None,
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
                                    "requirement_mapping_basis": (
                                        "The question explicitly requests canceled "
                                        "sales."
                                    ),
                                    "requirement_ref": requirement.requirement_ref,
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
    choice_reviews = opposite_payload["subject_binding"]["branch_realizations"][0][
        "finite_choice_reviews"
    ]["source_sales.is_canceled"]["choice_reviews"]
    choice_reviews["false"]["requirement_mapping_basis"] = (
        "This choice contradicts the positive canceled condition."
    )
    choice_reviews["false"]["requirement_ref"] = requirement.requirement_ref
    choice_reviews["true"]["requirement_mapping_basis"] = None
    choice_reviews["true"]["requirement_ref"] = None

    with pytest.raises(ValidationError):
        validate(opposite_payload, build_semantic_source_binding_schema(request))
    with pytest.raises(
        ValueError,
        match="choice review selects an incompatible Boolean requirement",
    ):
        compile_source_binding_plan(opposite_payload, request=request)
