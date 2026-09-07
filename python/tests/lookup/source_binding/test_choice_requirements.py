from copy import deepcopy
from dataclasses import replace
from fervis.lookup.source_binding.invocation_bindings import invocation_value

import pytest
from jsonschema import ValidationError

from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceContractSnapshot,
)
from fervis.lookup.source_binding.model import (
    CandidateSourceStrategy,
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
from tests.lookup.source_binding._fixtures import (
    compile_binding_fixture,
    validate_binding_fixture,
)
from fervis.lookup.source_binding.model import (
    SemanticSourceBindingRequest,
    SourceMechanicKind,
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
        subject=Subject("s1", InstanceInterpretation.RESOURCE_POPULATION),
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
        strategy=CandidateSourceStrategy(requested_fact_id=fact.id, branches=(branch,)),
        source_catalog=AvailableSourceCatalog(
            contract_snapshot=SourceContractSnapshot.from_content("{}"),
            sources=(source,),
            relation_evidence=(),
        ),
        canonical_values=(),
    )
    [requirement] = index.boolean_requirements
    set_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "set"
    )
    payload = {
        "set_bindings": {
            set_ref: [
                {
                    "branch_id": branch.branch_id,
                    "mapping_basis": "Sale rows realize the requested sale set.",
                    "rows_ref": source.id,
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
                    "surface_ref": "source_surface:source_sales:parameter:is_canceled",
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
                        "source_surface:source_sales:parameter:is_canceled": {
                            "surface_mapping_basis": (
                                "The parameter controls canceled sale membership."
                            ),
                            "choice_reviews": {
                                "false": {
                                    "selected_by_requirements": [],
                                    "choice_domain_meaning": "The sale is not canceled.",
                                    "decision_basis": "Ordinary sales include non-canceled sales.",
                                    "baseline_decision": "INCLUDE",
                                },
                                "true": {
                                    "selected_by_requirements": [],
                                    "choice_domain_meaning": "The sale is canceled.",
                                    "decision_basis": "Canceled sales are outside the ordinary "
                                    "sale population.",
                                    "baseline_decision": "EXCLUDE",
                                },
                            },
                        }
                    },
                }
            ],
        },
    }

    validate_binding_fixture(payload, request=request)
    plan = compile_binding_fixture(payload, request=request)

    [application] = plan.invocation_applications
    assert application.owner_ref == requirement.requirement_ref
    assert request.source_catalog.choice_value(application.value_ref).value == "true"
    assert plan.subject_binding.branch_realizations[0].surface_reviews == ()
    assert application.owner_ref == requirement.requirement_ref
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
    # Polarity is part of the declared semantic mapping, not inferred from flag truth.
    assert negative_request.choice_value_requirement_refs(
        true_choice,
        branch_id=branch.branch_id,
    ) == (negative_requirement.requirement_ref,)

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

    unknown_choice_payload = deepcopy(payload)
    unknown_choice_payload["finite_choice_applications"][branch.branch_id][
        requirement.requirement_ref
    ]["selected_choice_values"] = ["unknown_choice"]

    with pytest.raises(ValidationError):
        validate_binding_fixture(unknown_choice_payload, request=request)
    with pytest.raises(
        ValueError,
        match="finite-choice application selects an incompatible choice",
    ):
        compile_binding_fixture(unknown_choice_payload, request=request)


@pytest.mark.parametrize("interpretation", list(InstanceInterpretation))
def test_population_owns_overrides_for_defaults_that_can_hide_records(interpretation):
    request, branch_id, set_ref = _source_required_choice_request(
        choices=("active", "all_persisted")
    )
    [source] = request.source_catalog.sources
    [param] = source.params
    from fervis.host_api.contracts.population import ParameterPopulation

    source = replace(
        source,
        params=(
            replace(
                param,
                required=False,
                default="active",
                population=ParameterPopulation(unfiltered_values=("all_persisted",)),
            ),
        ),
    )
    request = replace(
        request, source_catalog=replace(request.source_catalog, sources=(source,))
    )
    fact = replace(
        request.index.requested_fact,
        subject=replace(
            request.index.requested_fact.subject,
            instance_interpretation=interpretation,
        ),
    )
    request = replace(
        request, index=analyze_requested_fact(fact, inputs={}, input_denotations={})
    )
    assert not any(
        ref.startswith("subject_scope:")
        for ref in request.invocation_application_owner_refs
    )
    payload = _source_required_choice_payload(
        request,
        branch_id=branch_id,
        set_ref=set_ref,
        selected_choice="all_persisted",
        choice_reviews={},
    )
    payload["finite_choice_applications"][branch_id] = {}
    payload["subject_binding"]["branch_realizations"][0]["finite_choice_reviews"] = {}
    validate_binding_fixture(payload, request=request)
    plan = compile_binding_fixture(payload, request=request)
    assert {
        invocation_value(request, applied.value_ref).payload.value
        for applied in plan.invocation_applications
    } == {"all_persisted"}
    assert all(
        applied.target_applications[0].target_ref == param.param_ref
        for applied in plan.invocation_applications
    )


def test_source_required_presentation_choice_does_not_change_subject_membership() -> (
    None
):
    request, branch_id, set_ref = _source_required_choice_request(
        choices=("summary", "detailed")
    )
    from fervis.host_api.contracts import ParameterSemantics

    (source,) = request.source_catalog.sources
    request = replace(
        request,
        source_catalog=replace(
            request.source_catalog,
            sources=(
                replace(
                    source,
                    params=tuple(
                        replace(param, semantics=ParameterSemantics.RESPONSE_SHAPE)
                        for param in source.params
                    ),
                ),
            ),
        ),
    )
    payload = _source_required_choice_payload(
        request,
        branch_id=branch_id,
        set_ref=set_ref,
        selected_choice="summary",
        choice_reviews={
            choice: {
                "selected_by_requirements": [],
                "choice_domain_meaning": f"{choice.title()} changes response presentation only.",
                "decision_basis": "Presentation does not alter subject membership.",
                "baseline_decision": "INCLUDE",
            }
            for choice in ("summary", "detailed")
        },
    )

    validate_binding_fixture(payload, request=request)
    plan = compile_binding_fixture(payload, request=request)

    assert plan.subject_binding.branch_realizations[0].surface_reviews == ()


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
        subject=Subject("s1", InstanceInterpretation.RESOURCE_POPULATION),
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
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "set"
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
    owner_ref = f"source_required:{surface.target_ref}"
    return {
        "set_bindings": {
            set_ref: [
                {
                    "branch_id": branch_id,
                    "mapping_basis": "Sale rows realize the requested set.",
                    "rows_ref": source.id,
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
