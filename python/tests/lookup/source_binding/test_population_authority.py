"""Unqualified API resources cannot acquire a global lifecycle exclusion."""

from dataclasses import replace
from fervis.lookup.source_binding.invocation_bindings import invocation_value
import pytest
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from tests.lookup.source_binding.test_choice_requirements import (
    _source_required_choice_request,
)


@pytest.mark.parametrize(
    ("resource", "states"),
    [
        ("flow runs", ("COMPLETED", "CANCELLED", "SCHEDULED")),
        ("devices", ("active", "planned", "failed")),
    ],
)
def test_unowned_lifecycle_classification_cannot_narrow_a_resource(resource, states):
    request, branch, set_ref = _source_required_choice_request(choices=states)
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, f"count of {resource}")
    fact = request.index.requested_fact
    fact = replace(
        fact, origin=origin, sets=tuple(replace(s, origin=origin) for s in fact.sets)
    )
    (source,) = request.source_catalog.sources
    source = replace(
        source,
        label=resource,
        params=tuple(replace(p, required=False) for p in source.params),
    )
    request = replace(
        request,
        index=analyze_requested_fact(fact, inputs={}, input_denotations={}),
        source_catalog=replace(request.source_catalog, sources=(source,)),
    )
    from fervis.lookup.source_binding.parser import (
        compile_source_realization,
        compile_source_binding_plan,
    )
    from fervis.lookup.source_binding.verification import (
        verify_source_strategy,
        VerifiedSourceStrategy,
    )
    from fervis.lookup.source_binding.schema import build_semantic_source_binding_schema
    from jsonschema import validate, ValidationError

    realization = compile_source_realization(
        {
            "set_bindings": {
                set_ref: [
                    {
                        "branch_id": branch,
                        "mapping_basis": resource,
                        "rows_ref": source.id,
                    }
                ]
            },
            "fact_bindings": {},
            "association_bindings": {},
        },
        request=request,
    )
    payload = {
        "resolved_input_applications": {branch: []},
        "finite_choice_applications": {branch: {}},
        "choice_requirement_applications": {branch: {}},
    }
    validate(payload, build_semantic_source_binding_schema(realization.request))
    plan = compile_source_binding_plan(payload, realization=realization)
    assert plan.invocation_applications == ()
    assert plan.subject_binding.branch_realizations[0].surface_reviews == ()
    assert isinstance(
        verify_source_strategy(plan, request=realization.request),
        VerifiedSourceStrategy,
    )
    (surface,) = request.source_catalog.choice_surfaces
    payload["choice_requirement_applications"][branch][surface.surface_ref] = {
        state: {
            "mapping_basis": "Implicit lifecycle classification",
            "selected_by_requirements": [],
        }
        for state in states
    }
    with pytest.raises(ValidationError):
        validate(payload, build_semantic_source_binding_schema(realization.request))
    with pytest.raises(ValueError):
        compile_source_binding_plan(payload, realization=realization)


def test_nonfinite_default_filter_requires_population_coverage():
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query
    from fervis.lookup.relation_catalog.row_sources.model import (
        RowSourceParam,
        RowSourceValueType,
    )
    from fervis.lookup.source_binding.verification import (
        verify_source_strategy,
        SourceStrategyVerificationFailure,
    )

    original = employee_query()
    (source,) = original.request.source_catalog.sources
    source = replace(
        source,
        params=(
            RowSourceParam(
                "min_salary",
                "param.min_salary",
                "Minimum salary",
                RowSourceValueType.DECIMAL,
                default="150",
                description="Return only employees whose salary is at least this amount.",
            ),
        ),
    )
    request = replace(
        original.request,
        source_catalog=replace(original.request.source_catalog, sources=(source,)),
    )
    result = verify_source_strategy(original.binding_plan, request=request)
    assert isinstance(result, SourceStrategyVerificationFailure)


def test_finite_filter_coverage_cannot_be_replaced_by_a_subset():
    from fervis.lookup.source_binding.parser import (
        compile_source_realization,
        compile_source_binding_plan,
    )
    from fervis.lookup.source_binding.verification import (
        verify_source_strategy,
        VerifiedSourceStrategy,
        SourceStrategyVerificationFailure,
        SourceStrategyVerificationFailureReason,
    )

    request, branch, set_ref = _source_required_choice_request(
        choices=("active", "inactive")
    )
    (source,) = request.source_catalog.sources
    from fervis.host_api.contracts.population import (
        ParameterPopulation,
        ParameterRowValues,
    )
    from fervis.lookup.relation_catalog.row_sources.model import (
        RowSourceField,
        RowSourceValueType,
        RowSourceCandidateKey,
        RowSourceKeyComponent,
    )
    from fervis.lookup.answer_program.relations import FieldBindingRole

    source = replace(
        source,
        fields=(
            RowSourceField(
                "id",
                "field.id",
                "ID",
                RowSourceValueType.STRING,
                (FieldBindingRole.IDENTITY,),
            ),
            RowSourceField(
                "state",
                "field.state",
                "State",
                RowSourceValueType.CHOICE,
                (),
                choices=("active", "inactive"),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                "pk", "record", (RowSourceKeyComponent("id", "id"),), primary=True
            ),
        ),
        params=(
            replace(
                source.params[0],
                population=ParameterPopulation(
                    field_path="field.state",
                    value_mapping=(
                        ParameterRowValues("active", ("active",)),
                        ParameterRowValues("inactive", ("inactive",)),
                    ),
                ),
            ),
        ),
    )
    request = replace(
        request, source_catalog=replace(request.source_catalog, sources=(source,))
    )
    realization = compile_source_realization(
        {
            "set_bindings": {
                set_ref: [
                    {
                        "branch_id": branch,
                        "mapping_basis": "Declared resource",
                        "rows_ref": request.row_references_for_set(set_ref)[0],
                    }
                ]
            },
            "fact_bindings": {},
            "association_bindings": {},
        },
        request=request,
    )
    plan = compile_source_binding_plan(
        {
            "resolved_input_applications": {branch: []},
            "finite_choice_applications": {branch: {}},
            "choice_requirement_applications": {branch: {}},
        },
        realization=realization,
    )
    assert {
        invocation_value(request, app.value_ref).payload.value
        for app in plan.invocation_applications
    } == {"active", "inactive"}
    assert isinstance(
        verify_source_strategy(plan, request=realization.request),
        VerifiedSourceStrategy,
    )
    narrowed = replace(plan, invocation_applications=plan.invocation_applications[:1])
    result = verify_source_strategy(narrowed, request=realization.request)
    assert isinstance(result, SourceStrategyVerificationFailure)
    assert (
        result.reason
        is SourceStrategyVerificationFailureReason.INSUFFICIENT_COMPLETENESS
    )
