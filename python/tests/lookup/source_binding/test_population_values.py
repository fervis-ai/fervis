"""Coverage requires a source predicate over a declared row domain."""

from dataclasses import replace
import pytest

from fervis.host_api.contracts.population import ParameterPopulation, ParameterRowValues
from fervis.lookup.relation_catalog.row_sources.model import (
    RowSourceField,
    RowSourceParam,
    RowSourceValueType,
)
from fervis.lookup.source_binding.population_values import (
    population_values,
    covers_population,
)
from tests.lookup.source_binding.test_choice_requirements import (
    _source_required_choice_request,
)


def _source():
    request, _, _ = _source_required_choice_request(choices=("false", "true"))
    source = request.source_catalog.sources[0]
    field = RowSourceField(
        "active", "field.active", "Activity", RowSourceValueType.BOOLEAN, ()
    )
    param = replace(
        source.params[0], source="query",
        population=ParameterPopulation(
            field_path="field.active",
            value_mapping=(
                ParameterRowValues("false", ("false",)),
                ParameterRowValues("true", ("true",)),
            ),
        ),
    )
    return replace(source, fields=(field,), params=(param,)), param


def test_partition_must_cover_the_returned_domain_not_only_legal_requests():
    source, param = _source()
    assert population_values(source, param) == ("false", "true")
    assert covers_population(source, param, ("false", "true"))
    assert not covers_population(source, param, ("true",))
    assert population_values(source, replace(param, population=None)) == ()


@pytest.mark.parametrize(
    "change", [dict(nullable=True), dict(declared_value_domain=False)]
)
def test_nullable_or_observed_domains_cannot_prove_exhaustion(change):
    source, param = _source()
    source = replace(source, fields=(replace(source.fields[0], **change),))
    assert population_values(source, param) == ()
    assert not covers_population(source, param, ("false", "true"))


def test_finite_threshold_arguments_do_not_exhaust_numeric_rows():
    source, param = _source()
    field = replace(source.fields[0], type=RowSourceValueType.DECIMAL)
    param = replace(
        param,
        choices=("100", "200"),
        population=ParameterPopulation(
            field_path="field.active",
            value_mapping=(
                ParameterRowValues("100", ("100",)),
                ParameterRowValues("200", ("200",)),
            ),
        ),
    )
    assert population_values(replace(source, fields=(field,)), param) == ()


def test_an_unfiltered_argument_does_not_require_a_finite_returned_domain():
    source, param = _source()
    param = replace(
        param,
        choices=("active", "all_persisted"),
        population=ParameterPopulation(unfiltered_values=("all_persisted",)),
    )
    assert population_values(replace(source, fields=()), param) == ("all_persisted",)
    assert covers_population(source, param, ("all_persisted",))
    assert not covers_population(source, param, ("active",))


def test_population_contract_cannot_fabricate_a_request_value():
    source, param = _source()
    param = replace(param, population=ParameterPopulation(unfiltered_values=("all",)))
    with pytest.raises(ValueError, match="undeclared argument"):
        population_values(source, param)


def test_explicit_multivalue_predicate_requires_executable_union_identity():
    from fervis.lookup.question_contract.model import FactTerm
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.semantic_types import BooleanType
    from fervis.lookup.source_binding.parser import (
        compile_source_realization,
        compile_source_binding_plan,
    )
    from fervis.lookup.source_binding.verification import (
        verify_source_strategy,
        SourceStrategyVerificationFailure,
    )

    request, branch, set_ref = _source_required_choice_request(
        choices=("draft", "completed")
    )
    fact = request.index.requested_fact
    fact = replace(
        fact,
        facts=(FactTerm("selected_state", "s1", BooleanType(), fact.origin),),
        qualification_ref="selected_state",
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    request = replace(
        request,
        index=index,
        strategy=replace(
            request.strategy,
            branches=(
                replace(
                    request.strategy.branches[0],
                    qualification_clause_refs=tuple(
                        c.clause_ref for c in index.qualification.clauses
                    ),
                ),
            ),
        ),
    )
    realization = compile_source_realization(
        {
            "set_bindings": {
                set_ref: [
                    {
                        "branch_id": branch,
                        "mapping_basis": "Requested rows",
                        "rows_ref": request.source_catalog.sources[0].id,
                    }
                ]
            },
            "fact_bindings": {"fact_1:fact:selected_state": []},
            "association_bindings": {},
        },
        request=request,
    )
    owner = index.boolean_requirements[0].requirement_ref
    surface = realization.request.source_catalog.choice_surfaces[0]
    plan = compile_source_binding_plan(
        {
            "resolved_input_applications": {branch: []},
            "choice_requirement_applications": {branch: {}},
            "finite_choice_applications": {
                branch: {
                    owner: {
                        "application_basis": "Both states satisfy the explicit condition",
                        "surface_ref": surface.surface_ref,
                        "selected_choice_values": ["draft", "completed"],
                    }
                }
            },
        },
        realization=realization,
    )
    result = verify_source_strategy(plan, request=realization.request)
    assert isinstance(result, SourceStrategyVerificationFailure)
    assert "invocation_union_identity:source_sales" in result.failed_requirement_refs


def test_declared_population_and_nullability_survive_api_response_normalization():
    from fervis.host_api.contracts import (
        EndpointContract,
        ParameterContract,
        ResponseFieldContract,
        PaginationContract,
        PaginationKind,
    )
    from fervis.lookup.relation_catalog.from_host_api import (
        relation_catalog_from_endpoint_contracts,
    )
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog

    effect = ParameterPopulation(
        field_path="results.active",
        value_mapping=(
            ParameterRowValues("false", ("false",)),
            ParameterRowValues("true", ("true",)),
        ),
    )
    contract = EndpointContract(
        "items",
        "items",
        "GET",
        "/items",
        "",
        "items",
        resource_names=("items",),
        query_params=(
            ParameterContract("active", "boolean", default=True, population=effect),
        ),
        response_fields=(
            ResponseFieldContract("results", "array", "results", nullable=False),
            ResponseFieldContract(
                "active", "boolean", "results.active", nullable=False
            ),
        ),
        pagination=PaginationContract(
            kind=PaginationKind.OFFSET,
            position_query_param="offset",
            page_size_query_param="limit",
            results_path="results",
            total_path="total",
            page_size=10,
            max_page_size=100,
        ),
    )
    catalog = relation_catalog_from_endpoint_contracts((contract,))
    source = next(
        s for s in build_api_row_source_catalog(catalog).sources if s.row_path == "data"
    )
    assert population_values(source, source.params[0]) == ("false", "true")
    assert source.params[0].population.field_path == "data.active"
    assert source.fields[0].nullable is False
    assert (
        contract.query_params[0].to_public_dict()["population"]["fieldPath"]
        == "results.active"
    )


@pytest.mark.parametrize('type_name,argument', [(RowSourceValueType.INTEGER, '0'), (RowSourceValueType.STRING, 'all'), (RowSourceValueType.DECIMAL, '0.5')])
def test_documented_unfiltered_arguments_do_not_require_an_enum(type_name, argument):
    source, param = _source()
    param = replace(param, type=type_name, choices=(), default=10,
                    population=ParameterPopulation(unfiltered_values=(argument,)))
    assert population_values(source, param) == (argument,)
    assert covers_population(source, param, (argument,))


def test_open_integer_control_rejects_non_integer_argument():
    source, param = _source()
    param = replace(param, type=RowSourceValueType.INTEGER, choices=(),
                    population=ParameterPopulation(unfiltered_values=('all',)))
    with pytest.raises(ValueError):
        population_values(source, param)


def test_closed_row_domain_can_be_covered_by_documented_open_argument_values():
    source, param = _source()
    param = replace(param, type=RowSourceValueType.STRING, choices=(),
                    population=ParameterPopulation(field_path='field.active', value_mapping=(
                        ParameterRowValues('enabled', ('true',)),
                        ParameterRowValues('disabled', ('false',)),
                    )))
    assert population_values(source, param) == ('enabled', 'disabled')
    assert covers_population(source, param, ('enabled', 'disabled'))
    assert not covers_population(source, param, ('enabled',))
    assert population_values(replace(source, fields=(replace(source.fields[0], nullable=True),)), param) == ()


def test_typed_argument_aliases_cannot_claim_different_row_populations():
    source, param = _source()
    param = replace(param, type=RowSourceValueType.DECIMAL, choices=(),
        population=ParameterPopulation(field_path='field.active', value_mapping=(
            ParameterRowValues('1', ('true',)), ParameterRowValues('1.0', ('false',)),
        )))
    with pytest.raises(ValueError, match='same typed argument'):
        population_values(source, param)
