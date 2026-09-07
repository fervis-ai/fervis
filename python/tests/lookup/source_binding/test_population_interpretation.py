"""Source meaning is separate from question intent and declared API authority."""

from dataclasses import replace
import pytest
from jsonschema import validate, ValidationError

from fervis.lookup.source_binding.population_interpretation import (
    SourcePopulationTurnPrompt, population_interpretation_schema,
    population_interpretation_targets, parse_population_interpretations,
)
from fervis.lookup.source_binding.population_values import population_values
from fervis.lookup.turn_prompts import TurnPromptContext
from tests.lookup.source_binding.test_population_values import _source
from tests.lookup.source_binding.test_choice_requirements import _source_required_choice_request


def _request():
    request, _, _ = _source_required_choice_request(choices=('false', 'true'))
    source, param = _source()
    source = replace(source, params=(replace(param, population=None, default='true',
        description='Filter returned rows to those whose active flag equals this value.'),))
    return replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))


def _payload(request):
    key, = population_interpretation_targets(request)
    return {'parameters': {key: {'kind': 'finite_domain_cover', 'mapping_basis': 'Declared equality filter on returned active flag',
        'field_ref': 'field.active', 'arguments': [{'argument': 'true', 'row_values': ['true']}, {'argument': 'false', 'row_values': ['false']}]}}}


def test_interpretation_preserves_declared_catalog_and_ignores_question_text():
    request = _request()
    payload = _payload(request)
    validate(payload, population_interpretation_schema(request, finite_cover=True))
    interpreted = replace(request, population_interpretations=parse_population_interpretations(payload, request=request, finite_cover=True))
    source = request.source_catalog.sources[0]
    param = source.params[0]
    assert param.population is None
    assert interpreted.source_catalog is request.source_catalog
    effective = replace(param, population=interpreted.parameter_population(source.id, param.param_ref))
    assert population_values(source, effective) == ('false', 'true')
    assert population_interpretation_targets(interpreted) == {}
    prompt = SourcePopulationTurnPrompt(request).to_model_payload(TurnPromptContext(current_question='private question sentinel'))
    assert 'private question sentinel' not in prompt.prompt_text


@pytest.mark.parametrize('change', [dict(nullable=True), dict(declared_value_domain=False)])
def test_inferred_categorical_coverage_requires_declared_nonnullable_domain(change):
    request = _request()
    source = request.source_catalog.sources[0]
    source = replace(source, fields=(replace(source.fields[0], **change),))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    payload = _payload(request)
    with pytest.raises(ValidationError):
        validate(payload, population_interpretation_schema(request, finite_cover=True))
    with pytest.raises(ValueError, match='exact source parameter scope'):
        parse_population_interpretations(payload, request=request, finite_cover=True)


def test_unknown_meaning_remains_unknown_instead_of_an_exhaustiveness_claim():
    request = _request()
    key, = population_interpretation_targets(request)
    payload = {'parameters': {key: {'kind': 'unknown', 'mapping_basis': 'Row admission is undocumented'}}}
    interpreted = replace(request, population_interpretations=parse_population_interpretations(payload, request=request, finite_cover=True))
    source = interpreted.source_catalog.sources[0]
    assert interpreted.parameter_population(source.id, source.params[0].param_ref) is None
    assert population_interpretation_targets(interpreted) == {}


@pytest.mark.parametrize('declared', [False, True])
def test_required_nonfiltering_control_is_bindable_without_becoming_a_predicate(declared):
    from fervis.host_api.contracts.population import ParameterPopulation
    from fervis.lookup.source_binding.parser import compile_source_realization, compile_source_binding_plan
    from fervis.lookup.source_binding.verification import verify_source_strategy, VerifiedSourceStrategy
    request = _request()
    source = request.source_catalog.sources[0]
    param = replace(source.params[0], required=True, default=None, description='Ordering direction only; no rows are removed',
                    population=ParameterPopulation(preserves_population=True) if declared else None)
    source = replace(source, fields=(), params=(param,))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    if not declared:
        key, = population_interpretation_targets(request)
        request = replace(request, population_interpretations=parse_population_interpretations({'parameters': {
            key: {'kind': 'preserves_all_rows_for_every_argument', 'mapping_basis': 'Only ordering changes'}
        }}, request=request))
    assert not request._parameter_controls_rows(source.id, param.param_ref)
    owner, = request.invocation_application_owner_refs
    assert owner == f'source_required:{param.param_ref}'
    branch = request.strategy.branches[0].branch_id
    realized = compile_source_realization({
        'set_bindings': {request.index.subject_obligation.subject_set_ref.token: [
            { 'branch_id': branch, 'mapping_basis': 'Requested rows', 'rows_ref': source.id}]},
        'fact_bindings': {}, 'association_bindings': {},
    }, request=request)
    surface = request.source_catalog.choice_surfaces[0]
    plan = compile_source_binding_plan({'resolved_input_applications': {branch: []},
        'finite_choice_applications': {branch: {owner: {'application_basis': 'Supply the required presentation control',
            'surface_ref': surface.surface_ref, 'selected_choice_values': ['true']}}},
        'choice_requirement_applications': {branch: {}}}, realization=realized)
    assert isinstance(verify_source_strategy(plan, request=realized.request), VerifiedSourceStrategy)
    assert plan.boolean_bindings == {}
    from fervis.lookup.question_contract.model import FactTerm
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.semantic_types import BooleanType
    fact = request.index.requested_fact
    index = analyze_requested_fact(replace(fact, facts=(FactTerm('state', 's1', BooleanType(), fact.origin),),
        qualification_ref='state'), inputs={}, input_denotations={})
    qualified = replace(request, index=index, strategy=replace(request.strategy, branches=(replace(
        request.strategy.branches[0], qualification_clause_refs=tuple(c.clause_ref for c in index.qualification.clauses)),)))
    requirement = index.boolean_requirements[0]
    assert qualified.finite_choice_options_for_owner(requirement.requirement_ref, branch_id=branch) == ()


def test_partial_argument_knowledge_does_not_fabricate_a_field_mapping():
    request = _request()
    payload = _payload(request)
    item = next(iter(payload['parameters'].values()))
    item['arguments'] = [{'argument': 'true', 'row_values': ['false', 'true']}]
    validate(payload, population_interpretation_schema(request, finite_cover=True))
    interpreted = replace(request, population_interpretations=parse_population_interpretations(payload, request=request, finite_cover=True))
    source = request.source_catalog.sources[0]
    param = source.params[0]
    effective = replace(param, population=interpreted.parameter_population(source.id, param.param_ref))
    assert population_values(source, effective) == ('true',)
    assert [value.argument for value in effective.population.value_mapping] == ['true']


def test_unfiltered_argument_cannot_prove_a_row_state_predicate():
    from fervis.host_api.contracts.population import ParameterPopulation
    from fervis.lookup.question_contract.model import FactTerm
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.semantic_types import BooleanType
    request = _request()
    source = request.source_catalog.sources[0]
    source = replace(source, params=(replace(source.params[0], population=ParameterPopulation(unfiltered_values=('true',))),))
    fact = request.index.requested_fact
    index = analyze_requested_fact(replace(fact, facts=(FactTerm('state', 's1', BooleanType(), fact.origin),),
        qualification_ref='state'), inputs={}, input_denotations={})
    branch = replace(request.strategy.branches[0], qualification_clause_refs=tuple(c.clause_ref for c in index.qualification.clauses))
    request = replace(request, index=index, source_catalog=replace(request.source_catalog, sources=(source,)),
        strategy=replace(request.strategy, branches=(branch,)))
    options = request.finite_choice_options_for_owner(index.boolean_requirements[0].requirement_ref, branch_id=branch.branch_id)
    assert [choice.value for _, choices in options for choice in choices] == ['false']


@pytest.mark.parametrize("type_name", ["integer", "decimal", "number", "float", "double"])
@pytest.mark.parametrize("required", [False, True])
def test_documented_open_numeric_control_reaches_the_compiled_invocation(required, type_name):
    from fervis.lookup.relation_catalog.row_sources import RowSourceValueType
    from fervis.lookup.source_binding.parser import compile_source_realization, compile_source_binding_plan
    from fervis.lookup.source_binding.verification import verify_source_strategy, VerifiedSourceStrategy
    from fervis.lookup.fact_compilation import compile_verified_source_strategy
    from fervis.lookup.source_binding.invocation_bindings import invocation_value

    request = _request()
    source = request.source_catalog.sources[0]
    param = replace(source.params[0], type=RowSourceValueType(type_name), choices=(), required=required, default=None if required else 10,
                    description='Return at most this number of entries; 0 disables this limit.')
    source = replace(source, params=(param,))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    from fervis.lookup.source_binding.model import source_binding_clarification, source_inputs_allow_candidate_discovery
    assert source_inputs_allow_candidate_discovery(source, values=())
    if required:
        assert source_binding_clarification(request) is not None
    key, = population_interpretation_targets(request)
    payload = {'parameters': {key: {'kind': 'unfiltered_values', 'mapping_basis': 'The documentation states that 0 disables this limit.', 'values': ['0']}}}
    validate(payload, population_interpretation_schema(request))
    request = replace(request, population_interpretations=parse_population_interpretations(payload, request=request))
    from fervis.lookup.source_binding.model import source_binding_clarification
    assert source_binding_clarification(request) is None
    branch = request.strategy.branches[0].branch_id
    realized = compile_source_realization({
        'set_bindings': {request.index.subject_obligation.subject_set_ref.token: [
            { 'branch_id': branch, 'mapping_basis': 'Rows', 'rows_ref': source.id}]},
        'fact_bindings': {}, 'association_bindings': {},
    }, request=request)
    plan = compile_source_binding_plan({'resolved_input_applications': {branch: []},
        'choice_requirement_applications': {branch: {}}, 'finite_choice_applications': {branch: {}}}, realization=realized)
    [application] = plan.invocation_applications
    value = invocation_value(realized.request, application.value_ref)
    assert value.payload.value == '0'
    verified = verify_source_strategy(plan, request=realized.request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compiled = compile_verified_source_strategy(verified)
    assert len(compiled.answer_program.relations) == 1
    assert compiled.answer_program.relations[0].source.param_bindings


def test_source_realization_receives_the_verified_retrieval_arguments():
    from fervis.lookup.source_binding.prompt import SemanticSourceRealizationTurnPrompt
    request = _request()
    request = replace(request, population_interpretations=parse_population_interpretations(_payload(request), request=request, finite_cover=True))
    prompt = SemanticSourceRealizationTurnPrompt(request).to_model_payload(TurnPromptContext(current_question='Count entries'))
    [source_payload] = SemanticSourceRealizationTurnPrompt(request)._sources_payload()["sources"]
    assert source_payload["parameters"][0]["complete_read_arguments"] == ["false", "true"]
    assert '"union_identity_field_refs"' in prompt.prompt_text


def test_invalid_candidate_coverage_does_not_discard_valid_candidate_evidence():
    request = _request()
    source = request.source_catalog.sources[0]
    other = replace(source, id='other', params=(replace(source.params[0], param_ref='other.active'),))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source, other)))
    keys = tuple(population_interpretation_targets(request, finite_cover=True))
    payload = _payload(_request())
    valid = next(iter(payload['parameters'].values()))
    invalid = {**valid, 'arguments': [{'argument': 'true', 'row_values': ['true']}]}
    payload = {'parameters': {keys[0]: valid, keys[1]: invalid}}
    failures = []
    interpreted = parse_population_interpretations(payload, request=request, finite_cover=True, on_failure=failures.append)
    assert len(failures) == 1
    assert 'other.active' in str(failures[0])
    assert interpreted[0].population is not None
    assert interpreted[1].population is None
    assert interpreted[1].validation_error
    # Strict callers still reject invalid evidence rather than silently accepting it.
    with pytest.raises(ValueError, match='complete and irredundant'):
        parse_population_interpretations(payload, request=request, finite_cover=True)


def test_model_evidence_cannot_partition_rows_with_equivalent_numeric_arguments():
    from fervis.lookup.relation_catalog.row_sources import RowSourceValueType
    request = _request()
    source = request.source_catalog.sources[0]
    source = replace(source, params=(replace(source.params[0], type=RowSourceValueType.DECIMAL, choices=('1', '1.0'), default='1'),))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    key, = population_interpretation_targets(request, finite_cover=True)
    payload = {'parameters': {key: {'kind': 'finite_domain_cover', 'mapping_basis': 'Proposed partition',
        'field_ref': 'field.active', 'arguments': [{'argument': '1', 'row_values': ['true']}, {'argument': '1.0', 'row_values': ['false']}]}}}
    validate(payload, population_interpretation_schema(request, finite_cover=True))
    with pytest.raises(ValueError, match='same typed argument'):
        parse_population_interpretations(payload, request=request, finite_cover=True)


def test_open_optional_parameter_comparison_is_interpreted_from_source_contract():
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query
    from fervis.lookup.source_binding.population_interpretation import SourcePopulationTurnPrompt
    request = employee_query(manager_minimum=True).request
    source = request.source_catalog.sources[0]
    source = replace(source, params=(replace(source.params[0], population=None, source='query',
        description='Return exactly those employee rows whose salary is strictly greater than the supplied argument.'),))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    key, = population_interpretation_targets(request)
    payload = {'parameters': {key: {'kind': 'row_comparison', 'mapping_basis': 'The parameter compares the returned salary using strict greater-than.',
        'field_ref': 'field.salary', 'operator': 'gt'}}}
    validate(payload, population_interpretation_schema(request))
    request = replace(request, population_interpretations=parse_population_interpretations(payload, request=request))
    owner = next(item.requirement_ref for item in request.index.boolean_requirements if item.atom_ref.value_ref.endswith(':minimum_manager_salary'))
    assert request.direct_value_options_for_owner(owner, branch_id='branch')
    prompt = SourcePopulationTurnPrompt(replace(request, population_interpretations=())).to_model_payload(TurnPromptContext(current_question='CONSUMER_SECRET'))
    assert 'CONSUMER_SECRET' not in prompt.prompt_text
    assert 'strictly greater' in prompt.prompt_text
