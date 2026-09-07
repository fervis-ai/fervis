"""A fixed returned fact cannot acquire unrelated invocation semantics later."""

from dataclasses import replace
import pytest

from fervis.host_api.contracts.population import ParameterPopulation, ParameterRowValues
from fervis.lookup.question_contract.model import FactTerm
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.semantic_types import BooleanType
from fervis.lookup.relation_catalog.row_sources import RowSourceField, RowSourceValueType
from fervis.lookup.source_binding.parser import compile_source_realization
from tests.lookup.source_binding.test_choice_requirements import _source_required_choice_request


def _realization(mapped, mapping_field="field.flag"):
    request, branch, set_ref = _source_required_choice_request(choices=('yes', 'no'))
    fact = request.index.requested_fact
    fact = replace(fact, facts=(FactTerm('flag', 's1', BooleanType(), fact.origin),), qualification_ref='flag')
    field = RowSourceField('flag', 'field.flag', 'Flag', RowSourceValueType.BOOLEAN, ())
    source = request.source_catalog.sources[0]
    effect = ParameterPopulation(field_path=mapping_field, value_mapping=(
        ParameterRowValues('yes', ('true',)), ParameterRowValues('no', ('false',)))) if mapped else None
    source = replace(source, fields=(field, replace(field, id="other", field_ref="field.other")), params=(replace(source.params[0], required=False, population=effect),))
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    strategy = replace(request.strategy, branches=tuple(replace(b, qualification_clause_refs=tuple(c.clause_ref for c in index.qualification.clauses)) for b in request.strategy.branches))
    request = replace(request, index=index, strategy=strategy,
                      source_catalog=replace(request.source_catalog, sources=(source,)))
    owner = request.index.boolean_requirements[0].requirement_ref
    # Predicates with no returned realization may still be supplied by the API.
    assert request.finite_choice_options_for_owner(owner, branch_id=branch)
    realization = compile_source_realization({
        'set_bindings': {set_ref: [{ 'branch_id': branch, 'mapping_basis': 'Rows', 'rows_ref': source.id}]},
        'fact_bindings': {'fact_1:fact:flag': [{'branch_id': branch, 'mapping_basis': 'Returned flag', 'field_ref': f'source_field:{source.id}:flag'}]},
        'association_bindings': {},
    }, request=request)
    return realization, owner, branch


@pytest.mark.parametrize("mapped", [False, True])
def test_request_predicate_preserves_the_selected_fact_correspondence(mapped):
    realization, owner, branch = _realization(mapped)
    assert bool(realization.request.finite_choice_options_for_owner(owner, branch_id=branch)) is mapped


@pytest.mark.parametrize('argument,valid', [('yes', True), ('no', False)])
def test_api_predicate_must_agree_with_returned_field_truth(argument, valid):
    from fervis.lookup.source_binding.parser import compile_source_binding_plan
    from fervis.lookup.source_binding.verification import verify_source_strategy, VerifiedSourceStrategy
    realization, owner, branch = _realization(True)
    source = realization.request.source_catalog.sources[0]
    plan = compile_source_binding_plan({
        'resolved_input_applications': {branch: []},
        'finite_choice_applications': {branch: {owner: {
            'application_basis': 'Apply the requested flag predicate',
            'surface_ref': f'source_surface:{source.id}:parameter:mode',
            'selected_choice_values': [argument]}}},
        'choice_requirement_applications': {branch: {f'source_surface:{source.id}:field:flag': {
            'false': {'mapping_basis': 'False does not satisfy flag', 'selected_by_requirements': []},
            'true': {'mapping_basis': 'True satisfies flag', 'selected_by_requirements': [owner]}}}},
    }, realization=realization)
    assert isinstance(verify_source_strategy(plan, request=realization.request), VerifiedSourceStrategy) is valid


def test_request_mapping_to_another_field_cannot_replace_the_fixed_fact():
    realization, owner, branch = _realization(True, 'field.other')
    assert not realization.request.finite_choice_options_for_owner(owner, branch_id=branch)


@pytest.mark.parametrize('admitted,valid', [('true', True), ('false', False), (None, False)])
def test_direct_input_must_agree_with_declared_field_mapping(monkeypatch, admitted, valid):
    from fervis.lookup.source_binding.model import InvocationValueApplication, InvocationTargetApplication
    from fervis.lookup.answer_program.values import ValueProjectionKind
    from fervis.lookup.source_binding.verification import verify_source_strategy, VerifiedSourceStrategy
    from fervis.lookup.relation_catalog.row_sources import RowSourceParam
    from tests.lookup.source_binding import test_boolean_category_predicates as fixture
    saved = {}

    class Captured(Exception):
        pass

    def capture(plan, *, request):
        saved.update(plan=plan, request=request)
        raise Captured

    monkeypatch.setattr(fixture, 'verify_source_strategy', capture)
    with pytest.raises(Captured):
        fixture.test_text_category_uses_boolean_predicate_without_casting_its_raw_value()
    request, plan = saved['request'], saved['plan']
    source = request.source_catalog.sources[0]
    field = source.field('is_active')
    param = RowSourceParam('state_filter', 'param.state_filter', 'State filter', RowSourceValueType.STRING,
        population=ParameterPopulation(field_path=field.field_ref, value_mapping=(ParameterRowValues('active', (admitted,)),)) if admitted is not None else None)
    source = replace(source, params=(param,))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    owner = request.index.boolean_requirements[0].requirement_ref
    branch = request.strategy.branches[0].branch_id
    value_ref = request.canonical_values[0].canonical_value_id
    assert any(option.target_ref == param.param_ref for option in request.direct_value_options_for_owner(owner, branch_id=branch)) is (admitted is not None)
    application = InvocationValueApplication('direct', branch, source.id, value_ref, owner, (
        InvocationTargetApplication('Apply category', param.param_ref, value_ref, ValueProjectionKind.WHOLE_VALUE, None),))
    plan = replace(plan, invocation_applications=(application,))
    assert isinstance(verify_source_strategy(plan, request=request), VerifiedSourceStrategy) is valid


@pytest.mark.parametrize('change', ['operator', 'reversed', 'computed'])
def test_scalar_parameter_correspondence_preserves_the_exact_comparison(change):
    from fervis.lookup.question_contract.model import Arithmetic
    from fervis.lookup.expression_operators import ExpressionBinaryOperator as B
    from fervis.lookup.source_binding.verification import verify_source_strategy, SourceStrategyVerificationFailure
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query
    verified = employee_query(manager_minimum=True)
    request = verified.request
    fact = request.index.requested_fact
    expression = next(item for item in fact.expressions if item.id == 'minimum_manager_salary')
    extra = ()
    if change == 'operator':
        changed = replace(expression, operator=B.GTE)
    elif change == 'reversed':
        changed = replace(expression, left_ref=expression.right_ref, right_ref=expression.left_ref)
    else:
        extra = (Arithmetic('scaled_salary', B.ADD, (expression.left_ref, expression.left_ref), fact.origin),)
        changed = replace(expression, left_ref='scaled_salary')
    fact = replace(fact, expressions=(*extra, *(changed if item.id == expression.id else item for item in fact.expressions)))
    index = analyze_requested_fact(fact, inputs=request.index.input_by_ref, input_denotations=request.index.input_denotation_by_ref)
    request = replace(request, index=index)
    owner = next(item.requirement_ref for item in index.boolean_requirements if item.atom_ref.value_ref.endswith(':minimum_manager_salary'))
    assert not request.direct_value_options_for_owner(owner, branch_id='branch')
    assert isinstance(verify_source_strategy(verified.binding_plan, request=request), SourceStrategyVerificationFailure)


def test_argument_entity_type_does_not_establish_which_rows_a_parameter_selects(monkeypatch):
    from fervis.lookup.relation_catalog import EntityKeyComponentTarget
    from fervis.lookup.relation_catalog.row_sources import RowSourceParam
    from tests.lookup.fact_compilation import test_compiler as fixture
    saved = {}
    class Captured(Exception):
        pass
    def capture(verified):
        saved['request'] = verified.request
        raise Captured
    monkeypatch.setattr(fixture, 'compile_verified_source_strategy', capture)
    with pytest.raises(Captured):
        fixture.test_co_resident_exists_filters_subject_rows_before_counting()
    request = saved['request']
    source = request.source_catalog.sources[0]
    param = RowSourceParam('excluded_area', 'param.excluded_area', 'Area to exclude', RowSourceValueType.STRING,
        entity_target=EntityKeyComponentTarget('area', 'primary_key', 'area_id'))
    source = replace(source, params=(param,))
    request = replace(request, source_catalog=replace(request.source_catalog, sources=(source,)))
    options = [option for owner in request.invocation_application_owner_refs
               for option in request.direct_value_options_for_owner(owner, branch_id=request.strategy.branches[0].branch_id)]
    assert not options


def test_request_optimization_retains_the_fixed_returned_predicate():
    from fervis.lookup.source_binding.model import SourceRealization, SourceMechanicKind
    from fervis.lookup.source_binding.parser import compile_source_binding_plan
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query
    verified = employee_query(manager_minimum=True)
    request, plan = verified.request, verified.binding_plan
    owner = next(item.requirement_ref for item in request.index.boolean_requirements if item.atom_ref.value_ref.endswith(':minimum_manager_salary'))
    realized = SourceRealization(request, plan.set_bindings, plan.fact_bindings, plan.association_bindings)
    bound = compile_source_binding_plan({
        'resolved_input_applications': {'branch': [{'kind': 'request_application',
            'owner_ref': owner, 'value_ref': 'minimum_value', 'target_ref': 'employees.min_salary',
            'value_component': 'WHOLE_VALUE', 'mapping_basis': 'Declared strict salary bound'}]},
        'finite_choice_applications': {'branch': {}}, 'choice_requirement_applications': {'branch': {}},
    }, realization=realized)
    [predicate] = bound.boolean_bindings[owner]
    assert {item.kind for item in predicate.mechanics} == {
        SourceMechanicKind.INVOCATION_PREDICATE, SourceMechanicKind.RETURNED_ROW_PREDICATE}
