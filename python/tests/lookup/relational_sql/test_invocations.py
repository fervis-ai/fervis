"""One API definition can supply independently bound relations in one answer."""
from types import SimpleNamespace

import pytest

from fervis.lookup.relational_sql.authoring import parse_query_answer
from fervis.lookup.relational_sql.execution import QueryValidationError
from tests.lookup.relational_sql.test_authoring import payload


def authored(arguments=None, query=None, tables=None):
    return parse_query_answer(payload(
        query=query or 'SELECT (SELECT COUNT(*) FROM first_item) + (SELECT COUNT(*) FROM second_item) AS total',
        request_arguments=arguments or [
            {'view': 'items', 'instance': 'first_item', 'parameter_ref': 'id', 'binding': 'p1'},
            {'view': 'items', 'instance': 'second_item', 'parameter_ref': 'id', 'binding': 'p2'},
        ]), table_names={'items', 'other'}, parameter_names={'p1', 'p2'},
        request_parameters={'items': {'id'}}, tables=tables)


def test_same_endpoint_can_supply_two_independently_bound_sql_relations():
    from fervis.lookup.relational_sql.acquisition import ApiView
    from fervis.lookup.relational_sql.binding import bind_query_answer
    from fervis.lookup.answer_program.values import BindingSet, ParameterRef

    answer = authored()
    expressions = {'p1': ParameterRef('first'), 'p2': ParameterRef('second')}
    menu = SimpleNamespace(expressions=expressions, descriptions={'p1': {}, 'p2': {}},
        program_inputs=SimpleNamespace(parameters=(), bindings=BindingSet()))
    bound = bind_query_answer(answer, menu, (ApiView('items', 'source', {'id': 'id'}, {}),))
    selected = {view.name: view for view in bound.views if view.name in answer.referenced_views}
    assert set(selected) == {'first_item', 'second_item'}
    assert selected['first_item'].arguments == {'id': expressions['p1']}
    assert selected['second_item'].arguments == {'id': expressions['p2']}
    assert {view.row_source_id for view in selected.values()} == {'source'}


@pytest.mark.parametrize('arguments', [
    [{'view': 'items', 'instance': 'other', 'parameter_ref': 'id', 'binding': 'p1'}],
    [{'view': 'items', 'instance': 'first_item', 'parameter_ref': 'id', 'binding': 'p1'},
     {'view': 'other', 'instance': 'first_item', 'parameter_ref': 'id', 'binding': 'p2'}],
    [{'view': 'items', 'instance': 'first_item', 'parameter_ref': 'id', 'binding': 'p1'},
     {'view': 'items', 'instance': 'FIRST_ITEM', 'parameter_ref': 'id', 'binding': 'p2'}],
    [{'view': 'invented', 'instance': 'first_item', 'parameter_ref': 'id', 'binding': 'p1'}],
])
def test_invocations_cannot_shadow_tables_or_mix_definitions(arguments):
    with pytest.raises(QueryValidationError):
        authored(arguments=arguments)


def test_repeated_parameter_within_one_invocation_is_rejected():
    arguments = [
        {'view': 'items', 'instance': 'first_item', 'parameter_ref': 'id', 'binding': binding}
        for binding in ('p1', 'p2')]
    with pytest.raises(QueryValidationError, match='repeated'):
        authored(arguments=arguments, query='SELECT COUNT(*) AS total FROM first_item')


def test_named_invocations_execute_and_replay_without_crossing_arguments():
    from fervis.lookup.relational_sql.binding import bind_query_answer, reads_requiring_access_discovery
    from fervis.lookup.relational_sql.acquisition import ApiView
    from fervis.lookup.relational_sql.compiler import compile_query_answer
    from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType, BindingSet
    from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from tests.lookup.relational_engine.test_dependent_reads import _program

    _, _, catalog = _program()
    source = build_api_row_source_catalog(catalog).sources[1]
    args = [{'view': 'items', 'instance': name, 'parameter_ref': 'facility_id', 'binding': binding}
            for name, binding in [('first_item', 'p1'), ('second_item', 'p2')]]
    answer = parse_query_answer(payload(query='SELECT SUM(id) AS total FROM (SELECT id FROM first_item UNION ALL SELECT id FROM second_item)',
        request_arguments=args), table_names={'items'}, parameter_names={'p1', 'p2'},
        request_parameters={'items': {'facility_id'}})
    expressions = {name: ConstantRef(name, 'question', FactValue.literal(id=name,
        literal_type=LiteralType.NUMBER, value=str(value), proof_refs=('question',)))
        for name, value in [('p1', 1), ('p2', 2)]}
    menu = SimpleNamespace(expressions=expressions, descriptions={'p1': {}, 'p2': {}},
        program_inputs=SimpleNamespace(parameters=(), bindings=BindingSet()))
    views = (ApiView('items', source.id, {'id': source.fields[0].id}, {}),)
    assert reads_requiring_access_discovery(answer, views, catalog=catalog) == ()
    bound = bind_query_answer(answer, menu, views)
    compiled = compile_query_answer(question='Total across two facilities?', query=answer.query,
        views=bound.views, output_types=answer.output_types, result_contract=answer.result, catalog=catalog)
    program = decode_answer_program(canonical_answer_program_json(compiled.program))
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, args))
            return {'responseStatus': 200, 'responseBody': [{'id': int(args['facility_id']) * 10}]}
    for _ in range(2):
        result = invoke_answer_program(program=program, bindings=compiled.bindings,
            environment=ExecutionEnvironment(catalog=catalog), ports=RuntimePorts(Port(), LookupMemory()))
        assert result.issue is None
        assert list(result.fact_result.outcome.projected_rows[0].values.values()) == [30]
    assert calls == [('instruments', {'facility_id': 1}), ('instruments', {'facility_id': 2})] * 2


def test_reference_argument_operations_are_owned_by_fact_namespace():
    from fervis.lookup.relational_sql.binding import bind_query_answer
    from fervis.lookup.relational_sql.acquisition import ApiView
    from fervis.lookup.answer_program.values import BindingSet
    from fervis.lookup.answer_program.expressions import FieldRef
    answer = authored()
    def bind(fact):
        menu = SimpleNamespace(expressions={'p1': FieldRef('id'), 'p2': FieldRef('id')},
            descriptions={name: {'kind': 'reference_argument', 'relation_id': fact+'.'+name} for name in ('p1', 'p2')},
            program_inputs=SimpleNamespace(parameters=(), bindings=BindingSet()))
        return bind_query_answer(answer, menu, (ApiView('items', 'source', {'id': 'id'}, {}),), namespace=fact+'.')
    first, second = bind('fact_1'), bind('fact_2')
    assert {operation.id for operation in first.argument_operations}.isdisjoint(
        operation.id for operation in second.argument_operations)
    for bound in (first, second):
        outputs = {operation.output_relation for operation in bound.argument_operations}
        assert {view.argument_relation_id for view in bound.views if view.name in answer.referenced_views} <= outputs


def test_declared_invocations_bind_sql_range_aliases_without_changing_the_query_population():
    answer = authored(query='SELECT (SELECT COUNT(*) FROM items AS first_item) + (SELECT COUNT(*) FROM items AS second_item) AS total')
    assert set(answer.referenced_views) == {'first_item', 'second_item'}
    from fervis.lookup.relational_sql.execution import execute_query, SqlTable
    result = execute_query(answer.query, tables={
        'first_item': SqlTable({'id': 'integer'}, ({'id': 1}, {'id': 2})),
        'second_item': SqlTable({'id': 'integer'}, ({'id': 3},)),
    })
    assert result.rows == ((3,),)


def test_invocation_range_alias_in_cte_does_not_capture_the_cte_reference():
    answer = authored(arguments=[{'view':'items','instance':'selected','parameter_ref':'id','binding':'p1'}],
        query='WITH selected AS (SELECT id FROM items AS selected) SELECT COUNT(*) AS total FROM selected')
    from fervis.lookup.relational_sql.execution import execute_query, SqlTable
    assert execute_query(answer.query, tables={'selected': SqlTable({'id':'integer'}, ({'id':1},))}).rows == ((1,),)
