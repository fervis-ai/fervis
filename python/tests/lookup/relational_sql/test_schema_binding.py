"""SQL owns its output inventory; binding observes schemas, never API rows."""
import pytest
from fervis.lookup.relational_sql.execution import SqlTable, QueryValidationError, describe_query


def test_describe_cte_join_keeps_hidden_ordering_column_and_uuid():
    tables = {'sales': SqlTable({'id':'UUID','staff_id':'UUID','amount':'DECIMAL(38,18)'}, ()),
              'staff': SqlTable({'id':'UUID'}, ())}
    assert describe_query('WITH selected AS (SELECT s.id, s.amount FROM sales s JOIN staff t ON s.staff_id=t.id) '
        'SELECT id AS sale_id, amount FROM selected ORDER BY amount DESC', tables=tables) == {
            'sale_id':'uuid', 'amount':'number'}


@pytest.mark.parametrize('expression,expected', [
    ('COUNT(*)','integer'), ('SUM(amount)','number'), ('AVG(amount)','number'),
    ('SUM(amount) / COUNT(*)','number'), ('MAX(day)','date'),
    ("day + INTERVAL '1 day'",'datetime'), ('LOWER(name)','string'),
    ('amount > $threshold','boolean'), ('$threshold * amount','number'),
])
def test_describe_expression_types_without_values(expression, expected):
    assert describe_query(f'SELECT {expression} AS result FROM records', tables={
        'records':SqlTable({'amount':'DECIMAL(38,18)','day':'DATE','name':'TEXT'}, ())},
        parameter_types={'threshold':'DECIMAL(38,18)'}) == {'result':expected}


@pytest.mark.parametrize('query', [
    'SELECT missing FROM records', 'SELECT amount AS x, amount AS x FROM records',
    "SELECT INTERVAL '1 day' AS duration FROM records", 'SELECT * FROM read_csv_auto(\'x\')',
    'SELECT $missing FROM records',
])
def test_describe_rejects_invalid_or_unsupported_schema(query):
    with pytest.raises(QueryValidationError):
        describe_query(query, tables={'records':SqlTable({'amount':'DECIMAL(38,18)'}, ())})


def test_describe_ignores_rows_including_unconvertible_values():
    assert describe_query('SELECT amount FROM records', tables={
        'records':SqlTable({'amount':'DECIMAL(38,18)'}, ({'amount':object()},))}) == {'amount':'number'}


@pytest.mark.parametrize('cte', ['__fervis_parameters__', '__FERVIS_PARAMETERS__'])
def test_parameter_schema_cannot_be_shadowed_by_cte(cte):
    assert describe_query(f"WITH {cte} AS (SELECT 'oops' AS threshold) SELECT $threshold AS value",
        tables={}, parameter_types={'threshold':'BIGINT'}) == {'value':'integer'}


@pytest.mark.parametrize('query', ['SELECT $x', 'SELECT $x + 1', 'WITH t AS (SELECT $x) SELECT * FROM t'])
def test_parameter_dependent_output_names_require_aliases(query):
    with pytest.raises(QueryValidationError, match='alias'):
        describe_query(query, tables={}, parameter_types={'x':'BIGINT'})
    from fervis.lookup.relational_sql.column_usage import project_query
    from fervis.lookup.relational_sql.execution import execute_query
    normalized, _ = project_query(query, {})
    schema = describe_query(normalized, tables={}, parameter_types={'x':'BIGINT'})
    executed = execute_query(normalized, tables={}, parameters={'x':2})
    assert tuple(schema) == executed.columns
    assert tuple(schema.values()) == ('integer',)


@pytest.mark.parametrize('precision', ['$places', 'CAST($places AS INTEGER)'])
def test_schema_binding_preserves_grounded_constant_rounding(precision):
    from decimal import Decimal
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    from fervis.lookup.relational_sql.execution import execute_query
    from tests.lookup.relational_sql.test_authoring import payload
    query = f'SELECT ROUND(amount, {precision}) AS total FROM records'
    definitions = {'records':{'columns':{'amount':{'type':'number'}}}}
    answer = parse_query_answer(payload(query=query), table_names=set(definitions), tables=definitions,
        parameter_names={'places'}, parameter_descriptions={'places':{'value_type':'integer','value':'1'}})
    assert answer.output_types == {'total':'number'}
    for places, expected in [(1,Decimal('12.3')), (2,Decimal('12.34'))]:
        result = execute_query(answer.query, tables={'records':SqlTable({'amount':'DECIMAL(10,2)'},
            ({'amount':Decimal('12.34')},))}, parameters={'places':places})
        assert result.rows == ((expected,),)


def test_persisted_numeric_parameter_keeps_its_domain_when_rebound_to_fraction():
    from dataclasses import replace
    from decimal import Decimal
    from fervis.lookup.answer_program.values import FactValue, LiteralType, BindingSet
    from fervis.lookup.grounding.semantic import CanonicalInputValue
    from fervis.lookup.relational_sql.parameters import query_parameter_menu
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    from fervis.lookup.relational_sql.binding import bind_query_answer
    from fervis.lookup.relational_sql.catalog import build_query_view_catalog
    from fervis.lookup.relational_sql.compiler import compile_query_answer
    from fervis.lookup.relation_catalog import RelationCatalog
    from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
    from fervis.lookup.question_contract import InputTerm, InputDenotation
    from fervis.lookup.question_contract.model import InputDenotationKind
    from fervis.lookup.semantic_types import NumericType, SourceOrigin, SourceOriginKind
    from tests.lookup.relational_engine.test_dependent_reads import _read
    from tests.lookup.relational_sql.test_authoring import payload
    value = FactValue.literal(id='factor',known_input_id='i1',literal_type=LiteralType.NUMBER,
        value='1',proof_refs=('question_input:i1',))
    menu = query_parameter_menu((CanonicalInputValue(value.id,'i1',('fact_1:sql_input:i1',),value,value.proof_refs),))
    catalog = RelationCatalog(reads=(_read('items'),))
    views = build_query_view_catalog(catalog)
    view = views.views[0]
    answer = parse_query_answer(payload(query=f'SELECT id * $p1_1 AS total FROM "{view.name}"'),
        tables=views.tables,table_names=set(views.tables),parameter_names=set(menu.expressions),
        parameter_descriptions=menu.descriptions)
    assert answer.output_types == {'total':'number'}
    bound = bind_query_answer(answer,menu,views.views)
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'the supplied factor')
    term = InputTerm('i1',origin,'1',NumericType())
    denotation = InputDenotation('d1','i1','the factor','Multiply by the supplied factor.',None,InputDenotationKind.NON_IDENTITY_SCALAR)
    compiled = compile_query_answer(question='Multiply the item ID by the supplied factor.',query=answer.query,
        views=bound.views,output_types=answer.output_types,result_contract=answer.result,catalog=catalog,
        query_parameters=bound.query_parameters,parameters=bound.parameters,bindings=bound.bindings,
        inputs=(term,),input_denotations=(denotation,))
    program = decode_answer_program(canonical_answer_program_json(compiled.program))
    class Port:
        def read(self, **kwargs):return {'responseStatus':200,'responseBody':[{'id':2}]}
    for factor,expected in [('1',Decimal('2')),('1.5',Decimal('3'))]:
        changed = replace(value,payload=replace(value.payload,value=factor))
        bindings = BindingSet.from_bindings(tuple(replace(binding,value=changed) for binding in compiled.bindings.bindings))
        result = invoke_answer_program(program=program,bindings=bindings,environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(),LookupMemory()))
        assert result.issue is None
        assert next(iter(result.fact_result.outcome.projected_rows[0].values.values())) == expected
