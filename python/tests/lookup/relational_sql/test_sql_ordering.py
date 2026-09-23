"""Lower SQL ordering and explicit public fields without duplicate inventories."""
import pytest
from fervis.lookup.relational_sql.query_projection import lower_query_projection
from fervis.lookup.relational_sql.execution import execute_query, SqlTable, QueryValidationError


@pytest.mark.parametrize('sort_expression,valid', [('total', True), ('SUM(score)', True), ('MAX(score)', False)])
def test_ordering_preserves_the_question_reference_to_a_returned_value(sort_expression, valid):
    from types import SimpleNamespace
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    from tests.lookup.relational_sql.test_authoring import payload
    meaning = SimpleNamespace(result_kind='qualifying_instances', output_kinds=('value',),
        output_origins=('total score',), ordering_origins=('that returned total',),
        ordering_value_refs=('r1',), ordering_group_refs=(None,), requested_value_refs=('r1',),
        result_key_count=0, selection_kind='first_rank_with_ties')
    body = payload(query=f'SELECT SUM(score) AS total FROM items GROUP BY id ORDER BY {sort_expression} DESC',mode='rows')
    def parse():
        return parse_query_answer(body,table_names={'items'},parameter_names=set(),meaning=meaning,
            tables={'items':{'columns':{'id':{'type':'integer'},'score':{'type':'integer'}}}})
    if not valid:
        with pytest.raises(QueryValidationError, match='requested value'):
            parse()
        return
    answer = parse()
    assert answer.result.ordering[0].column == answer.outputs[0].column


@pytest.mark.parametrize('sort_columns,valid', [('n, total', True), ('total, n', False)])
def test_multiple_requested_value_references_keep_their_own_output_after_group_keys(sort_columns, valid):
    from types import SimpleNamespace
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    from tests.lookup.relational_sql.test_authoring import payload
    meaning = SimpleNamespace(result_kind='grouped_results',output_kinds=('value','value','value'),
        output_origins=('group','total','count'),ordering_origins=('count','total'),
        ordering_value_refs=('r2','r1'),ordering_group_refs=(None,None),requested_value_refs=('r1','r2'),
        result_key_count=1,selection_kind='all_results')
    body=payload(query=f'SELECT id, SUM(score) AS total, COUNT(*) AS n FROM items GROUP BY id ORDER BY {sort_columns}',
        mode='rows',outputs=[{'kind':'value','column':name,'label':name} for name in ('id','total','n')])
    def parse():
        return parse_query_answer(body,table_names={'items'},parameter_names=set(),meaning=meaning,
            tables={'items':{'columns':{'id':{'type':'integer'},'score':{'type':'integer'}}}})
    if valid:
        assert tuple(item.column for item in parse().result.ordering)==('n','total')
    else:
        with pytest.raises(QueryValidationError, match='requested value'):
            parse()


def test_hidden_order_expression_and_display_property_are_projected():
    query, orders = lower_query_projection('SELECT id FROM items ORDER BY COALESCE(score, 0) DESC',
        {'items':{'id','name','score'}}, public_columns=('id','name'))
    result = execute_query(query, tables={'items':SqlTable({'id':'BIGINT','name':'TEXT','score':'BIGINT'},
        ({'id':1,'name':'A','score':None},{'id':2,'name':'B','score':-1}))})
    assert len(orders)==1 and orders[0].descending
    assert result.columns[:2]==('id','name')
    assert result.rows==((1,'A',0),(2,'B',-1))
    assert 'ORDER BY' not in query


def test_existing_aliases_and_positional_ordering_remain_single_fields():
    query, orders = lower_query_projection('SELECT id, score AS rank_value FROM items ORDER BY 2 DESC, id ASC',
        {'items':{'id','score'}})
    assert [(order.column,order.descending) for order in orders]==[('rank_value',True),('id',False)]
    assert execute_query(query,tables={'items':SqlTable({'id':'BIGINT','score':'BIGINT'},())}).columns==('id','rank_value')


def test_grouped_hidden_aggregate_and_cte_property():
    query, orders = lower_query_projection('WITH members AS (SELECT id, score FROM items) '
        'SELECT id FROM members GROUP BY id ORDER BY SUM(score) DESC', {'items':{'id','score'}})
    result=execute_query(query,tables={'items':SqlTable({'id':'BIGINT','score':'BIGINT'},({'id':1,'score':2},{'id':1,'score':3}))})
    assert result.rows==((1,5),)
    assert result.columns[-1]==orders[0].column


def test_hidden_alias_does_not_shadow_selected_column():
    query, orders = lower_query_projection('SELECT id AS __fervis_order_1 FROM items ORDER BY score',
        {'items':{'id','score'}})
    assert orders[0].column!='__fervis_order_1'


@pytest.mark.parametrize('query', [
    'SELECT DISTINCT id FROM items ORDER BY score',
    'SELECT id FROM items ORDER BY score NULLS FIRST',
    'SELECT id FROM items ORDER BY 9',
])
def test_unsafe_or_unsupported_order_projection_is_rejected(query):
    with pytest.raises(QueryValidationError):lower_query_projection(query,{'items':{'id','score'}})


def test_union_order_on_selected_column_preserves_set_grain():
    query, orders = lower_query_projection('SELECT id FROM items UNION SELECT id FROM items ORDER BY id DESC',
        {'items':{'id'}})
    result=execute_query(query,tables={'items':SqlTable({'id':'BIGINT'},({'id':1},{'id':1}))})
    assert result.rows==((1,),)
    assert orders[0].column=='id' and orders[0].descending


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('outer_rank', [False, True])
def test_distinct_on_keeps_its_observation_selection_before_result_ranking(reverse, outer_rank):
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    rows = [{'id':1,'score':2},{'id':1,'score':7},{'id':2,'score':4}]
    if reverse:rows.reverse()
    sql = 'SELECT DISTINCT ON (id) id, score FROM items ORDER BY score DESC'
    if outer_rank:sql = f'SELECT id, score FROM ({sql}) AS latest ORDER BY id DESC'
    body = {'query':sql,'mode':'rows','api_invocations':[{'view':'items','name':'items','arguments':[],'population_bindings':[]}],
        'outputs':[{'kind':'value','column':'id','label':'item'}],'interpretations':[]}
    answer = parse_query_answer(body,table_names={'items'},parameter_names=set(),
        tables={'items':{'columns':{'id':{'type':'integer'},'score':{'type':'integer'}}}})
    result = execute_query(answer.query,tables={'items':SqlTable({'id':'BIGINT','score':'BIGINT'},tuple(rows))})
    assert set(result.rows)=={(1,7),(2,4)}
    assert answer.result.ordering[0].column==('id' if outer_rank else 'score')


def test_distinct_on_hidden_sort_key_keeps_the_same_selected_record():
    query, orders = lower_query_projection('SELECT DISTINCT ON (id) id FROM items ORDER BY score DESC',
        {'items':{'id','score'}})
    result=execute_query(query,tables={'items':SqlTable({'id':'BIGINT','score':'BIGINT'},
        ({'id':1,'score':2},{'id':1,'score':7},{'id':2,'score':4}))})
    assert {row[0]:row[1] for row in result.rows}=={1:7,2:4}
    assert orders[0].descending
