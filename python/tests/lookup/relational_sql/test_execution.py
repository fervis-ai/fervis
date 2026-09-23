from decimal import Decimal
import pytest
from fervis.lookup.relational_sql.execution import SqlTable, execute_query, QueryValidationError


def test_decimal_arithmetic_does_not_cross_a_float_boundary():
    tables = {'amounts': SqlTable({'value': 'DECIMAL(30,20)'}, ({'value': Decimal('0.1')}, {'value': Decimal('0.2')}))}
    result = execute_query('SELECT SUM(value)*0.5 AS half, AVG(value) AS mean, 1/3 AS fraction FROM amounts', tables=tables)
    assert result.rows == ((Decimal('0.15'), Decimal('0.15'), Decimal('0.333333333333333333')),)
    assert result.source_tables == ('amounts',)


def test_correlated_absence_preserves_null_foreign_keys():
    tables = {
        'people': SqlTable({'id': 'INT'}, ({'id': 1}, {'id': 2})),
        'payments': SqlTable({'person_id': 'INT'}, ({'person_id': 1}, {'person_id': None})),
    }
    result = execute_query('SELECT p.id FROM people p WHERE NOT EXISTS (SELECT 1 FROM payments x WHERE x.person_id=p.id)', tables=tables)
    assert result.rows == ((2,),)
    assert result.source_tables == ('payments', 'people')


@pytest.mark.parametrize('query', [
    'DROP TABLE people',
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM '/etc/passwd'",
    "SELECT load_extension('anything')",
    'SELECT * FROM people; DELETE FROM people',
    'SELECT missing FROM people',
    'SELECT * FROM unregistered',
    'SELECT random() FROM people',
])
def test_query_boundary_rejects_mutations_external_access_and_unknown_names(query):
    with pytest.raises(QueryValidationError):
        execute_query(query, tables={'people': SqlTable({'id': 'INT'}, ({'id': 1},))})


def test_bound_values_are_data_even_when_they_contain_sql():
    value = "x'; DROP TABLE people; --"
    tables = {'people': SqlTable({'id': 'INT', 'name': 'TEXT'}, ({'id': 1, 'name': value},))}
    result = execute_query('SELECT id FROM people WHERE name=:name', tables=tables, parameters={'name': value})
    assert result.rows == ((1,),)


def test_empty_sum_and_average_preserve_sql_null_semantics():
    result = execute_query('SELECT SUM(value) AS total, AVG(value) AS mean FROM amounts', tables={'amounts': SqlTable({'value': 'DECIMAL'}, ())})
    assert result.rows == ((None, None),)


def test_decimal_totals_are_exact_and_independent_of_caller_context():
    from decimal import localcontext
    value = Decimal('12345678901234567890.123456789012345678')
    tables = {'t': SqlTable({'v': 'DECIMAL(38,18)'}, ({'v': value},))}
    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            assert execute_query('SELECT SUM(v) FROM t', tables=tables).rows == ((value,),)
    tables['t'] = SqlTable({'v': 'DECIMAL(38,18)'}, ({'v': value}, {'v': value.copy_negate()}, {'v': Decimal('0.000000000000000001')}))
    assert execute_query('SELECT SUM(v) FROM t', tables=tables).rows == ((Decimal('0.000000000000000001'),),)


def test_positional_ordering_and_grouping_reference_output_expressions():
    tables = {'t': SqlTable({'id': 'INT', 'v': 'INT'}, ({'id': 2, 'v': 3}, {'id': 1, 'v': 4}, {'id': 2, 'v': 5}))}
    assert execute_query('SELECT id FROM t ORDER BY 1 LIMIT 1', tables=tables).rows == ((1,),)
    assert execute_query('SELECT id, SUM(v) AS s FROM t GROUP BY 1 ORDER BY 1', tables=tables).rows == ((1, Decimal(4)), (2, Decimal(8)))


@pytest.mark.parametrize(('expression', 'expected'), [
    ("CAST('false' AS BOOLEAN)", False),
    ("CAST('true' AS BOOLEAN)", True),
    ('CAST(0 AS BOOLEAN)', False),
    ('CAST(2 AS BOOLEAN)', True),
    ('CAST(NULL AS BOOLEAN)', None),
    ("CAST('1.2' AS DECIMAL(10,2))", Decimal('1.20')),
])
def test_casts_follow_declared_types(expression, expected):
    assert execute_query(f'SELECT {expression} AS value', tables={}).rows == ((expected,),)


def test_invalid_boolean_cast_is_rejected():
    with pytest.raises(QueryValidationError):
        execute_query("SELECT CAST('not-a-boolean' AS BOOLEAN)", tables={})


def test_window_and_safe_cast_execute_with_database_semantics():
    tables = {'t': SqlTable({'id': 'INT'}, ({'id': 2}, {'id': 1}))}
    assert execute_query('SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS n FROM t ORDER BY id', tables=tables).rows == ((1, 1), (2, 2))
    assert execute_query("SELECT TRY_CAST('x' AS DECIMAL)", tables={}).rows == ((None,),)


def test_boolean_connectives_preserve_three_valued_logic():
    tables = {'t': SqlTable({'a': 'BOOLEAN', 'b': 'BOOLEAN'}, ({'a': None, 'b': False}, {'a': None, 'b': True}, {'a': True, 'b': False}))}
    assert execute_query('SELECT a AND b AS both, a OR b AS either FROM t', tables=tables).rows == ((False, None), (None, True), (False, True))


def test_every_required_observation_uses_a_correlated_double_absence():
    tables = {
        'workers': SqlTable({'id': 'INT'}, ({'id': 1}, {'id': 2}, {'id': 3})),
        'requirements': SqlTable({'id': 'INT'}, ({'id': 7}, {'id': 8})),
        'observations': SqlTable({'worker': 'INT', 'requirement': 'INT', 'accepted': 'BOOLEAN'}, (
            {'worker': 1, 'requirement': 7, 'accepted': True},
            {'worker': 1, 'requirement': 8, 'accepted': True},
            {'worker': 2, 'requirement': 7, 'accepted': True},
            {'worker': 2, 'requirement': 8, 'accepted': False},
            {'worker': None, 'requirement': 8, 'accepted': True},
        )),
    }
    query = '''SELECT w.id FROM workers w WHERE NOT EXISTS (
        SELECT 1 FROM requirements r WHERE NOT EXISTS (
            SELECT 1 FROM observations o WHERE o.worker = w.id
            AND o.requirement = r.id AND o.accepted = TRUE))'''
    assert execute_query(query, tables=tables).rows == ((1,),)
    tables['requirements'] = SqlTable({'id': 'INT'}, ())
    assert execute_query(query, tables=tables).rows == ((1,), (2,), (3,))


@pytest.mark.parametrize(('expression', 'expected'), [
    ('MIN(v)', Decimal('1.2')), ('MAX(v)', Decimal('2.8')),
    ('ABS(-1.2)', Decimal('1.2')), ('ROUND(1.234, 2)', Decimal('1.23')),
    ("LOWER('ABC')", 'abc'), ("UPPER('abc')", 'ABC'),
    ("TRIM(' ab ')", 'ab'), ("CONCAT('a', 'b')", 'ab'),
    ("CONCAT_WS('-', 'a', 'b')", 'a-b'), ("LENGTH('abc')", 3),
    ("SUBSTRING('abcd', 2, 2)", 'bc'), ('COALESCE(NULL, 7)', 7),
    ('NULLIF(7,7)', None), ('CASE WHEN TRUE THEN 1 ELSE 2 END', 1),
    ("EXTRACT(YEAR FROM CAST('2026-09-08' AS DATE))", 2026),
])
def test_advertised_scalar_and_aggregate_capabilities(expression, expected):
    table = SqlTable({'v': 'DECIMAL(10,2)'}, ({'v': Decimal('1.2')}, {'v': Decimal('2.8')}))
    result = execute_query(f'SELECT {expression} AS value FROM t', tables={'t': table})
    assert result.rows and all(row == (expected,) for row in result.rows)


def test_outer_join_aggregation_preserves_unmatched_rows_and_distinct_values():
    tables = {
        'parents': SqlTable({'id': 'INT'}, ({'id': 1}, {'id': 2})),
        'children': SqlTable({'parent_id': 'INT', 'value': 'INT'}, (
            {'parent_id': 1, 'value': 7}, {'parent_id': 1, 'value': 7}, {'parent_id': 1, 'value': 8},
        )),
    }
    query = '''SELECT p.id, COUNT(c.value) AS n, COUNT(DISTINCT c.value) AS d
               FROM parents p LEFT JOIN children c ON c.parent_id = p.id
               GROUP BY p.id ORDER BY p.id'''
    assert execute_query(query, tables=tables).rows == ((1, 3, 2), (2, 0, 0))


def test_exact_input_values_are_never_rounded_to_fit_a_view():
    table = SqlTable({'v': 'DECIMAL(10,2)'}, ({'v': Decimal('1.234')},))
    with pytest.raises(QueryValidationError, match='declared scale'):
        execute_query('SELECT SUM(v) FROM t', tables={'t': table})


def test_scientific_literals_use_fixed_point_in_comparisons_and_output():
    assert execute_query('SELECT 1e-3 + 2e-3', tables={}).rows == ((Decimal('0.003'),),)
    with pytest.raises(QueryValidationError):
        execute_query('SELECT 1 WHERE CAST(9007199254740993 AS DOUBLE) = 9007199254740992', tables={})


def test_average_distinct_and_null_division_follow_numeric_contract():
    table = SqlTable({'v': 'DECIMAL(10,2)'}, ({'v': Decimal('1')}, {'v': Decimal('1')}, {'v': Decimal('4')}, {'v': None}))
    assert execute_query('SELECT AVG(DISTINCT v), NULL/2, 1/0 FROM t', tables={'t': table}).rows == ((Decimal('2.5'), None, None),)


def test_query_execution_is_interrupted_at_its_deadline():
    table = SqlTable({'id': 'BIGINT'}, tuple({'id': i} for i in range(1000)))
    with pytest.raises(QueryValidationError, match='Interrupt'):
        execute_query('SELECT SUM(a.id*b.id*c.id*d.id) FROM t a, t b, t c, t d', tables={'t': table}, timeout_seconds=0.01)


@pytest.mark.parametrize('query', [
    'SELECT CAST(9007199254740993.0 // 1.0 AS BIGINT)',
    'SELECT 1 WHERE 0.3 // 0.1 < 3',
    'SELECT 1 WHERE 12345678901234567890.1234567890123456789 = 12345678901234567890.1234567890123456788',
])
def test_intermediate_approximate_arithmetic_cannot_hide_in_a_predicate_or_cast(query):
    with pytest.raises(QueryValidationError, match='precision|approximate'):
        execute_query(query, tables={})


def test_average_modifiers_are_preserved_on_both_numerator_and_denominator():
    table = SqlTable({'id': 'INT', 'v': 'DECIMAL(10,2)'}, ({'id': 1, 'v': Decimal('1')}, {'id': 2, 'v': Decimal('3')}, {'id': 3, 'v': None}))
    assert execute_query('SELECT AVG(v) FILTER (WHERE id=1) FROM t', tables={'t': table}).rows == ((Decimal('1'),),)
    assert execute_query('SELECT AVG(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t ORDER BY id', tables={'t': table}).rows == ((Decimal('1'),), (Decimal('2'),), (Decimal('2'),))
    assert execute_query('SELECT AVG(DISTINCT v) FILTER (WHERE id<3) OVER () FROM t ORDER BY id', tables={'t': table}).rows == ((Decimal('2'),),) * 3


def test_naive_and_timezone_aware_timestamp_views_preserve_their_distinct_types():
    from datetime import datetime, timezone
    assert execute_query('SELECT v FROM t', tables={'t': SqlTable({'v': 'TIMESTAMP'}, ({'v': '2026-09-08T01:00:00'},))}).rows == ((datetime(2026,9,8,1),),)
    result = execute_query('SELECT v FROM t', tables={'t': SqlTable({'v': 'TIMESTAMPTZ'}, ({'v': '2026-09-08T01:00:00+03:00'},))})
    assert result.rows == ((datetime(2026,9,7,22,tzinfo=timezone.utc),),)
    assert execute_query('SELECT v FROM t', tables={'t': SqlTable({'v': 'TIMESTAMP'}, ())}).rows == ()


@pytest.mark.parametrize('unit,amount,expected', [('DAY',1,'2026-09-10'),('MONTH',1,'2026-10-09'),('YEAR',1,'2027-09-09')])
def test_integer_calendar_intervals_do_not_introduce_approximate_arithmetic(unit,amount,expected):
    from datetime import date
    result = execute_query(f"SELECT CAST(day + INTERVAL {amount} {unit} AS DATE) AS shifted FROM dates",
        tables={'dates':SqlTable({'day':'DATE'}, ({'day':date(2026,9,9)},))})
    assert result.rows == ((date.fromisoformat(expected),),)


def test_local_day_range_spans_dst_without_assuming_twenty_four_hours():
    from datetime import date, datetime
    rows = tuple({'at':datetime.fromisoformat(value)} for value in (
        '2026-03-08T05:00:00+00:00', '2026-03-09T03:30:00+00:00', '2026-03-09T04:00:00+00:00'))
    result = execute_query('SELECT COUNT(*) AS n FROM events WHERE at >= CAST($day AS TIMESTAMP) AND at < CAST($day AS TIMESTAMP) + INTERVAL 1 DAY',
        tables={'events':SqlTable({'at':'TIMESTAMPTZ'}, rows)}, parameters={'day':date(2026,3,8)}, timezone='America/New_York')
    assert result.rows == ((2,),)


@pytest.mark.parametrize('comparison,expected', [('<',Decimal('2')),('>',Decimal('10'))])
def test_filtered_average_preserves_independent_average_in_predicate(comparison,expected):
    result = execute_query(f'SELECT AVG(v) FILTER (WHERE v {comparison} (SELECT AVG(v) FROM t)) AS mean FROM t',
        tables={'t':SqlTable({'v':'INTEGER'}, tuple({'v':value} for value in (1,3,10)))})
    assert result.rows == ((expected,),)


def test_window_average_preserves_independent_average_in_partition():
    result = execute_query('SELECT v, AVG(v) OVER (PARTITION BY v > (SELECT AVG(v) FROM t)) AS mean FROM t ORDER BY v',
        tables={'t':SqlTable({'v':'INTEGER'}, tuple({'v':value} for value in (1,3,10)))})
    assert result.rows == ((1,Decimal('2')), (3,Decimal('2')), (10,Decimal('10')))
