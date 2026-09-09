"""Invalid closed SQL literals fail during authoring, before REST acquisition."""
import pytest

from fervis.lookup.relational_sql.authoring import parse_query_answer
from fervis.lookup.relational_sql.execution import QueryValidationError
from tests.lookup.relational_sql.test_authoring import payload


@pytest.mark.parametrize('expression',[
    "CAST('2026-03-01 00:00:00 Africa/Nairobi' AS TIMESTAMP)",
    "CAST('2026-02-30' AS DATE)",
    "CAST('not-a-number' AS INTEGER)",
    "CAST(('2026-02-30') AS DATE)",
    "CAST(-129 AS TINYINT)",
])
def test_authoring_rejects_invalid_literal_cast_even_without_api_rows(expression):
    with pytest.raises(QueryValidationError,match='literal cast'):
        parse_query_answer(payload(query=f'SELECT COUNT(*) AS total FROM items WHERE {expression} IS NOT NULL'),
            table_names={'items'},parameter_names=set())


@pytest.mark.parametrize('expression',[
    "CAST('2026-03-01 00:00:00+03:00' AS TIMESTAMPTZ)",
    "CAST('2026-03-01 00:00:00 Africa/Nairobi' AS TIMESTAMPTZ)",
    "TRY_CAST('not-a-number' AS INTEGER)",
    "CAST((-128) AS TINYINT)",
    "CAST('1000000000 days' AS INTERVAL)",
    "CAST($p1 AS TIMESTAMP)",
])
def test_authoring_preserves_valid_casts_and_runtime_bound_values(expression):
    parse_query_answer(payload(query=f'SELECT COUNT(*) AS total FROM items WHERE {expression} IS NOT NULL'),
        table_names={'items'},parameter_names={'p1'})


def test_large_valid_interval_is_not_materialized_as_a_python_intermediate():
    from fervis.lookup.relational_sql.execution import execute_query
    assert execute_query("SELECT CAST('1000000000 days' AS INTERVAL) IS NOT NULL AS ok",tables={}).rows==((True,),)
