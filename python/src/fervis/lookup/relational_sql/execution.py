"""Read-only DuckDB execution over current-run REST observations.

SQLGlot owns syntax validation only. DuckDB owns relational operators, casts,
null semantics and exact fixed-point arithmetic. Decimal division and AVG use
an explicit DECIMAL(38,18), ties-to-even policy instead of binary floats.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from threading import Timer
from typing import Any, Mapping

import json

import duckdb
from duckdb.func import FunctionNullHandling
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

class QueryValidationError(ValueError):
    """The query is outside the read-only, declared-view contract."""


@dataclass(frozen=True)
class SqlTable:
    columns: Mapping[str, str]
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class SqlResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    source_tables: tuple[str, ...]


_FUNCTIONS = frozenset({
    'AND', 'OR', 'EXISTS', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ABS', 'ROUND', 'LOWER', 'UPPER',
    'TRIM', 'CONCAT', 'CONCAT_WS', 'LENGTH', 'SUBSTRING', 'COALESCE', 'NULLIF',
    'IF', 'CASE', 'CAST', 'TRY_CAST', 'EXTRACT', 'STR_TO_TIME', 'DATE_STR_TO_DATE',
    'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'DATE_TRUNC', 'TIMESTAMP_TRUNC',
})


def _validate(query: str, tables: Mapping[str, SqlTable]) -> tuple[exp.Query, tuple[str, ...]]:
    try:
        statements = sqlglot.parse(query, read='duckdb')
    except SqlglotError as exc:
        raise QueryValidationError('Invalid relational query') from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise QueryValidationError('Exactly one SELECT query is required')
    statement = statements[0]
    for node in statement.walk():
        if isinstance(node, exp.DataType) and node.this in {exp.DType.FLOAT, exp.DType.DOUBLE}:
            raise QueryValidationError('Approximate numeric types are unsupported')
        if isinstance(node, (exp.DML, exp.DDL, exp.Command, exp.Into)):
            raise QueryValidationError('Query mutations are not allowed')
        if isinstance(node, exp.Func):
            name = node.name.upper() if isinstance(node, exp.Anonymous) else node.sql_name()
            if name not in _FUNCTIONS:
                if isinstance(node.parent,exp.Table) and name.lower() in {table.lower() for table in tables}:
                    raise QueryValidationError('Registered API views are SQL tables, not SQL functions; bind their REST parameters in api_invocations')
                raise QueryValidationError(f'Unsupported query function: {name}')
    sources = set()
    table_names = {name.casefold():name for name in tables}
    for scope in traverse_scope(statement):
        for _, source in scope.selected_sources.values():
            if not isinstance(source, exp.Table):
                continue
            if (not isinstance(source.this, exp.Identifier)
                or source.db or source.catalog or source.name.casefold() not in table_names):
                raise QueryValidationError('Query references an unregistered view')
            name = table_names[source.name.casefold()]
            source.set('this', exp.to_identifier(name, quoted=True))
            sources.add(name)
    return statement, tuple(sorted(sources))



_NUMERIC_CONTEXT = Context(prec=78, rounding=ROUND_HALF_EVEN)
_QUOTIENT_TYPE = 'DECIMAL(38,18)'
_QUANTUM = Decimal('0.000000000000000001')


def _divide(left: str | None, right: str | None) -> Decimal | None:
    if left is None or right is None:
        return None
    with localcontext(_NUMERIC_CONTEXT):
        numerator, denominator = Decimal(left), Decimal(right)
        if not denominator:
            return None
        quotient = numerator / denominator
        result = quotient.quantize(_QUANTUM)
        if quotient and not result:
            raise QueryValidationError('Decimal division underflow')
        if result.copy_abs() >= Decimal('1e20'):
            raise QueryValidationError('Decimal division overflow')
        return result


def _lower_numeric_operations(node: exp.Expr) -> exp.Expr:
    if isinstance(node, exp.Interval) and node.args.get('unit') is not None:
        value = node.this
        sign = 1
        if isinstance(value, exp.Neg):
            value, sign = value.this, -1
        if isinstance(value, exp.Literal):
            try:
                amount = Decimal(value.this) * sign
            except ArithmeticError:
                amount = None
            if amount is not None and amount.is_finite() and amount == amount.to_integral_value():
                # DuckDB expands INTERVAL n DAY through a DOUBLE/trunc macro.
                # The typed string form preserves an integer calendar amount
                # without weakening the numeric-plan checks for actual data.
                return exp.Cast(this=exp.Literal.string(f"{int(amount)} {node.args['unit'].name}"),
                                to=exp.DataType.build('INTERVAL'))
    if isinstance(node, exp.Literal) and node.is_number:
        value = Decimal(node.this)
        exponent = value.as_tuple().exponent
        if (not isinstance(exponent, int) or value.adjusted() >= 38 or exponent < -38
                or max(value.adjusted() + 1, 0) + max(-exponent, 0) > 38):
            raise QueryValidationError('Numeric literal exceeds fixed-point precision')
        return exp.Literal.number(format(value, 'f'))
    aggregate = node
    while isinstance(aggregate, (exp.Window, exp.Filter)):
        aggregate = aggregate.this
    if isinstance(aggregate, exp.Avg):
        # FILTER and OVER belong to each aggregate, never to the scalar quotient.
        def replace_average(kind):
            def replace_owned(expression):
                if isinstance(expression, (exp.Window, exp.Filter)):
                    wrapper = expression.copy()
                    wrapper.set('this', replace_owned(expression.this))
                    return wrapper
                return kind(this=expression.this.copy())
            return replace_owned(node)
        node = exp.Div(this=replace_average(exp.Sum), expression=replace_average(exp.Count))
    if isinstance(node, exp.Div):
        return exp.Anonymous(this='FERVIS_DECIMAL_DIVIDE', expressions=[
            exp.Cast(this=value.transform(_lower_numeric_operations), to=exp.DataType.build('VARCHAR'))
            for value in (node.this, node.expression)
        ])
    return node


def _identifier(name: str) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect='duckdb')


def _checked_type(raw: str) -> exp.DataType:
    kind = exp.DataType.build(raw, dialect='duckdb')
    if kind.this not in {exp.DType.BOOLEAN, exp.DType.TINYINT, exp.DType.SMALLINT,
                         exp.DType.INT, exp.DType.BIGINT, exp.DType.INT128,
                         exp.DType.DECIMAL, exp.DType.TEXT, exp.DType.VARCHAR,
                         exp.DType.DATE, exp.DType.TIMESTAMP, exp.DType.TIMESTAMPNTZ, exp.DType.TIMESTAMPTZ,
                         exp.DType.TIME}:
        raise QueryValidationError('Unsupported API-view column type')
    return kind


def _input_value(value: Any, kind: exp.DataType) -> Any:
    if value is None:
        return None
    if kind.this == exp.DType.DECIMAL:
        value = Decimal(str(value))
        if not value.is_finite():
            raise QueryValidationError('Non-finite API numeric value')
        dimensions = [int(item.this.this) for item in kind.expressions]
        precision, scale = (dimensions + [0])[:2] if dimensions else (18, 3)
        with localcontext(_NUMERIC_CONTEXT):
            if value != value.quantize(Decimal(1).scaleb(-scale)):
                raise QueryValidationError('API numeric value exceeds declared scale')
            if value.copy_abs() >= Decimal(10) ** (precision-scale):
                raise QueryValidationError('API numeric value exceeds declared precision')
    return value


def _verify_bound_numeric_types(connection, query: str) -> None:
    # Inspect the engine's bound plan before execution, including predicates,
    # casts and joins. Looking only at final cells cannot detect hidden float
    # conversions. This contract is exercised against the pinned engine version.
    plan = json.loads(connection.execute('SELECT json_serialize_plan(?)', [query]).fetchone()[0])
    if plan.get('error') or not isinstance(plan.get('plans'), list) or len(plan['plans']) != 1:
        raise QueryValidationError('Query could not produce a validated bound plan')
    pending = [plan]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if 'type_info' in node and node.get('id') in {'FLOAT', 'DOUBLE'}:
                raise QueryValidationError('Query contains approximate intermediate arithmetic')
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def execute_query(query: str, *, tables: Mapping[str, SqlTable],
                  parameters: Mapping[str, Any] | None = None,
                  timeout_seconds: float = 30, timezone: str = "UTC") -> SqlResult:
    timezone = query_timezone(timezone)
    if not 0 < timeout_seconds <= 30:
        raise QueryValidationError('Invalid query execution deadline')
    if len(query) > 100_000 or sum(len(table.rows) for table in tables.values()) > 200_000:
        raise QueryValidationError('Relational query resource limit exceeded')
    statement, sources = _validate(query, tables)
    from .column_usage import required_columns, ROW_PRESENCE_COLUMN
    required_columns(query,{name:table.columns for name,table in tables.items()})
    if len({name.casefold() for name in tables}) != len(tables):
        raise QueryValidationError('API view names collide')
    parameters = parameters or {}
    def bind(node: exp.Expr) -> exp.Expr:
        if isinstance(node, exp.Placeholder):
            if node.name not in parameters:
                raise QueryValidationError(f'Unknown query parameter: {node.name}')
            return exp.convert(parameters[node.name])
        return node
    lowered = statement.transform(bind).transform(_lower_numeric_operations)
    connection = duckdb.connect(':memory:', config={
        'enable_external_access': False,
        'autoload_known_extensions': False,
        'autoinstall_known_extensions': False,
        'python_enable_replacements': False,
        'threads': 1,
        'memory_limit': '256MB',
        'max_temp_directory_size': '0B',
    })
    timer = Timer(timeout_seconds, connection.interrupt)
    timer.daemon = True
    try:
        # ICU is bundled with the pinned Python wheel. External access and
        # automatic installation remain disabled during this trusted load.
        connection.execute('LOAD icu')
        connection.execute("SET TimeZone = ?", [timezone])
        connection.create_function('FERVIS_DECIMAL_DIVIDE', _divide,
            ['VARCHAR', 'VARCHAR'], duckdb.sqltype(_QUOTIENT_TYPE), null_handling=FunctionNullHandling.SPECIAL)
        for name, table in tables.items():
            types = {column: _checked_type(kind) for column, kind in table.columns.items()}
            if not types:
                types={ROW_PRESENCE_COLUMN:_checked_type('BOOLEAN')}
            definitions = ', '.join(f'{_identifier(column)} {kind.sql(dialect="duckdb")}'
                                    for column, kind in types.items())
            connection.execute(f'CREATE TABLE {_identifier(name)} ({definitions})')
            rows = []
            for row in table.rows:
                if not table.columns:
                    rows.append((True,))
                    continue
                if any(column not in row for column in types):
                    raise QueryValidationError('An API row is missing a declared column')
                rows.append(tuple(_input_value(row[column], kind) for column,kind in types.items()))
            if rows:
                placeholders = ', '.join('?' for _ in types)
                connection.executemany(f'INSERT INTO {_identifier(name)} VALUES ({placeholders})', rows)
        timer.start()
        query_sql = lowered.sql(dialect='duckdb', identify=True)
        _verify_bound_numeric_types(connection, query_sql)
        cursor = connection.execute(query_sql)
        columns = tuple(item[0] for item in cursor.description)
        result_rows = tuple(tuple(row) for row in cursor.fetchmany(200_001))
        if len(result_rows) > 200_000:
            raise QueryValidationError('Relational result row limit exceeded')
        if any(isinstance(value, float) for row in result_rows for value in row):
            raise QueryValidationError('Query produced an approximate numeric result')
        return SqlResult(columns, result_rows, sources)
    except (duckdb.Error, SqlglotError, InvalidOperation, ValueError, TypeError) as exc:
        raise QueryValidationError(f'Relational query could not be evaluated: {exc}') from exc
    finally:
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        connection.close()


def query_timezone(name):
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        return ZoneInfo(name).key
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise QueryValidationError('SQL timezone is not a supported timezone name') from exc
