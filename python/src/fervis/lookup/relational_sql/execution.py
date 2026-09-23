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
    casts = [node for node in statement.walk() if type(node) is exp.Cast and _literal_operand(node.this)]
    if casts:
        with _connection() as connection:
            try:
                connection.execute('SELECT '+', '.join('('+node.sql(dialect='duckdb')+') IS NULL' for node in casts)).fetchall()
            except duckdb.Error as exc:
                raise QueryValidationError(f'Invalid SQL literal cast: {exc}') from exc
    return statement, tuple(sorted(sources))



def _literal_operand(node):
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Neg):
        node = node.this
        while isinstance(node, exp.Paren):
            node = node.this
        return isinstance(node, exp.Literal) and node.is_number
    return isinstance(node, (exp.Literal, exp.Boolean, exp.Null))


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
                         exp.DType.TIME, exp.DType.UUID}:
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


def _connection():
    connection = duckdb.connect(':memory:', config={
        'enable_external_access': False,
        'autoload_known_extensions': False,
        'autoinstall_known_extensions': False,
        'python_enable_replacements': False,
        'threads': 1,
        'memory_limit': '256MB',
        'max_temp_directory_size': '0B',
    })
    # ICU is bundled with the pinned wheel; installation and external access stay disabled.
    try:
        connection.execute('LOAD icu')
    except BaseException:
        connection.close()
        raise
    return connection


def execute_query(query: str, *, tables: Mapping[str, SqlTable],
                  parameters: Mapping[str, Any] | None = None,
                  parameter_types: Mapping[str, str] | None = None,
                  timeout_seconds: float = 30, timezone: str = "UTC") -> SqlResult:
    timezone = query_timezone(timezone)
    if not 0 < timeout_seconds <= 30:
        raise QueryValidationError('Invalid query execution deadline')
    if len(query) > 100_000 or sum(len(table.rows) for table in tables.values()) > 200_000:
        raise QueryValidationError('Relational query resource limit exceeded')
    statement, sources = _validate(query, tables)
    from .column_usage import required_columns
    required_columns(query,{name:table.columns for name,table in tables.items()})
    if len({name.casefold() for name in tables}) != len(tables):
        raise QueryValidationError('API view names collide')
    lowered = _bind_scalar_parameters(statement, parameters or {}, parameter_types=parameter_types).transform(_lower_numeric_operations)
    connection = _query_connection(timezone)
    timer = Timer(timeout_seconds, connection.interrupt)
    timer.daemon = True
    try:
        _load_tables(connection, tables)
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


def _load_tables(connection, tables, *, include_rows=True):
    from .column_usage import ROW_PRESENCE_COLUMN
    for name, table in tables.items():
        types = {column: _checked_type(kind) for column, kind in table.columns.items()}
        if not types:
            types={ROW_PRESENCE_COLUMN:_checked_type('BOOLEAN')}
        definitions = ', '.join(f'{_identifier(column)} {kind.sql(dialect="duckdb")}'
                                for column, kind in types.items())
        connection.execute(f'CREATE TABLE {_identifier(name)} ({definitions})')
        rows = []
        for row in table.rows if include_rows else ():
            if not table.columns:
                rows.append((True,))
                continue
            if any(column not in row for column in types):
                raise QueryValidationError('An API row is missing a declared column')
            rows.append(tuple(_input_value(row[column], kind) for column,kind in types.items()))
        if rows:
            placeholders = ', '.join('?' for _ in types)
            connection.executemany(f'INSERT INTO {_identifier(name)} VALUES ({placeholders})', rows)


def _query_connection(timezone):
    connection = _connection()
    try:
        connection.execute("SET TimeZone = ?", [timezone])
        connection.create_function('FERVIS_DECIMAL_DIVIDE', _divide,
            ['VARCHAR', 'VARCHAR'], duckdb.sqltype(_QUOTIENT_TYPE), null_handling=FunctionNullHandling.SPECIAL)
        return connection
    except BaseException:
        connection.close()
        raise


def describe_query(query: str, *, tables: Mapping[str, SqlTable],
                   parameter_types: Mapping[str, str] | None = None,
                   parameter_values: Mapping[str, Any] | None = None,
                   timezone: str = 'UTC') -> dict[str, str]:
    """Bind a stable SQL projection against declared types without observing rows."""
    if len(query) > 100_000:
        raise QueryValidationError('Relational query resource limit exceeded')
    statement, _ = _validate(query, tables)
    from .column_usage import required_columns
    required_columns(query, {name:table.columns for name,table in tables.items()})
    if len({name.casefold() for name in tables}) != len(tables):
        raise QueryValidationError('API view names collide')
    for select in statement.find_all(exp.Select):
        if any(not isinstance(item, exp.Alias) and tuple(item.find_all(exp.Placeholder))
               for item in select.expressions):
            raise QueryValidationError('Parameter-dependent SQL projections require stable aliases')
    parameters = parameter_types or {}
    occupied = {node.name.casefold() for node in statement.find_all(exp.Identifier)} | {name.casefold() for name in tables}
    parameter_table = '__fervis_parameters__'
    while parameter_table.casefold() in occupied:
        parameter_table += '_'
    def bind(node):
        if isinstance(node, exp.Placeholder):
            if node.name not in parameters:
                raise QueryValidationError(f'Unknown query parameter: {node.name}')
            return exp.Subquery(this=exp.select(exp.column(node.name, quoted=True)).from_(
                exp.Table(this=exp.to_identifier(parameter_table, quoted=True))))
        return node
    constants = {name:value for name,value in (parameter_values or {}).items() if value is not None}
    if not constants.keys() <= parameters.keys():
        raise QueryValidationError('Grounded SQL constants require declared parameter types')
    lowered = _bind_scalar_parameters(statement, constants, parameter_types=parameters, missing=bind).transform(_lower_numeric_operations)
    with _query_connection(query_timezone(timezone)) as connection:
        timer = Timer(30, connection.interrupt)
        timer.daemon = True
        timer.start()
        try:
            _load_tables(connection, {**tables, **({parameter_table:SqlTable(parameters, ())} if parameters else {})},
                         include_rows=False)
            sql = lowered.sql(dialect='duckdb', identify=True)
            _verify_bound_numeric_types(connection, sql)
            description = connection.execute('DESCRIBE '+sql).fetchall()
        except duckdb.Error as exc:
            raise QueryValidationError(f'SQL schema could not be bound: {exc}') from exc
        finally:
            timer.cancel()
            timer.join()
    result: dict[str, str] = {}
    domains = {'BOOLEAN':'boolean', 'TINYINT':'integer', 'SMALLINT':'integer',
               'INTEGER':'integer', 'BIGINT':'integer', 'HUGEINT':'integer',
               'UTINYINT':'integer', 'USMALLINT':'integer', 'UINTEGER':'integer',
               'UBIGINT':'integer', 'UHUGEINT':'integer', 'VARCHAR':'string',
               'DATE':'date', 'TIMESTAMP':'datetime', 'TIMESTAMP WITH TIME ZONE':'datetime',
               'UUID':'uuid'}
    for name, kind, *_ in description:
        if not name or name.casefold() in {column.casefold() for column in result}:
            raise QueryValidationError('SQL output names must be nonempty and unique')
        domain = 'number' if kind.startswith('DECIMAL(') else domains.get(kind)
        if domain is None:
            raise QueryValidationError(f'SQL output has an unsupported type: {kind}')
        result[name] = domain
    if not result:
        raise QueryValidationError('SQL must return at least one column')
    return result


def _bind_scalar_parameters(statement, parameters, *, parameter_types=None, missing=None):
    """Grounded scalar constants have the same SQL semantics in both phases."""
    def bind(node):
        if isinstance(node, exp.Placeholder):
            if node.name in parameters:
                from uuid import UUID
                value = parameters[node.name]
                literal = (exp.Cast(this=exp.Literal.string(str(value)), to=exp.DataType.build('UUID'))
                           if isinstance(value, UUID) else exp.convert(value))
                kind = (parameter_types or {}).get(node.name, '')
                return exp.Cast(this=literal, to=_checked_type(kind)) if kind.startswith('DECIMAL(') else literal
            if missing is not None:
                return missing(node)
            raise QueryValidationError(f'Unknown query parameter: {node.name}')
        return node
    return statement.transform(bind)
