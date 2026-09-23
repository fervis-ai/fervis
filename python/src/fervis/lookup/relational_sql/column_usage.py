"""Resolve SQL field dependencies without inferring API or business meaning."""
import json
from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.errors import SqlglotError

ROW_PRESENCE_COLUMN='__fervis_row_presence__'


def required_columns(query, columns_by_view):
    return project_query(query, columns_by_view)[1]


def project_query(query, columns_by_view, *, output_columns=None):
    from .execution import QueryValidationError
    names={name.lower():name for name in columns_by_view}
    columns={name:{column.lower():column for column in values} for name,values in columns_by_view.items()}
    schema={name:{column:'TEXT' for column in values} or {ROW_PRESENCE_COLUMN:'BOOLEAN'}
            for name,values in columns_by_view.items()}
    try:
        parsed = parse_one(query, read='duckdb')
    except SqlglotError as exc:
        raise QueryValidationError('Invalid SQL query: ' + str(exc)) from exc
    if parsed is None:
        raise QueryValidationError('SQL query is empty')
    try:
        statement=qualify(parsed.copy(),dialect='duckdb',schema=schema)
    except SqlglotError as exc:
        raise QueryValidationError(_column_diagnostic(parsed, columns_by_view, str(exc))) from exc
    if output_columns is not None:
        require_output_columns(statement.named_selects, output_columns)
    used={name:set() for name in columns_by_view}
    for scope in traverse_scope(statement):
        for column in scope.columns:
            owner=scope
            source=None
            while owner is not None:
                if column.table in owner.sources:
                    source=owner.sources[column.table]
                    break
                owner=owner.parent
            if not isinstance(source,exp.Table) or source.name.lower() not in names:
                continue
            view=names[source.name.lower()]
            name=columns[view].get(column.name.lower())
            if name is None:
                raise QueryValidationError(_column_diagnostic(parsed, columns_by_view, f'{view}.{column.name} is not declared'))
            used[view].add(name)
    return statement.sql(dialect="duckdb"), {name:frozenset(fields) for name,fields in used.items()}


def _column_diagnostic(statement, columns_by_view, detail):
    referenced = {table.name.casefold() for table in statement.find_all(exp.Table)}
    query_columns = {column.name.casefold() for column in statement.find_all(exp.Column)}
    selected = {view: sorted(columns) for view, columns in columns_by_view.items() if view.casefold() in referenced}
    alternatives = {view: sorted(column for column in columns if column.casefold() in query_columns)
                    for view, columns in columns_by_view.items() if view.casefold() not in referenced
                    and any(column.casefold() in query_columns for column in columns)}
    return ('SQL query references an undeclared column: ' + detail + '. Declared columns in selected views: '
            + json.dumps(selected, sort_keys=True) + '. Other views containing referenced column names: '
            + json.dumps(alternatives, sort_keys=True))


def require_output_columns(actual, declared):
    from .execution import QueryValidationError
    actual, declared = tuple(actual), tuple(declared)
    if set(actual) != set(declared) or len(set(actual)) != len(actual) or len(set(declared)) != len(declared):
        raise QueryValidationError(f'SQL result columns do not match the program declaration. Outer query columns: {actual}; declared columns: {declared}. CTE and subquery columns are not outer result columns.')
