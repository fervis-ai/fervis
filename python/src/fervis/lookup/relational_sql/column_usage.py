"""Resolve SQL field dependencies without inferring API or business meaning."""
from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.errors import SqlglotError

ROW_PRESENCE_COLUMN='__fervis_row_presence__'


def required_columns(query, columns_by_view):
    return project_query(query, columns_by_view)[1]


def project_query(query, columns_by_view):
    from .execution import QueryValidationError
    names={name.lower():name for name in columns_by_view}
    columns={name:{column.lower():column for column in values} for name,values in columns_by_view.items()}
    schema={name:{column:'TEXT' for column in values} or {ROW_PRESENCE_COLUMN:'BOOLEAN'}
            for name,values in columns_by_view.items()}
    try:
        statement=qualify(parse_one(query,read='duckdb'),dialect='duckdb',schema=schema)
    except SqlglotError as exc:
        raise QueryValidationError('SQL query references an undeclared column') from exc
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
                raise QueryValidationError('SQL query references an undeclared column')
            used[view].add(name)
    return statement.sql(dialect="duckdb"), {name:frozenset(fields) for name,fields in used.items()}
