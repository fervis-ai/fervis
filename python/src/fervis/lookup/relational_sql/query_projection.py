"""Lower explicit public fields and SQL sort expressions into one projection."""
from sqlglot import exp, parse_one
from .column_usage import project_query
from .execution import QueryValidationError
from .results import ResultOrder


def lower_query_projection(query, columns_by_view, *, public_columns=()):
    normalized, _ = project_query(query, columns_by_view)
    statement = parse_one(normalized, read='duckdb')

    def append(expression, name):
        distinct = statement.args.get('distinct')
        if not isinstance(statement, exp.Select) or distinct and not distinct.args.get('on'):
            raise QueryValidationError('Hidden projections through DISTINCT or set operations require an explicit outer SELECT')
        statement.append('expressions', exp.alias_(expression.copy(), name, quoted=True))

    selected = {name.casefold() for name in statement.named_selects}
    for name in public_columns:
        if name.casefold() not in selected:
            append(exp.column(name, quoted=True), name)
            selected.add(name.casefold())
    normalized, _ = project_query(statement.sql(dialect='duckdb'), columns_by_view)
    statement = parse_one(normalized, read='duckdb')
    order = statement.args.get('order')
    if order is None:
        return normalized, ()
    projections = {item.alias_or_name:item for item in statement.selects}
    occupied = {node.name.casefold() for node in statement.find_all(exp.Identifier)}
    occupied.update(column.casefold() for columns in columns_by_view.values() for column in columns)
    ordering = []
    for position, item in enumerate(order.expressions, start=1):
        if item.args.get('nulls_first'):
            raise QueryValidationError('Canonical result ordering requires NULLS LAST')
        expression = item.this
        name = None
        if isinstance(expression, exp.Literal) and expression.is_int:
            ordinal = int(expression.this)
            if not 1 <= ordinal <= len(projections):
                raise QueryValidationError('SQL ordering position is outside the result projection')
            name = tuple(projections)[ordinal-1]
        elif isinstance(expression, exp.Column) and not expression.table:
            name = next((column for column in projections if column.casefold() == expression.name.casefold()), None)
        if name is None:
            def body(projection):
                return projection.this if isinstance(projection, exp.Alias) else projection
            name = next((column for column,projection in projections.items()
                         if body(projection) == expression), None)
        if name is None:
            name = f'__fervis_order_{position}'
            while name.casefold() in occupied:
                name += '_'
            append(expression, name)
            occupied.add(name.casefold())
        ordering.append(ResultOrder(name, bool(item.args.get('desc'))))
    distinct = statement.args.get('distinct')
    if isinstance(distinct, exp.Distinct) and distinct.args.get('on'):
        # DISTINCT ON consumes ordering to choose observations, before public ranking.
        names = tuple(statement.named_selects)
        statement = exp.select(*(exp.column(name, quoted=True) for name in names)).from_(
            exp.Subquery(this=statement, alias=exp.TableAlias(this=exp.to_identifier('__fervis_selected__', quoted=True))))
    else:
        statement.set('order', None)
    normalized, _ = project_query(statement.sql(dialect='duckdb'), columns_by_view)
    return normalized, tuple(ordering)
