"""Compile literal matching over complete, observed candidate values."""

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from .execution import QueryValidationError


def reject_reference_truncation(query):
    try:
        statement = parse_one(query, dialect='duckdb')
    except SqlglotError as exc:
        raise QueryValidationError('Invalid reference query') from exc
    if statement is None:
        raise QueryValidationError('Reference query is empty')
    if any(isinstance(node, (exp.Limit, exp.Offset, exp.RowNumber, exp.TableSample))
           or isinstance(node, exp.Distinct) and node.args.get('on') is not None
           for node in statement.walk()):
        raise QueryValidationError('Reference candidates cannot be truncated; requested extrema must preserve ties')


def literal_match_query(query, *, column, parameter, tables):
    try:
        statement = parse_one(query, dialect='duckdb')
        _candidate_projection(statement, column=column, parameter=parameter, tables=tables)
        candidates = statement.subquery('__fervis_reference_candidates')
        return exp.select('*').from_(candidates).where(exp.EQ(
            this=exp.column(column, quoted=True),
            expression=exp.Placeholder(this=parameter),
        )).sql(dialect='duckdb')
    except (SqlglotError, ValueError) as exc:
        if isinstance(exc, QueryValidationError):
            raise
        raise QueryValidationError('Literal matching requires a complete observed candidate projection') from exc


def _candidate_projection(statement, *, column, parameter, tables):
    if statement.args.get('with_'):
        raise QueryValidationError('Literal candidate projections cannot contain auxiliary queries')
    if isinstance(statement, exp.Union):
        _candidate_projection(statement.this, column=column, parameter=parameter, tables=tables)
        _candidate_projection(statement.expression, column=column, parameter=parameter, tables=tables)
        return
    if not isinstance(statement, exp.Select):
        raise QueryValidationError('Literal matching requires candidate projections')
    if any(statement.args.get(key) for key in (
        'joins', 'group', 'having', 'qualify', 'order', 'limit', 'offset', 'windows',
    )):
        raise QueryValidationError('Literal candidate selection belongs to the compiler, not the authored query')
    source = statement.args.get('from_')
    table = source.this if source is not None else None
    if not isinstance(table, exp.Table) or table.name not in tables:
        raise QueryValidationError('Literal matching requires a declared candidate view')
    expression = next((item for item in statement.expressions if item.alias_or_name.lower() == column.lower()), None)
    if expression is None:
        raise QueryValidationError('Literal matching requires a declared observed column')
    value = expression.this if isinstance(expression, exp.Alias) else expression
    if not list(value.find_all(exp.Column)):
        raise QueryValidationError('Literal matching requires an observed value')
    normalized = _observed_expression(value, table=table, columns=tables[table.name]['columns'])
    where = statement.args.get('where')
    if where is not None:
        condition = where.this.unnest()
        observed = None
        if isinstance(condition, exp.EQ):
            for left, right in ((condition.this, condition.expression), (condition.expression, condition.this)):
                if isinstance(right, exp.Placeholder) and right.name == parameter:
                    observed = _observed_expression(left, table=table, columns=tables[table.name]['columns'])
        if observed is None or _normalized_columns(observed) != _normalized_columns(normalized):
            raise QueryValidationError('Literal candidate predicates must exactly repeat the compiler-owned match')
        # The outer, null-preserving equality entails this duplicate condition.
        # Removing it does not change the complete set of matching identities.
        statement.set('where', None)
    if isinstance(expression, exp.Alias):
        expression.set('this', normalized)
    else:
        expression.replace(exp.alias_(normalized, column, quoted=True))


def _observed_expression(node, *, table, columns):
    if isinstance(node, exp.Column):
        if node.name.casefold() not in {name.casefold() for name in columns} or node.table and node.table.casefold() != table.alias_or_name.casefold():
            raise QueryValidationError('Literal matching requires an observed candidate field')
        return node.copy()
    if isinstance(node, exp.Literal) and node.is_string and not any(char.isalnum() for char in node.this):
        return node.copy()
    if isinstance(node, (exp.Concat, exp.DPipe)):
        parts = node.expressions if isinstance(node, exp.Concat) else (node.this, node.expression)
        normalized = [_observed_expression(part, table=table, columns=columns) for part in parts]
        if not normalized:
            raise QueryValidationError('Literal formatting requires observed components')
        result = normalized[0]
        for part in normalized[1:]:
            result = exp.DPipe(this=result, expression=part)
        return result
    if isinstance(node, (exp.Paren, exp.Lower, exp.Upper)):
        result = node.copy()
        result.set('this', _observed_expression(node.this, table=table, columns=columns))
        return result
    raise QueryValidationError('Literal matching permits observed fields and null-preserving formatting, not invented values or fallback branches')


def verify_reference_candidate_completeness(program):
    """Recheck persisted reference ancestry before any source read."""
    from fervis.lookup.answer_program.operations import SqlQuerySpec
    from fervis.lookup.plan_execution.errors import VerificationError
    producers = {operation.output_relation: operation for operation in program.operations if operation.output_relation}
    pending = [operation for operation in program.operations
               if isinstance(operation.spec, SqlQuerySpec) and operation.spec.reference_input_ref]
    seen = set()
    while pending:
        operation = pending.pop()
        if operation.id in seen:
            continue
        seen.add(operation.id)
        if isinstance(operation.spec, SqlQuerySpec):
            try:
                reject_reference_truncation(operation.spec.query)
            except QueryValidationError as exc:
                raise VerificationError(str(exc)) from exc
        pending.extend(producers[relation] for relation in operation.input_relation_ids if relation in producers)


def _normalized_columns(expression):
    return expression.transform(lambda node: exp.column(node.name.casefold()) if isinstance(node, exp.Column) else node)
