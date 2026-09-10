"""Preserve scalar types and identity authorities at SQL parameter use sites."""

from sqlglot import exp, parse_one
from sqlglot.optimizer.scope import Scope, traverse_scope
from .column_usage import project_query
from .execution import QueryValidationError


def validate_sql_parameter_uses(query, tables, descriptions, *, lookup_input_ref=''):
    names = {node.name for node in parse_one(query, read='duckdb').find_all(exp.Placeholder)}
    if any(descriptions.get(name, {}).get('kind') == 'definition' or
           descriptions.get(name, {}).get('kind') == 'reference_literal' and
           (not lookup_input_ref or descriptions[name].get('input_ref') != lookup_input_ref)
           for name in names):
        raise QueryValidationError('Reference literals are REST resource-address values; SQL must use a typed reference slot')
    validate_identity_key_uses(query, tables, descriptions)


def validate_identity_key_uses(query, tables, parameters):
    keys = {
        name: description
        for name, description in parameters.items()
        if description.get("identity")
    }
    if not tables:
        return
    canonical, _ = project_query(
        query, {name: table.get("columns", {}) for name, table in tables.items()}
    )
    authorities = {}
    for view, table in tables.items():
        for key in table.get("candidate_keys", ()):
            for component, column in key["components"].items():
                authorities.setdefault((view.lower(), column.lower()), set()).add(
                    (key["entity_kind"], key["key_id"], component)
                )
        for key in table.get("entity_references", ()):
            for component, column in key["components"].items():
                authorities.setdefault((view.lower(), column.lower()), set()).add(
                    (key["target_entity_kind"], key["target_key_id"], component)
                )
    scopes = list(traverse_scope(parse_one(canonical, read="duckdb")))
    scope_index = {id(scope.expression): scope for scope in scopes}
    for scope in scopes:
        for predicate in scope.find_all(exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ, exp.In, exp.Between):
            if predicate.find_ancestor(exp.Select) is not scope.expression:
                continue
            right = ((predicate.args['low'], predicate.args['high']) if isinstance(predicate, exp.Between) else
                     (predicate.args.get('query') or tuple(predicate.expressions)
                      if isinstance(predicate, exp.In) else predicate.expression))
            pairs = [(predicate.this, right), (right, predicate.this)]
            left_keys = [authorities.get(origin, set()) for origin in _expression_origins(scope, predicate.this, scope_index)]
            right_keys = [authorities.get(origin, set()) for origin in _expression_origins(scope, right, scope_index)]
            if any(left and right and left.isdisjoint(right) for left in left_keys for right in right_keys):
                raise QueryValidationError("SQL compares keys from different identity authorities; use a declared relationship")
            for column, operand in pairs:
                if column is None or operand is None:
                    continue
                from fervis.lookup.plan_execution.declared_values import declared_comparison_types_compatible
                left_types = _unchanged_operand_types(scope, column, tables, parameters, scope_index)
                right_types = _unchanged_operand_types(scope, operand, tables, parameters, scope_index)
                for actual_type, actual_choice in left_types:
                    for expected_type, expected_choice in right_types:
                        if (actual_choice or expected_choice) and not declared_comparison_types_compatible(actual_type, expected_type):
                            raise QueryValidationError('Catalog interpretation type does not match its observed comparison field; preserve the literal input or explicitly cast the expression')
                for placeholder in _placeholders(operand):
                    if placeholder.find_ancestor(exp.Select) is not scope.expression or placeholder.name not in keys:
                        continue
                    description = keys[placeholder.name]
                    identity = description["identity"]
                    expected = (identity["entity_kind"], identity["key_id"],
                                description["projection"].removeprefix("key_component:"))
                    origins = _expression_origins(scope, column, scope_index)
                    if not origins or any(expected not in authorities.get(origin, set()) for origin in origins):
                        raise QueryValidationError("An identity key parameter must compare with its declared key or reference column")


def _placeholders(expression):
    if isinstance(expression, tuple):
        return (placeholder for item in expression for placeholder in _placeholders(item))
    return expression.find_all(exp.Placeholder)


def _expression_origins(scope, expression, scope_index):
    if isinstance(expression, tuple):
        return set().union(*(_expression_origins(scope, item, scope_index) for item in expression))
    if expression is None:
        return set()
    if isinstance(expression, exp.Column):
        return _column_origins(scope, expression, scope_index)
    if isinstance(expression, exp.Subquery):
        expression = expression.this
    if isinstance(expression, exp.Query):
        inner = scope_index.get(id(expression))
        return set().union(*(_projection_origins(inner, index, scope_index)
            for index in range(len(expression.selects)))) if inner else set()
    return set().union(*(_expression_origins(scope, child, scope_index) for child in expression.iter_expressions()))


def _column_origins(scope, column, scope_index):
    while scope is not None and column.table not in scope.sources:
        scope = scope.parent
    if scope is None:
        return set()
    source = scope.sources[column.table]
    if isinstance(source, exp.Table):
        return {(source.name.lower(), column.name.lower())}
    if not isinstance(source, Scope):
        return set()
    names = source.outer_columns or source.expression.named_selects
    if column.name not in names:
        return set()
    return _projection_origins(source, names.index(column.name), scope_index)


def _projection_origins(scope, index, scope_index):
    if scope.union_scopes:
        return set().union(*(_projection_origins(branch, index, scope_index) for branch in scope.union_scopes))
    return _expression_origins(scope, scope.expression.selects[index], scope_index)


def _unchanged_operand_types(scope, expression, tables, parameters, scope_index):
    """Trace values through projections; explicit computations own conversions."""
    if isinstance(expression, tuple):
        return tuple(origin for item in expression for origin in
                     _unchanged_operand_types(scope, item, tables, parameters, scope_index))
    while isinstance(expression, (exp.Paren, exp.Alias)):
        expression = expression.this
    if isinstance(expression, exp.Placeholder):
        description = parameters.get(expression.name, {})
        value_type = description.get('value_type') or description.get('type')
        return ((value_type, description.get('kind') == 'catalog_choice'),) if value_type else ()
    if isinstance(expression, exp.Subquery):
        expression = expression.this
    if isinstance(expression, exp.Query):
        inner = scope_index.get(id(expression))
        if inner is None or len(expression.selects) != 1:
            return ()
        return _unchanged_projection_types(inner, 0, tables, parameters, scope_index)
    if not isinstance(expression, exp.Column):
        return ()
    while scope is not None and expression.table not in scope.sources:
        scope = scope.parent
    if scope is None:
        return ()
    source = scope.sources[expression.table]
    if isinstance(source, exp.Table):
        table = next((value for name, value in tables.items() if name.casefold() == source.name.casefold()), {})
        definition = next((value for name, value in table.get('columns', {}).items()
                           if name.casefold() == expression.name.casefold()), {})
        return ((definition['type'], False),) if definition.get('type') else ()
    if not isinstance(source, Scope):
        return ()
    names = source.outer_columns or source.expression.named_selects
    if expression.name not in names:
        return ()
    return _unchanged_projection_types(source, names.index(expression.name), tables, parameters, scope_index)


def _unchanged_projection_types(scope, index, tables, parameters, scope_index):
    if scope.union_scopes:
        return tuple(origin for branch in scope.union_scopes for origin in
                     _unchanged_projection_types(branch, index, tables, parameters, scope_index))
    return _unchanged_operand_types(scope, scope.expression.selects[index], tables, parameters, scope_index)
