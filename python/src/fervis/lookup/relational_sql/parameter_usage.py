"""Preserve scalar types and identity authorities at SQL parameter use sites."""

from sqlglot import exp, parse_one
from sqlglot.optimizer.scope import Scope, traverse_scope
from .column_usage import project_query
from .execution import QueryValidationError


def compatible_argument(parameter, description, *, literal_lookup=False):
    if description.get("kind") == "definition":
        return False
    identity = description.get("identity")
    target = parameter.get("entity_target")
    if target and not identity and not literal_lookup:
        return False
    if identity and target:
        return (
            identity["entity_kind"],
            identity["key_id"],
            description["projection"],
        ) == (
            target["entity_kind"],
            target["key_id"],
            "key_component:" + target["component_id"],
        )
    source_type = description.get("value_type", description.get("type"))
    target_type = parameter.get("type")
    aliases = {
        "choice": "string",
        "enum": "string",
        "int": "integer",
        "long": "integer",
        "numeric": "number",
        "decimal": "number",
        "float": "number",
        "double": "number",
        "bool": "boolean",
    }
    if (literal_lookup or not target) and target_type == 'uuid' and source_type == 'string':
        from fervis.lookup.plan_execution.declared_values import parse_declared_value
        try:
            parse_declared_value(description.get('value', description.get('label')), 'uuid')
        except (ValueError, TypeError):
            return False
        source_type = 'uuid'
    source_type = aliases.get(source_type, source_type)
    target_type = aliases.get(target_type, target_type)
    if source_type in {None, "unknown", "any"} or target_type in {
        None,
        "unknown",
        "any",
    }:
        return True
    return source_type == target_type or (source_type == "uuid" and target_type == "string") or (
        source_type == "integer" and target_type == "number"
    )


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
        for predicate in scope.find_all(exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.In):
            if predicate.find_ancestor(exp.Select) is not scope.expression:
                continue
            right = (predicate.args.get('query') or tuple(predicate.expressions)
                     if isinstance(predicate, exp.In) else predicate.expression)
            pairs = [(predicate.this, right), (right, predicate.this)]
            left_keys = [authorities.get(origin, set()) for origin in _expression_origins(scope, predicate.this, scope_index)]
            right_keys = [authorities.get(origin, set()) for origin in _expression_origins(scope, right, scope_index)]
            if any(left and right and left.isdisjoint(right) for left in left_keys for right in right_keys):
                raise QueryValidationError("SQL compares keys from different identity authorities; use a declared relationship")
            for column, operand in pairs:
                if column is None or operand is None:
                    continue
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


def validate_bound_program_parameters(program, bindings, row_sources, contracts):
    """Apply the same parameter checks to persisted and rebound SQL programs."""
    from dataclasses import asdict
    from fervis.lookup.answer_program.operations import SqlQuerySpec
    from fervis.lookup.answer_program.values import ParameterRef, ConstantRef
    from fervis.lookup.answer_program.relations import SourceKind
    from fervis.lookup.relation_catalog.row_sources.lookup import (
        row_source_for_relation,
    )
    from .parameters import projection_description

    queries = [
        operation.spec
        for operation in program.operations
        if isinstance(operation.spec, SqlQuerySpec)
    ]
    if not queries:
        return

    def describe(expression, target=None, relation_id=None):
        from fervis.lookup.answer_program.expressions import FieldRef
        if isinstance(expression, FieldRef):
            contract = contracts[relation_id]
            matches = [(key, component) for key in contract.entity_keys for component in key.components
                       if component.field_id == expression.field_id]
            if target:
                matches = [(key, component) for key, component in matches
                           if (key.entity_kind, key.key_id, component.component_id) ==
                           (target.entity_kind, target.key_id, target.component_id)]
            if matches:
                key, component = matches[0]
                return {'identity': {'entity_kind': key.entity_kind, 'key_id': key.key_id},
                        'projection': 'key_component:' + component.component_id,
                        'value_type': contract.field_types[expression.field_id]}
            return {'value_type': contract.field_types[expression.field_id]}
        if isinstance(expression, ParameterRef):
            binding = bindings.get(expression.parameter_id)
            value = binding.value if binding is not None else None
        elif isinstance(expression, ConstantRef):
            value = expression.value
        else:
            return {}
        return (
            projection_description(value, expression.component, expression.item_index)
            if value is not None
            else {}
        )

    parameter_inputs = {parameter.id: parameter.input_ref for parameter in program.parameters}
    lookup_relations = {}
    for spec in queries:
        if not spec.lookup_input_ref:
            continue
        if not any(isinstance(item.expression, ParameterRef) and
                   parameter_inputs.get(item.expression.parameter_id) == spec.lookup_input_ref
                   for item in spec.parameters):
            raise QueryValidationError('Literal lookup must consume its original declared input')
        for item in spec.inputs:
            lookup_relations.setdefault(item.relation_id, set()).add(spec.lookup_input_ref)

    for relation in program.relations:
        if relation.source.kind is not SourceKind.API_READ:
            continue
        source = row_source_for_relation(relation, row_sources=row_sources)
        for binding in relation.source.param_bindings:
            parameter = next(
                item for item in source.params if item.id == binding.param_id
            )
            literal_lookup = (isinstance(binding.value_expr, ParameterRef) and
                parameter_inputs.get(binding.value_expr.parameter_id) in lookup_relations.get(relation.id, set()))
            if not compatible_argument(asdict(parameter), describe(binding.value_expr, parameter.entity_target, relation.source.argument_relation_id), literal_lookup=literal_lookup):
                raise QueryValidationError(
                    "Request argument type or identity authority does not match the bound parameter"
                )
    for spec in queries:
        tables = {}
        for item in spec.inputs:
            contract = contracts[item.relation_id]
            keys = [
                {
                    "entity_kind": key.entity_kind,
                    "key_id": key.key_id,
                    "components": {component.component_id: column.name},
                }
                for key in contract.entity_keys
                for component in key.components
                for column in item.columns
                if column.field_id == component.field_id
            ]
            tables[item.name] = {
                "columns": {
                    column.name: {
                        "type": contract.field_types.get(column.field_id, "unknown")
                    }
                    for column in item.columns
                },
                "candidate_keys": keys,
            }
        validate_identity_key_uses(
            spec.query,
            tables,
            {item.name: describe(item.expression) for item in spec.parameters},
        )
