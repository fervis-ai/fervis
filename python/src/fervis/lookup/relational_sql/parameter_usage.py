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
    if identity and not target and description.get('kind') != 'field_projection':
        return False
    if description.get('kind') == 'reference_literal' and not literal_lookup and not (
        parameter.get('source') == 'path' or parameter.get('type') == 'uuid'
    ):
        return False
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
    source_type = aliases.get(source_type, source_type)
    target_type = aliases.get(target_type, target_type)
    if source_type == 'string' and target_type in {'integer', 'number', 'boolean', 'date', 'datetime', 'time', 'uuid'} and (
        literal_lookup or (not target and description.get('kind') == 'reference_literal')
    ):
        from fervis.lookup.plan_execution.declared_values import parse_declared_value
        try:
            parse_declared_value(description.get('value', description.get('label')), target_type)
        except (ValueError, TypeError):
            return False
        source_type = target_type
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

    producers = {operation.output_relation:operation for operation in program.operations if operation.output_relation}
    def guarded(relation_id):
        operation = producers.get(relation_id)
        if operation is None:
            return False
        return (isinstance(operation.spec, SqlQuerySpec) and bool(operation.spec.reference_input_ref)
                or any(guarded(ref) for ref in operation.input_relation_ids))
    from fervis.lookup.question_contract.model import InputDenotationKind
    denotations = {item.input_ref:item for item in getattr(program, 'input_denotations', ())}
    parameter_inputs = {parameter.id:parameter.input_ref for parameter in program.parameters}
    def describe(expression, target=None, relation_id=None, literal_lookup=False):
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
                return {'kind':'reference_argument' if guarded(relation_id) else 'field_projection',
                        'identity': {'entity_kind': key.entity_kind, 'key_id': key.key_id},
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
        description = projection_description(value, expression.component, expression.item_index) if value is not None else {}
        input_ref = parameter_inputs.get(expression.parameter_id) if isinstance(expression, ParameterRef) else None
        denotation = denotations.get(input_ref)
        if denotation is not None and denotation.kind is InputDenotationKind.IDENTITY_REFERENCE and not description.get('identity'):
            description = {**description, 'input_ref':input_ref, 'kind':'reference_literal' if literal_lookup or not denotation.reference_descriptions else 'definition'}
        return description

    parameter_inputs = {parameter.id: parameter.input_ref for parameter in program.parameters}
    lookup_relations = {}
    for spec in queries:
        if not spec.lookup_input_ref:
            continue
        if not any(isinstance(item.expression, ParameterRef) and
                   parameter_inputs.get(item.expression.parameter_id) == spec.lookup_input_ref
                   for item in spec.parameters):
            raise QueryValidationError('Literal lookup must consume its original declared input')
        denotation = denotations.get(spec.lookup_input_ref)
        if denotation is not None and any(
            describe(item.expression, literal_lookup=True).get('value') in denotation.reference_descriptions
            for item in spec.parameters if isinstance(item.expression, ParameterRef)
            and parameter_inputs.get(item.expression.parameter_id) == spec.lookup_input_ref
        ):
            raise QueryValidationError('A descriptive reference cannot be rebound as a literal lookup')
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
            if not compatible_argument(asdict(parameter), describe(binding.value_expr, parameter.entity_target, relation.source.argument_relation_id, literal_lookup=literal_lookup), literal_lookup=literal_lookup):
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
        validate_sql_parameter_uses(spec.query, tables,
            {item.name: describe(item.expression, literal_lookup=(isinstance(item.expression, ParameterRef) and
                parameter_inputs.get(item.expression.parameter_id) == spec.lookup_input_ref)) for item in spec.parameters},
            lookup_input_ref=spec.lookup_input_ref)
