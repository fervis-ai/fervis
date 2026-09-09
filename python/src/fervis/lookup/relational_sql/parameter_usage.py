"""Preserve scalar types and identity authorities at SQL parameter use sites."""

from sqlglot import exp, parse_one
from sqlglot.optimizer.scope import Scope, traverse_scope
from .column_usage import project_query
from .execution import QueryValidationError


def compatible_argument(parameter, description):
    if description.get("kind") == "definition":
        return False
    identity = description.get("identity")
    target = parameter.get("entity_target")
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
        "uuid": "string",
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
    if source_type in {None, "unknown", "any"} or target_type in {
        None,
        "unknown",
        "any",
    }:
        return True
    return source_type == target_type or (
        source_type == "integer" and target_type == "number"
    )


def validate_identity_key_uses(query, tables, parameters):
    keys = {
        name: description
        for name, description in parameters.items()
        if description.get("identity")
    }
    if not keys or not tables:
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
    for scope in traverse_scope(parse_one(canonical, read="duckdb")):
        for predicate in scope.find_all(
            exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.In
        ):
            pairs = [
                (predicate.this, predicate.expression),
                (predicate.expression, predicate.this),
            ]
            if isinstance(predicate, exp.In):
                pairs = [(predicate.this, predicate)]
            for column, operand in pairs:
                if not isinstance(column, exp.Column) or operand is None:
                    continue
                for placeholder in operand.find_all(exp.Placeholder):
                    if placeholder.name not in keys:
                        continue
                    description = keys[placeholder.name]
                    identity = description["identity"]
                    expected = (
                        identity["entity_kind"],
                        identity["key_id"],
                        description["projection"].removeprefix("key_component:"),
                    )
                    origins = _column_origins(scope, column)
                    if not origins or any(
                        expected not in authorities.get(origin, set())
                        for origin in origins
                    ):
                        raise QueryValidationError(
                            "An identity key parameter must compare with its declared key or reference column"
                        )


def _column_origins(scope, column):
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
    return _projection_origins(source, names.index(column.name))


def _projection_origins(scope, index):
    if scope.union_scopes:
        return set().union(
            *(_projection_origins(branch, index) for branch in scope.union_scopes)
        )
    expression = scope.expression.selects[index]
    while isinstance(expression, exp.Alias):
        expression = expression.this
    return (
        _column_origins(scope, expression)
        if isinstance(expression, exp.Column)
        else set()
    )


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

    def describe(expression):
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

    for relation in program.relations:
        if relation.source.kind is not SourceKind.API_READ:
            continue
        source = row_source_for_relation(relation, row_sources=row_sources)
        for binding in relation.source.param_bindings:
            parameter = next(
                item for item in source.params if item.id == binding.param_id
            )
            if not compatible_argument(asdict(parameter), describe(binding.value_expr)):
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
