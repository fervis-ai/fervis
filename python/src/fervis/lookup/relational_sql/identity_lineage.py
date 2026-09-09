"""Verify SQL identity columns against canonical input key contracts."""

from sqlglot import exp, parse_one
from sqlglot.lineage import lineage
from sqlglot.errors import SqlglotError
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.verification.contract_types import (
    RelationEntityKey,
    RelationEntityKeyComponent,
)


def sql_identity_keys(spec, contracts):
    if not spec.entity_keys:
        return ()
    schema = {
        item.name: {column.name: "TEXT" for column in item.columns}
        for item in spec.inputs
        if item.columns
    }
    try:
        nodes = lineage(
            None, spec.query, schema=schema, dialect="duckdb", trim_selects=False
        )
    except (SqlglotError, ValueError) as exc:
        raise VerificationError("SQL identity output has no field lineage") from exc
    from .acquisition import sql_value_type

    input_types = {
        (item.name.lower(), column.name.lower()): contracts[
            item.relation_id
        ].field_types.get(column.field_id)
        for item in spec.inputs
        for column in item.columns
    }
    output_types = {item.id.lower(): item.value_type for item in spec.outputs}
    result = []
    for projection in spec.entity_keys:
        component_ids = {item.component_id for item in projection.components}
        allowed = {component: set() for component in component_ids}
        memberships = {component: {} for component in component_ids}
        for item in spec.inputs:
            for key_index, key in enumerate(contracts[item.relation_id].entity_keys):
                if (
                    key.entity_kind != projection.entity_kind
                    or key.key_id != projection.key_id
                    or {component.component_id for component in key.components}
                    != component_ids
                ):
                    continue
                for component in key.components:
                    allowed[component.component_id].update(
                        (item.name.lower(), column.name.lower())
                        for column in item.columns
                        if column.field_id == component.field_id
                    )
                    for column in item.columns:
                        if column.field_id == component.field_id:
                            memberships[component.component_id].setdefault(
                                (item.name.lower(), column.name.lower()), set()
                            ).add(key_index)
        occurrence_sets = []
        key_choices = []
        for component in projection.components:
            node = nodes.get(component.field_id.lower())
            if node is None:
                raise VerificationError(
                    "SQL identity component is not an output column"
                )
            origins = _origins(node)
            if not origins or any(
                (view, column) not in allowed[component.component_id]
                for view, column, _ in origins
            ):
                raise VerificationError(
                    "SQL identity component does not preserve a declared key field"
                )
            target_type = output_types[component.field_id.lower()]
            if any(
                input_types[view, column] is None
                or sql_value_type(input_types[view, column]) != target_type
                for view, column, _ in origins
            ):
                raise VerificationError(
                    "SQL identity output type must preserve its source key type"
                )
            occurrence_sets.append(frozenset((view, path) for view, _, path in origins))
            choices = {}
            for view, column, path in origins:
                choices.setdefault((view, path), set()).update(
                    memberships[component.component_id][view, column]
                )
            key_choices.append(choices)
        if any(occurrences != occurrence_sets[0] for occurrences in occurrence_sets):
            raise VerificationError(
                "SQL identity components come from different record occurrences"
            )
        for occurrence in occurrence_sets[0]:
            if not set.intersection(*(choices[occurrence] for choices in key_choices)):
                raise VerificationError(
                    "SQL identity components must preserve one declared key tuple"
                )
        result.append(
            RelationEntityKey(
                projection.entity_kind,
                projection.key_id,
                tuple(
                    RelationEntityKeyComponent(item.component_id, item.field_id)
                    for item in projection.components
                ),
            )
        )
    return tuple(dict.fromkeys(result))


def _origins(node, path=()):
    expression = node.expression
    while isinstance(expression, exp.Alias):
        expression = expression.this
    if isinstance(expression, exp.Table):
        column = parse_one(node.name, into=exp.Column, dialect="duckdb").name
        return {(expression.name, column, (*path, expression.alias_or_name))}
    if not isinstance(expression, exp.Column):
        raise VerificationError(
            "SQL identity components must preserve key values without computation"
        )
    next_path = (*path, expression.table) if expression.table else path
    # A projected column can have alternative producers without an explicit
    # set-operation node: SQLGlot flattens UNIONs behind CTE references.
    return {
        origin
        for index, child in enumerate(node.downstream)
        for origin in _origins(
            child,
            (*next_path, ("set_branch", index))
            if len(node.downstream) > 1
            else next_path,
        )
    }
