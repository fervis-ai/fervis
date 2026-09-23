"""Evaluate all dynamic arguments together for each upstream relation row."""

from __future__ import annotations
from typing import Any
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.relations import Relation
from fervis.lookup.plan_execution.relations import RelationRows, CompletenessStatus
from fervis.lookup.plan_execution.operation_engine.expression_evaluator import (
    ExpressionEnvironment,
    evaluate_expression,
)
from fervis.lookup.outcomes.errors import IncompleteEvidenceError
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.relation_catalog.row_sources import RowSource
from fervis.lookup.relation_catalog.parameter_values import (
    catalog_parameter_wire_value,
    parse_catalog_parameter_value,
)
from fervis.lookup.canonical_data import canonical_runtime_json


def argument_rows(
    relation: Relation,
    source: RowSource,
    parent: RelationRows,
    static_args: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    if parent.completeness.status is not CompletenessStatus.COMPLETE:
        raise IncompleteEvidenceError(
            relation_id=parent.id, proof_refs=parent.completeness.proof_refs
        )
    expressions = tuple(
        binding
        for binding in relation.source.param_bindings
        if isinstance(binding.value_expr, FieldRef)
    )
    if not expressions:
        raise VerificationError(
            "argument relation requires row-valued request arguments"
        )
    rows: dict[str, dict[str, Any]] = {}
    for row in parent.rows:
        args = dict(static_args)
        environment = ExpressionEnvironment(row=row, field_types=parent.field_types)
        for binding in expressions:
            param = source.param(binding.param_id)
            value = evaluate_expression(
                binding.value_expr, environment=environment
            ).value
            if value is None:
                raise VerificationError("dependent request argument is null")
            args[param.param_ref] = parse_catalog_parameter_value(
                catalog_parameter_wire_value(value, type_name=param.type.value),
                type_name=param.type.value,
                choices=param.choices,
            )
        rows[canonical_runtime_json(args)] = args
    return tuple(rows.values())
