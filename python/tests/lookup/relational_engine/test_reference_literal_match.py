"""Reference literals are matched by declared scalar type, without SQL or guessing."""

import pytest

from fervis.lookup.answer_program.expressions import (
    FieldRef,
    FunctionExpression,
    ExpressionFunction,
    expression_input_id,
)
from fervis.lookup.answer_program.operations import FilterSpec
from fervis.lookup.answer_program.values import ConstantRef, FactValue
from fervis.lookup.contract_codec import (
    canonical_contract_json,
    decode_canonical_contract,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
)


@pytest.mark.parametrize(
    "type_name,literal,rows,expected",
    [
        ("integer", "2", [1, 2, None], [2]),
        ("integer", "2 OR 1=1", [1, 2, None], []),
        ("string", "002", ["2", "002", None], ["002"]),
        ("string", "Alpha", ["Alpha", "alpha", None], ["Alpha"]),
        ("boolean", "false", [True, False, None], [False]),
    ],
)
def test_reference_literal_match_preserves_declared_value_semantics(
    type_name, literal, rows, expected
):
    expression = FunctionExpression(
        ExpressionFunction.REFERENCE_LITERAL_MATCH,
        (
            FieldRef("candidate"),
            ConstantRef(
                "supplied",
                "reference-text@1",
                FactValue.named(id="supplied", text=literal),
            ),
        ),
    )
    expression = decode_canonical_contract(
        canonical_contract_json(expression), FunctionExpression
    )
    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    "candidates",
                    tuple({"candidate": value} for value in rows),
                    field_types={"candidate": type_name},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                ),
            ),
            operations=(
                ExecutableOperation(
                    "match", FilterSpec("candidates", expression), "matched"
                ),
            ),
            scalar_inputs=(
                ScalarInput(
                    expression_input_id(expression.arguments[1]),
                    literal,
                    value_type="named",
                ),
            ),
        )
    )
    assert result.issue is None
    assert [row["candidate"] for row in result.relation("matched").rows] == expected


@pytest.mark.parametrize("type_name", ["any", "json"])
def test_unknown_or_collection_fields_cannot_be_used_as_literal_identifiers(type_name):
    from fervis.lookup.plan_execution.errors import RelationEngineError
    from fervis.lookup.plan_execution.operation_engine.expression_evaluator import (
        evaluate_expression,
        ExpressionEnvironment,
    )

    value = ConstantRef(
        "literal", "reference-text@1", FactValue.named(id="literal", text="2")
    )
    expression = FunctionExpression(
        ExpressionFunction.REFERENCE_LITERAL_MATCH, (FieldRef("candidate"), value)
    )
    with pytest.raises(RelationEngineError, match="declared scalar"):
        evaluate_expression(
            expression,
            environment=ExpressionEnvironment(
                row={"candidate": 2},
                field_types={"candidate": type_name},
                scalars={expression_input_id(value): "2"},
                scalar_types={expression_input_id(value): "named"},
            ),
        )
