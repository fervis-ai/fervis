"""A guarded reference can publish a query-local occurrence without inventing a key."""

from dataclasses import replace

import pytest

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.operations import (
    Operation,
    FilterSpec,
    ProjectSpec,
    ReferenceGuardSpec,
    NamedExpression,
)
from fervis.lookup.answer_program.expressions import (
    FieldRef,
    FunctionExpression,
    ExpressionFunction,
)
from fervis.lookup.answer_program.values import ConstantRef, FactValue
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.source_reads.access_execution import execute_access_program
from fervis.lookup.contract_codec import (
    canonical_answer_program_json,
    decode_answer_program,
)
from tests.lookup.orchestration.test_logical_record_outputs import _record_candidates


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("forged_occurrence", [False, True])
def test_reference_guard_preserves_source_occurrence_and_literal_match_on_replay(
    count, forged_occurrence
):
    compiled, catalog = _record_candidates()
    answer = compiled.answer_program
    source = answer.result_projection.relation_outputs[0].relation_id
    projection = next(
        op for op in answer.operations if isinstance(op.spec, ProjectSpec)
    )
    occurrence = next(
        item.output_field
        for item in projection.spec.outputs
        if isinstance(item.expression, FunctionExpression)
        and item.expression.function is ExpressionFunction.ROW_NUMBER
    )
    alias = Operation(
        "reference.scope",
        ProjectSpec(
            source,
            (
                NamedExpression("reference_label", FieldRef("label")),
                NamedExpression(
                    "reference_occurrence",
                    FieldRef("id" if forged_occurrence else occurrence),
                ),
            ),
        ),
        "reference.scoped",
    )
    selected = Operation(
        "reference.select",
        FilterSpec(
            alias.output_relation,
            FunctionExpression(
                ExpressionFunction.REFERENCE_LITERAL_MATCH,
                (
                    FieldRef("reference_label"),
                    ConstantRef(
                        "lookup",
                        "literal-reference@1",
                        FactValue.named(id="lookup", text="A"),
                    ),
                ),
            ),
        ),
        "reference.selected",
    )
    guard = Operation(
        "reference.guard",
        ReferenceGuardSpec(
            selected.output_relation,
            ("reference_label", "reference_occurrence"),
            "i1",
            occurrence_fields=("reference_occurrence",),
        ),
        "reference.rows",
    )
    program = AnswerProgram(
        parameters=answer.parameters,
        relations=answer.relations,
        operations=(*answer.operations, alias, selected, guard),
    )
    program = decode_answer_program(canonical_answer_program_json(program))
    reads = []

    class Port:
        def read(self, **kwargs):
            reads.append(kwargs)
            return {
                "responseStatus": 200,
                "responseBody": [
                    {"id": 9, "label": "B"},
                    *[{"id": 1, "label": "A"} for _ in range(count)],
                ],
            }

    if forged_occurrence:
        from fervis.lookup.plan_execution.errors import VerificationError

        with pytest.raises(VerificationError, match="carried row number"):
            execute_access_program(
                program, catalog=catalog, read_session=ApiReadSession(Port())
            )
        assert reads == []
        return
    result = execute_access_program(
        program, catalog=catalog, read_session=ApiReadSession(Port())
    ).engine_output
    if count == 1:
        assert result.issue is None
        assert result.relation("reference.rows").rows == (
            {"reference_label": "A", "reference_occurrence": 2},
        )
    else:
        assert result.issue.reference.reason.value == (
            "NOT_FOUND" if count == 0 else "AMBIGUOUS_RESULT"
        )
    assert len(reads) == 1
