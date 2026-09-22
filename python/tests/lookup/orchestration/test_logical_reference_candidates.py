"""Typed record candidates retain ambiguity when guarded as references."""

from dataclasses import replace

import pytest

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.reference_compilation import compile_reference_result
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.source_reads.access_execution import execute_access_program
from fervis.lookup.contract_codec import (
    canonical_answer_program_json,
    decode_answer_program,
)
from tests.lookup.orchestration.test_logical_record_outputs import _record_candidates


@pytest.mark.parametrize("collapse_occurrences", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_typed_candidate_projection_preserves_reference_cardinality(
    count, collapse_occurrences
):
    compiled, catalog = _record_candidates()
    answer = compiled.answer_program
    if collapse_occurrences:
        from fervis.lookup.answer_program.operations import ProjectSpec
        from fervis.lookup.answer_program.expressions import (
            FieldRef,
            FunctionExpression,
            ExpressionFunction,
        )

        operations = []
        for operation in answer.operations:
            if isinstance(operation.spec, ProjectSpec):
                outputs = tuple(
                    replace(item, expression=FieldRef("id"))
                    if isinstance(item.expression, FunctionExpression)
                    and item.expression.function is ExpressionFunction.ROW_NUMBER
                    else item
                    for item in operation.spec.outputs
                )
                operation = replace(
                    operation, spec=replace(operation.spec, outputs=outputs)
                )
            operations.append(operation)
        answer = replace(answer, operations=tuple(operations))
    from fervis.lookup.question_contract import InputTerm
    from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType

    subject = InputTerm(
        "reference_input",
        SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "the entry"),
        "the entry",
        TextType(),
    )
    answer = replace(answer, inputs=(subject,))
    reference = compile_reference_result(
        answer,
        bindings=compiled.initial_bindings,
        input_ref=subject.id,
        output_types={"id": "integer", "label": "string"},
        reference_id="reference",
    )
    assert reference.input_refs == (subject.id,)
    assert reference.view.columns == {"id": "id", "label": "label"}
    assert reference.table["candidate_keys"] == []
    program = AnswerProgram(
        parameters=reference.program.parameters,
        relations=reference.program.relations,
        operations=reference.program.operations,
    )
    program = decode_answer_program(canonical_answer_program_json(program))
    reads = []

    class Port:
        def read(self, **kwargs):
            reads.append(kwargs)
            return {
                "responseStatus": 200,
                "responseBody": [{"id": 1, "label": "A"} for _ in range(count)],
            }

    if collapse_occurrences:
        from fervis.lookup.plan_execution.errors import VerificationError

        with pytest.raises(VerificationError, match="cannot deduplicate"):
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
        assert result.relation("reference.rows").rows == ({"id": 1, "label": "A"},)
    else:
        assert result.issue.reference.reason.value == (
            "NOT_FOUND" if count == 0 else "AMBIGUOUS_RESULT"
        )
