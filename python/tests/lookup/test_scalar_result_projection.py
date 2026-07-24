from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.result_projection import (
    ResultProjection,
    ScalarResultOutput,
)
from fervis.lookup.answer_rendering import render_fact_result
from fervis.lookup.outcomes.classification import classify_answer_result
from fervis.lookup.outcomes.model import FactResult
from fervis.lookup.plan_execution.operation_runtime import RelationEngineOutput


def test_scalar_result_keeps_runtime_identity_until_public_projection() -> None:
    program = AnswerProgram(
        result_projection=ResultProjection(
            scalar_outputs=(
                ScalarResultOutput(
                    id="public_output",
                    scalar_id="internal_scalar",
                    role="answer_value",
                ),
            ),
        )
    )

    result = classify_answer_result(
        program,
        engine_output=RelationEngineOutput(scalars={"internal_scalar": 7}),
    )

    assert isinstance(result, FactResult)
    assert result.outcome.scalars == {"internal_scalar": 7}
    assert render_fact_result(result).scalars == {"public_output": 7}
