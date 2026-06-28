from __future__ import annotations

from fervis.lookup.question_contract import (
    KnownInputKind,
    KnownInputSource,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerOutput,
    RequestedFactKnownInput,
)


def test_requested_fact_answer_output_serializes_description_only():
    output = RequestedFactAnswerOutput(
        id="answer_1",
        description="the amount she made yesterday",
    )
    fact = RequestedFact(
        id="fact_1",
        description="total sales amount for Alice yesterday",
        answer_outputs=(output,),
    )

    assert output.to_model_dict() == {
        "id": "answer_1",
        "description": "the amount she made yesterday",
    }
    assert fact.answer_request_model_dict()["answer_outputs"] == [
        {
            "description": "the amount she made yesterday",
        }
    ]


def test_question_contract_model_materializes_input_refs_as_fact_known_inputs():
    period = RequestedFactKnownInput(
        id="period",
        kind=KnownInputKind.TIME,
        source=KnownInputSource.QUESTION_CONTEXT,
        text="yesterday",
    )
    contract = QuestionContract(
        question_inputs=(period,),
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales yesterday",
                answer_expression=RequestedFactAnswerExpression(
                    family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
                ),
                answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
                input_refs=("period",),
            ),
        ),
    )

    assert contract.requested_facts[0].known_inputs == (period,)


def test_question_contract_serializes_fact_local_known_inputs_as_question_inputs():
    period = RequestedFactKnownInput(
        id="period",
        kind=KnownInputKind.TIME,
        source=KnownInputSource.QUESTION_CONTEXT,
        text="yesterday",
    )
    contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales yesterday",
                answer_expression=RequestedFactAnswerExpression(
                    family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
                ),
                answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
                known_inputs=(period,),
            ),
        ),
    )

    assert contract.to_model_dict() == {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "answer_requests": [
            {
                "id": "fact_1",
                "answer_fact": "sales yesterday",
                "answer_expression": {"family": "scalar_aggregate"},
                "input_requirements": {"time_requirements": []},
                "answer_outputs": [
                    {
                        "description": "answer_1",
                    }
                ],
                "input_decisions": [
                    {
                        "input_ref": "period",
                        "use_input": True,
                    }
                ],
            }
        ],
        "question_inputs": [
            {
                "id": "period",
                "kind": "time_text",
                "source": "question_context",
                "text": "yesterday",
            }
        ],
    }
