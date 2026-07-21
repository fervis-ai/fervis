from __future__ import annotations

from fervis.lookup.answer_program.values import LiteralType, LiteralValuePayload
from fervis.lookup.grounding.resolution.deterministic import (
    _deterministic_fact_values,
)
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
)


def _contract(*inputs: RequestedFactLiteralInput) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="requested count",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer_1",
                        role="ROW_COUNT",
                        description="count",
                    ),
                ),
                known_inputs=inputs,
            ),
        )
    )


def _literal(
    input_id: str,
    value: str,
    *,
    role: LiteralInputRole,
) -> RequestedFactLiteralInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=value,
        resolved_value_text=value,
        role=role,
    )


def test_predicate_and_threshold_values_are_grounded_without_identity_tasks() -> None:
    ledger = _deterministic_fact_values(
        _contract(
            _literal(
                "qi_state",
                "finished",
                role=LiteralInputRole.PREDICATE_VALUE,
            ),
            _literal(
                "qi_amount",
                "1000.00",
                role=LiteralInputRole.THRESHOLD_VALUE,
            ),
        ),
        runtime_values=None,
    )

    assert [(value.known_input_id, value.kind.value) for value in ledger.values] == [
        ("qi_state", "literal"),
        ("qi_amount", "literal"),
    ]
    state, amount = (value.payload for value in ledger.values)
    assert isinstance(state, LiteralValuePayload)
    assert state.literal_type is LiteralType.STRING
    assert state.value == "finished"
    assert isinstance(amount, LiteralValuePayload)
    assert amount.literal_type is LiteralType.NUMBER
    assert amount.value == "1000"


def test_reference_value_is_not_deterministically_grounded_as_a_scalar() -> None:
    ledger = _deterministic_fact_values(
        _contract(
            _literal(
                "qi_resource",
                "named resource",
                role=LiteralInputRole.REFERENCE_VALUE,
            )
        ),
        runtime_values=None,
    )

    assert ledger.values == ()
