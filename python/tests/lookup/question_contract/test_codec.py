from __future__ import annotations

from fervis.lookup.contract_codec import (
    canonical_contract_json,
    decode_canonical_contract,
)
from fervis.lookup.question_contract import QuestionContract
from tests.testkit.semantic_question_contracts import semantic_question_contract


def test_semantic_question_contract_round_trips_with_concrete_types() -> None:
    contract = semantic_question_contract(
        requested_fact_id="fact_1",
        output_ids=("output_1",),
        description="tasks",
    )

    encoded = canonical_contract_json(contract)
    decoded = decode_canonical_contract(encoded, QuestionContract)

    assert decoded == contract
    assert type(decoded.requested_facts[0].selection) is type(
        contract.requested_facts[0].selection
    )
