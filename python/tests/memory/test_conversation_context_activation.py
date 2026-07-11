from __future__ import annotations

from fervis.memory.prior_requests import (
    PriorRequestMemory,
    PriorRequestOutput,
)
from fervis.memory.artifacts import FactOutcome, build_fact_artifact
from fervis.memory.conversation_context import (
    ConversationMemoryActivation,
    ConversationMemoryActivationKind,
    ConversationMemoryCard,
    ConversationMemoryCardProjection,
    expand_activated_memory_cards,
)


def test_prior_request_activation_uses_the_typed_projection() -> None:
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        source_question="How many sales did we make?",
    )
    memory_id = "turn_1.prior_request.fact_1"
    card = ConversationMemoryCard(
        card_id=memory_id,
        memory_id=memory_id,
        kind="prior_answer_request",
        display=artifact.source_question,
    )
    prior_request = PriorRequestMemory(
        memory_id=memory_id,
        artifact_id=artifact.artifact_id,
        request_id="fact_1",
        answer_fact="sales count",
        answer_shape=None,
        output_frames=(
            PriorRequestOutput(
                output_id="answer_1",
                description="sales count",
                role="ROW_POPULATION",
            ),
        ),
    )
    projection = ConversationMemoryCardProjection(
        cards=(card,),
        activations=(
            ConversationMemoryActivation(
                card=card,
                kind=ConversationMemoryActivationKind.PRIOR_REQUEST,
                artifact_id=artifact.artifact_id,
                prior_request=prior_request,
            ),
        ),
        prior_requests=(prior_request,),
        private_cards={
            memory_id: {
                "kind": "prior_answer_request",
                "artifact_id": artifact.artifact_id,
                "request_shape": {"answer_fact_template": "stale serialized shape"},
            }
        },
    )

    activated = expand_activated_memory_cards(
        artifacts=(artifact,),
        memory_projection=projection,
        used_memory_ids=(memory_id,),
    )

    assert activated.by_memory_id[memory_id] == {
        "kind": "prior_answer_request",
        "source_question": "How many sales did we make?",
        "request_shape": {
            "answer_fact_template": "sales count",
            "answer_outputs": (
                {
                    "output_id": "answer_1",
                    "description": "sales count",
                    "role": "ROW_POPULATION",
                },
            ),
            "slots": (),
            "semantic_parts": (),
        },
    }
