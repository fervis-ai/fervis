from __future__ import annotations

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
        details={
            "semantic_frame": {
                "parts": [
                    {"part_id": "subject", "kind": "subject", "text": "sales"}
                ]
            }
        },
    )
    projection = ConversationMemoryCardProjection(
        cards=(card,),
        activations=(
            ConversationMemoryActivation(
                card=card,
                kind=ConversationMemoryActivationKind.PRIOR_REQUEST,
                artifact_id=artifact.artifact_id,
            ),
        ),
        private_cards={
            memory_id: {
                "kind": "prior_answer_request",
                "artifact_id": artifact.artifact_id,
                "semantic_frame": {"parts": []},
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
        "semantic_frame": {
            "parts": [
                {"part_id": "subject", "kind": "subject", "text": "sales"}
            ]
        },
    }
