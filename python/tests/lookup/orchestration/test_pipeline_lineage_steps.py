from __future__ import annotations

from types import SimpleNamespace

from fervis.lookup.continuations import (
    ContinuationCarriedInput,
    ContinuationPlan,
    ContinuationPlanKind,
    ContinuationReplacement,
)
from fervis.memory.prior_requests import PriorTimeScopeBinding
from fervis.lookup.conversation_resolution import ConversationInputProvenanceSet
from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.orchestration.pipeline import (
    _conversation_resolution_activation,
    _known_input_step_id,
)
from fervis.memory.conversation_context import ConversationReplaceablePart


def test_known_input_step_id_uses_question_contract_when_grounding_turn_did_not_run() -> None:
    calls: list[tuple[str, int]] = []

    class Sink:
        def model_turn_step_id(self, *, purpose: str, turn: int) -> str:
            calls.append((purpose, turn))
            return f"step:{purpose}:{turn}"

    state = SimpleNamespace(
        grounding=SimpleNamespace(turn=None),
        ports=SimpleNamespace(lineage_step_sink=Sink(), lineage_required=False),
        conversation_turn=None,
    )

    assert _known_input_step_id(state) == "step:question_contract:1"
    assert calls == [(ModelTurnPurpose.QUESTION_CONTRACT, 1)]


def test_known_input_step_id_uses_grounding_when_grounding_turn_ran() -> None:
    calls: list[tuple[str, int]] = []

    class Sink:
        def model_turn_step_id(self, *, purpose: str, turn: int) -> str:
            calls.append((purpose, turn))
            return f"step:{purpose}:{turn}"

    state = SimpleNamespace(
        grounding=SimpleNamespace(turn=object()),
        ports=SimpleNamespace(lineage_step_sink=Sink(), lineage_required=False),
        conversation_turn=None,
    )

    assert _known_input_step_id(state) == "step:grounding:3"
    assert calls == [(ModelTurnPurpose.GROUNDING, 3)]


def test_conversation_resolution_activation_includes_continuation_provenance() -> None:
    state = SimpleNamespace(
        conversation_resolution=SimpleNamespace(
            activation_payload=lambda: {
                "activated_memory_ids": ["turn_sales.value.q_time"]
            }
        ),
        continuation_plan=ContinuationPlan(
            kind=ContinuationPlanKind.SAME_FACT_INPUT_REPLACEMENT,
            current_question="What about Mombasa?",
            resolved_request_text="sales count in Mombasa this month",
            frame_id="context_frame_1",
            prior_answer_fact="sales count in Nairobi this month",
            replacements=(
                ContinuationReplacement(
                    part=ConversationReplaceablePart(
                        part_id="q_place",
                        kind="entity_identity",
                        text="Nairobi",
                    ),
                    current_text="Mombasa",
                ),
            ),
            carried_inputs=(
                ContinuationCarriedInput(
                    part=ConversationReplaceablePart(
                        part_id="q_time",
                        kind="time_scope",
                        text="this month",
                    ),
                    resolved_value_text="this month",
                    binding=PriorTimeScopeBinding(
                        value="this month",
                        display="this month",
                        resolved_start="2026-07-01",
                        resolved_end="2026-07-31",
                        granularity="month",
                        source_lineage=("turn_sales.value.q_time",),
                    ),
                ),
            ),
        ),
        conversation_input_provenance=ConversationInputProvenanceSet(),
    )

    activation = _conversation_resolution_activation(state)

    assert activation["activated_memory_ids"] == ["turn_sales.value.q_time"]
    assert activation["continuation"]["resolved_request_text"] == (
        "sales count in Mombasa this month"
    )
    assert activation["continuation"]["replacements"] == [
        {
            "part_id": "q_place",
            "kind": "entity_identity",
            "text": "Nairobi",
            "prior_text": "Nairobi",
            "current_text": "Mombasa",
        }
    ]
    assert activation["continuation"]["carried_inputs"][0]["binding"][
        "source_lineage"
    ] == ["turn_sales.value.q_time"]
