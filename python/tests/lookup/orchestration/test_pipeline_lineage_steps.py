from __future__ import annotations

from types import SimpleNamespace

from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.orchestration.pipeline import (
    _known_input_step_id,
)


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
