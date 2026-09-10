from types import SimpleNamespace

from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.available_sources import SourceContractSnapshot
from fervis.lookup.orchestration.semantic_compilation import (
    SemanticCompilationImpossible,
)
from fervis.lookup.orchestration import pipeline
from fervis.lookup.answer_rendering import render_fact_result, rendered_fact_text
from fervis.lookup.outcomes.model import Impossible
from tests.lookup.source_binding._candidate_fixture import daily_observation_request


def test_exhausted_source_search_returns_explained_impossibility_with_source_evidence(
    monkeypatch,
):
    request = daily_observation_request()
    contract = QuestionContract(
        inputs=(), requested_facts=(request.index.requested_fact,)
    )
    snapshot = SourceContractSnapshot.from_content("{}")
    outcome = SemanticCompilationImpossible(
        contract, (), ("fact_1",), snapshot, ("reviewed_read",)
    )
    state = SimpleNamespace(
        request=object(), ports=object(), semantic_usage={"inputTokens": 100}
    )
    captured = {}
    monkeypatch.setattr(
        pipeline, "_synthesize_result", lambda **kwargs: captured.update(kwargs)
    )
    pipeline._semantic_compilation_impossible_result(state, outcome)
    assert captured["status"] == "COMPLETED"
    assert captured["usage"] == {"inputTokens": 100}
    fact = captured["fact_result"]
    assert isinstance(fact.outcome, Impossible)
    assert fact.outcome.blocked_requirements[0].reviewed_read_ids == ("reviewed_read",)
    assert fact.outcome.proof_refs == (snapshot.ref,)
    message = rendered_fact_text(render_fact_result(fact))
    assert request.index.requested_fact.origin.meaning in message
    assert "available API evidence" in message
