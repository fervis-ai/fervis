from dataclasses import replace
import pytest

from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from fervis.lookup.question_contract.model import Quantifier
from fervis.lookup.query_enrichment.semantic import (
    SemanticQueryEnrichmentRequest,
    semantic_recall_buckets,
    semantic_recall_requirements,
)
from fervis.lookup.query_enrichment.semantic_prompt import (
    SemanticQueryEnrichmentTurnPrompt,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context


def test_recall_prompt_carries_meanings_for_its_dependency_references():
    index = employee_query(Quantifier.NOT_EXISTS).request.index
    request = SemanticQueryEnrichmentRequest(
        semantic_recall_buckets(index),
        (),
        ("employee", "manager"),
        requirements=semantic_recall_requirements(index),
    )
    invocation = SemanticQueryEnrichmentTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question="Employees without a qualifying manager.",
            conversation_context={},
        )
    )
    assert "manager_salary" in invocation.prompt.prompt_text
    assert "owner_refs" in invocation.prompt.prompt_text
    assert "employee" in invocation.prompt.prompt_text
    with pytest.raises(ValueError, match="requirement definitions"):
        SemanticQueryEnrichmentTurnPrompt(replace(request, requirements=()))
