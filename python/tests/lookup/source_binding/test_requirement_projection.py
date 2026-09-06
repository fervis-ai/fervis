"""Source selection must receive the declared graph, not reconstruct its edges."""

from fervis.lookup.source_binding.prompt import SemanticSourceRealizationTurnPrompt
from tests.lookup.relational_engine.test_scoped_compilation import employee_query


def test_realization_preserves_association_endpoint_roles():
    request = employee_query().request
    payload = SemanticSourceRealizationTurnPrompt(request)._requirements_payload()
    expected = {
        ref.token: (
            request.index.fact_local_ref_by_local_id[term.from_set_ref].token,
            request.index.fact_local_ref_by_local_id[term.to_set_ref].token,
        )
        for ref in request.index.association_requirement_refs
        for term in (request.index.term_by_ref[ref],)
    }
    assert expected
    assert {
        item["requirement_ref"]: (item.get("from_set_ref"), item.get("to_set_ref"))
        for item in payload["associations"]
    } == expected
