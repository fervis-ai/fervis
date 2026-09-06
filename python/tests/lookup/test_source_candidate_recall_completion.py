"""Structural feasibility must not discard later recalled evidence."""

from types import SimpleNamespace

from fervis.lookup.orchestration import semantic_compilation as module


def test_candidate_scope_reviews_all_recalled_batches_before_realization(monkeypatch):
    monkeypatch.setattr(module, "_bound_recall_selection", lambda selection, **kwargs: selection)
    first = SimpleNamespace(relation_catalog="first")
    second = SimpleNamespace(relation_catalog="second")
    catalog = SimpleNamespace(sources=(), relation_evidence=())
    catalog.select = lambda **kwargs: catalog
    reviewed = []
    monkeypatch.setattr(
        module,
        "combine_catalog_selection_batches",
        lambda batches, **kwargs: batches[-1],
    )
    monkeypatch.setattr(module, "build_row_source_catalog", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        module, "build_api_row_source_catalog", lambda *args, **kwargs: ()
    )
    monkeypatch.setattr(
        module, "combine_semantic_read_eligibility_results", lambda results: results
    )
    monkeypatch.setattr(
        module, "build_available_source_catalog", lambda *args, **kwargs: catalog
    )
    # Both batches are structurally feasible. Only the later one may contain the
    # semantically correct entity population; the model has not selected it yet.
    monkeypatch.setattr(
        module,
        "candidate_source_strategy",
        lambda *args: SimpleNamespace(branches=("feasible",)),
    )
    monkeypatch.setattr(
        module,
        "next_catalog_selection_batch",
        lambda catalog_selection, **kwargs: (
            second if catalog_selection is first else None
        ),
    )
    monkeypatch.setattr(
        module,
        "SemanticReadEligibilityRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        module,
        "_read_eligibility_turn",
        lambda eligibility_request, **kwargs: reviewed.append(
            eligibility_request.answer_catalog
        ),
    )
    result = module._prepare_source_candidates(
        initial_catalog_selection=first,
        initial_answer_sources=(),
        initial_eligibility=None,
        canonical_values=(),
        indexes=(object(),),
        context=None,
        request=SimpleNamespace(
            full_catalog=object(),
            memory_relations=(),
            run_id="test",
            max_catalog_reads_per_fact=1,
        ),
        on_turn=None,
    )
    assert reviewed == ["second"]
    assert result[0] is second
