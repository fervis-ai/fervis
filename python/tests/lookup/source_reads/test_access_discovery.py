from dataclasses import replace
import pytest
from jsonschema import validate
from tests.lookup.relational_engine.test_dependent_reads import _program
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_discovery import (
    AccessDiscoveryRequest,
    ReadAccessTurnPrompt,
    access_schema,
    parse_read_access,
)
from fervis.lookup.turn_prompts import TurnPromptContext


def test_access_contract_is_question_blind_and_requires_a_correlated_mapping():
    _, _, catalog = _program(paired=True)
    sources = build_api_row_source_catalog(catalog)
    parent = next(
        source for source in sources.sources if source.read_id == "facilities"
    )
    child = next(
        source for source in sources.sources if source.read_id == "instruments"
    )
    request = AccessDiscoveryRequest(catalog, sources, (child,))
    payload = {
        "source_traversals": {
            child.id: {
                parent.id: {
                    "kind": "complete_traversal",
                    "coverage_basis": "Each target request identifies one parent row.",
                    "argument_fields": {
                        "facility_id": "facilities.id",
                        "zone": "facilities.zone",
                    },
                }
            }
        }
    }
    validate(payload, access_schema(request))
    access = parse_read_access(payload, request=request)
    assert access.can_enumerate(child)
    prompt = ReadAccessTurnPrompt(request).to_model_payload(
        TurnPromptContext(current_question="CONSUMER_SECRET")
    )
    assert "CONSUMER_SECRET" not in prompt.prompt_text
    assert "facilities.id" in prompt.prompt_text
    payload["source_traversals"][child.id][parent.id]["argument_fields"][
        "facility_id"
    ] = "facilities.zone"
    with pytest.raises(ValueError):
        parse_read_access(payload, request=request)


def test_candidate_failure_does_not_claim_a_complete_domain():
    _, _, catalog = _program()
    sources = build_api_row_source_catalog(catalog)
    parent = next(
        source for source in sources.sources if source.read_id == "facilities"
    )
    child = next(
        source for source in sources.sources if source.read_id == "instruments"
    )
    request = AccessDiscoveryRequest(catalog, sources, (child,))
    payload = {
        "source_traversals": {
            child.id: {
                parent.id: {
                    "kind": "not_a_complete_traversal",
                    "coverage_basis": "No complete domain is declared.",
                }
            }
        }
    }
    access = parse_read_access(payload, request=request)
    assert not access.can_enumerate(child)


def test_pending_dependent_read_is_assessed_before_executable_filtering(monkeypatch):
    from types import SimpleNamespace
    from fervis.lookup.orchestration import semantic_compilation as compilation
    from fervis.lookup.relation_catalog.selection.model import (
        CatalogSelectionResult, RequestedFactCatalogSelection,
    )
    from fervis.lookup.relation_catalog.model import RelationCatalog

    _, _, catalog = _program()
    selection = CatalogSelectionResult(
        RelationCatalog(reads=(catalog.read("facilities"),)),
        (RequestedFactCatalogSelection("fact_1", ("instruments",), (),
                                       ("facilities",), ("instruments",)),),
        ("facilities",),
    )
    calls = []

    def model_turn(purpose, *, prompt, parse, **kwargs):
        calls.append(purpose)
        sources = build_api_row_source_catalog(catalog)
        parent, child = sources.sources
        return SimpleNamespace(result=parse({"source_traversals": {
            child.id: {parent.id: {
                "kind": "complete_traversal",
                "coverage_basis": "Every instrument belongs to one listed facility.",
                "argument_fields": {"facility_id": "facilities.id"},
            }}
        }}))

    monkeypatch.setattr(compilation, "_turn", model_turn)
    access = compilation._discover_catalog_read_access(
        selection, resolver_catalog=RelationCatalog(reads=()),
        request=SimpleNamespace(full_catalog=catalog),
        context=TurnPromptContext(current_question="How many instruments?"),
        on_turn=None,
    )
    bounded = compilation._bound_recall_selection(
        selection, full_catalog=catalog, values=(), read_access=access,
    )
    assert len(calls) == 1
    assert bounded.requested_fact_selections[0].unselected_positive_read_ids == (
        "instruments",
    )
