from fervis.lookup.source_reads.access_model import ReadAccessCatalog
import pytest
from jsonschema import validate
from tests.lookup.relational_engine.test_dependent_reads import _program
from fervis.lookup.relation_catalog import CatalogField, RelationCatalog
from dataclasses import replace
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_discovery import (
    AccessDiscoveryRequest,
    ReadAccessTurnPrompt,
    access_candidates,
    access_schema,
    parse_read_access,
)
from fervis.lookup.turn_prompts import TurnPromptContext


def test_unannotated_path_key_reviews_semantically_related_parent_before_type_only_candidate():
    from fervis.lookup.relation_catalog import CatalogParam, EndpointRead, RowPath, RowCardinality

    unrelated = EndpointRead(
        "a_orders", "orders", path="/orders", resource_names=("orders",),
        row_paths=(RowPath("root", "", RowCardinality.MANY),),
        fields=(CatalogField("orders.order_id", "uuid", path="order_id", row_path_id="root"),),
    )
    related = EndpointRead(
        "z_facilities", "facilities", path="/facilities", resource_names=("facilities",),
        row_paths=(RowPath("root", "", RowCardinality.MANY),),
        fields=(CatalogField("facilities.id", "uuid", path="id", row_path_id="root"),),
    )
    target = EndpointRead(
        "readings", "readings", path="/readings/{facility_id}",
        row_paths=(RowPath("root", "", RowCardinality.MANY),),
        fields=(CatalogField("readings.value", "number", path="value", row_path_id="root"),),
        params=(CatalogParam("facility_id", "facility_id", "path", "uuid", required=True),),
    )
    catalog = RelationCatalog(reads=(unrelated, related, target))
    sources = build_api_row_source_catalog(catalog)
    child = next(source for source in sources.sources if source.read_id == "readings")

    assert [parent.read_id for parent in access_candidates(
        child, sources=sources, catalog=catalog
    )] == ["z_facilities", "a_orders"]


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


def test_access_prompt_shows_only_fields_that_can_supply_an_argument():
    _, _, catalog = _program(paired=True)
    parent_read = catalog.read("facilities")
    parent_read = replace(parent_read, fields=(
        *parent_read.fields,
        CatalogField("unrelated_boolean", "boolean", path="unrelated_boolean",
                     row_path_id="root", metadata={"description": "UNRELATED_FIELD_MARKER"}),
    ))
    catalog = RelationCatalog(reads=(parent_read, *(
        read for read in catalog.reads if read.id != parent_read.id
    )))
    sources = build_api_row_source_catalog(catalog)
    child = next(source for source in sources.sources if source.read_id == "instruments")
    request = AccessDiscoveryRequest(catalog, sources, (child,))
    prompt = ReadAccessTurnPrompt(request).to_model_payload(
        TurnPromptContext(current_question="How many instruments?")
    )
    assert "facilities.id" in prompt.prompt_text
    assert "facilities.zone" in prompt.prompt_text
    assert "UNRELATED_FIELD_MARKER" not in prompt.prompt_text


def test_selected_dependent_read_discovers_complete_parent_traversal(monkeypatch):
    from types import SimpleNamespace
    from fervis.lookup.orchestration import semantic_compilation as compilation

    _, _, catalog = _program()
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
    access = compilation._discover_read_access(
        ("instruments",),
        request=SimpleNamespace(full_catalog=catalog, read_access=ReadAccessCatalog()),
        context=TurnPromptContext(current_question="How many instruments?"),
        on_turn=None,
    )
    child = next(source for source in build_api_row_source_catalog(catalog).sources if source.read_id == "instruments")
    assert len(calls) == 1
    assert access.can_enumerate(child)
