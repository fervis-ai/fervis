from dataclasses import replace

from jsonschema import validate

from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceContractSnapshot,
    SourceRelationEvidence,
)
from fervis.lookup.plan_selection.semantic import SemanticPlanSelectionRequest
from fervis.lookup.plan_selection.semantic_parser import parse_semantic_plan_selection
from fervis.lookup.plan_selection.semantic_schema import (
    build_semantic_plan_selection_schema,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from tests.lookup.grounding._fixtures import _staff_read
from tests.lookup.read_eligibility.test_semantic_read_eligibility import (
    _semantic_contract,
)


def test_direct_alignment_deterministically_produces_singleton_strategy() -> None:
    request, source_refs = _request(source_count=1)
    [source_ref] = source_refs
    payload = _payload(
        request,
        alignments={source_ref: "DIRECT"},
    )

    validate(payload, build_semantic_plan_selection_schema(request))
    [strategy] = parse_semantic_plan_selection(payload, request=request)

    assert strategy.source_assessments[0].alignment.value == "DIRECT"
    assert strategy.branches[0].source_refs == (source_ref,)


def test_partial_sources_require_declared_connectivity() -> None:
    request, source_refs = _request(source_count=2, connected=True)
    relation_ref = request.source_catalog.relation_evidence[0].evidence_ref
    payload = _payload(
        request,
        alignments={source_ref: "PARTIAL" for source_ref in source_refs},
    )

    [strategy] = parse_semantic_plan_selection(payload, request=request)
    assert strategy.branches[0].relation_evidence_refs == (relation_ref,)


def test_first_direct_source_wins_without_forwarding_partial_noise() -> None:
    request, source_refs = _request(source_count=3, connected=True)
    payload = _payload(
        request,
        alignments={
            source_refs[0]: "DIRECT",
            source_refs[1]: "DIRECT",
            source_refs[2]: "PARTIAL",
        },
    )

    [strategy] = parse_semantic_plan_selection(payload, request=request)

    assert len(strategy.branches) == 1
    assert strategy.branches[0].source_refs == (source_refs[0],)


def test_disconnected_partial_sources_do_not_form_a_strategy() -> None:
    request, source_refs = _request(source_count=2)
    payload = _payload(
        request,
        alignments={source_ref: "PARTIAL" for source_ref in source_refs},
    )

    validate(payload, build_semantic_plan_selection_schema(request))
    [strategy] = parse_semantic_plan_selection(payload, request=request)

    assert strategy.branches == ()


def test_not_aligned_source_cannot_enter_strategy() -> None:
    request, source_refs = _request(source_count=2, connected=True)
    payload = _payload(
        request,
        alignments={
            source_refs[0]: "PARTIAL",
            source_refs[1]: "NOT_ALIGNED",
        },
    )

    [strategy] = parse_semantic_plan_selection(payload, request=request)

    assert strategy.branches == ()


def test_empty_strategy_records_that_current_recall_is_incomplete() -> None:
    request, source_refs = _request(source_count=1)
    payload = _payload(
        request,
        alignments={source_refs[0]: "NOT_ALIGNED"},
    )

    [strategy] = parse_semantic_plan_selection(payload, request=request)
    assert strategy.branches == ()


def _request(
    *,
    source_count: int,
    connected: bool = False,
) -> tuple[SemanticPlanSelectionRequest, tuple[str, ...]]:
    [index] = _semantic_contract().semantic_indexes
    [base] = tuple(
        source
        for source in build_row_source_catalog(
            RelationCatalog(reads=(_staff_read(),))
        ).sources
        if source.read_id
    )
    sources = tuple(
        replace(base, id=f"source_{position}")
        for position in range(1, source_count + 1)
    )
    relations = (
        tuple(
            SourceRelationEvidence(
                evidence_ref=f"relation_{position}",
                left_source_ref=sources[position - 1].id,
                right_source_ref=sources[position].id,
                left_field_refs=(sources[position - 1].fields[0].field_ref,),
                right_field_refs=(sources[position].fields[0].field_ref,),
            )
            for position in range(1, len(sources))
        )
        if connected
        else ()
    )
    request = SemanticPlanSelectionRequest(
        indexes=(index,),
        source_catalog=AvailableSourceCatalog(
            contract_snapshot=SourceContractSnapshot.from_content("{}"),
            sources=sources,
            relation_evidence=relations,
        ),
    )
    return request, tuple(source.id for source in sources)


def _payload(
    request: SemanticPlanSelectionRequest,
    *,
    alignments: dict[str, str],
) -> dict[str, object]:
    [index] = request.indexes
    return {
        "source_assessments_by_requested_fact": {
            index.requested_fact_id: {
                source_ref: {
                    "basis": f"Assessment for {source_ref}.",
                    "alignment": alignment,
                }
                for source_ref, alignment in alignments.items()
            }
        }
    }


def _clause_refs(request: SemanticPlanSelectionRequest) -> list[str]:
    [index] = request.indexes
    return [item.clause_ref for item in index.qualification.clauses]
