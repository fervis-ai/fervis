"""Canonical evidence handling for terminal blocked-fact outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.fact_plan.row_sources import (
    build_row_source_catalog,
    read_evidence_ref,
    read_field_evidence_ref,
)
from fervis.lookup.question_contract import requested_fact_evidence_ref
from fervis.lookup.source_binding.candidates.contracts import FieldEvidence

if TYPE_CHECKING:
    from fervis.lookup.source_binding.candidates.model import SourceCandidate
    from fervis.lookup.source_binding.model import BoundSource


def canonical_blocked_evidence_refs(
    raw_evidence_refs: Iterable[str],
    *,
    source_evidence_refs: Mapping[str, tuple[str, ...]],
    requested_fact_ids: Iterable[str] = (),
    non_catalog_evidence_refs: Iterable[str] = (),
) -> tuple[str, ...]:
    requested_fact_refs = {
        requested_fact_evidence_ref(requested_fact_id)
        for requested_fact_id in requested_fact_ids
    }
    non_catalog_refs = set(non_catalog_evidence_refs)
    output: list[str] = []
    for ref in raw_evidence_refs:
        if ref in requested_fact_refs or ref in non_catalog_refs:
            continue
        output.extend(source_evidence_refs.get(ref, (ref,)))
    return tuple(dict.fromkeys(output))


def blocked_fact_refs_by_read_and_field(
    relation_catalog: RelationCatalog,
) -> dict[tuple[str, str], tuple[str, ...]]:
    row_sources = build_row_source_catalog(relation_catalog)
    refs: dict[tuple[str, str], list[str]] = {}
    for source in row_sources.sources:
        if source.kind != "api_read" or not source.read_id:
            continue
        for fact in source.blocked_facts:
            if not fact.field_id:
                continue
            key = (source.read_id, fact.field_id)
            refs.setdefault(key, []).extend(fact.proof_refs)
    return {key: tuple(dict.fromkeys(value)) for key, value in refs.items()}


def source_binding_evidence_refs(
    candidates: Iterable[SourceCandidate],
    *,
    relation_catalog: RelationCatalog,
) -> dict[str, tuple[str, ...]]:
    blocked_refs = blocked_fact_refs_by_read_and_field(relation_catalog)
    output: dict[str, tuple[str, ...]] = {}
    for candidate in candidates:
        source = candidate.source
        read_id = source.read_id if source is not None else ""
        if not read_id:
            continue
        for item in candidate.evidence_items:
            if not isinstance(item, FieldEvidence):
                continue
            refs = (
                read_field_evidence_ref(read_id=read_id, field_id=item.field_id),
                *blocked_refs.get((read_id, item.field_id), ()),
            )
            output[item.evidence_id] = tuple(dict.fromkeys(refs))
    return output


def bound_source_evidence_refs(
    bound_sources: Iterable[BoundSource],
    *,
    relation_catalog: RelationCatalog,
) -> dict[str, tuple[str, ...]]:
    blocked_refs = blocked_fact_refs_by_read_and_field(relation_catalog)
    output: dict[str, tuple[str, ...]] = {}
    for bound_source in bound_sources:
        source = bound_source.source
        read_id = source.read_id if source is not None else ""
        if not read_id:
            continue
        for item in bound_source.evidence_items:
            evidence_id = item.evidence_id.strip()
            if item.type == "row_population":
                output[evidence_id] = (read_evidence_ref(read_id),)
                continue
            field_id = item.field_id.strip()
            if not evidence_id or not field_id:
                continue
            refs = (
                read_field_evidence_ref(read_id=read_id, field_id=field_id),
                *blocked_refs.get((read_id, field_id), ()),
            )
            output[evidence_id] = tuple(dict.fromkeys(refs))
    return output
