"""Canonical evidence handling for terminal blocked-fact outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.fact_plan.row_sources import (
    build_row_source_catalog,
    read_field_evidence_ref,
)
from fervis.lookup.question_contract import requested_fact_evidence_ref


def canonical_blocked_evidence_refs(
    raw_evidence_refs: Iterable[str],
    *,
    source_evidence_refs: Mapping[str, tuple[str, ...]],
    requested_fact_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    requested_fact_refs = {
        requested_fact_evidence_ref(requested_fact_id)
        for requested_fact_id in requested_fact_ids
    }
    output: list[str] = []
    for ref in raw_evidence_refs:
        if ref in requested_fact_refs:
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


def blocked_field_evidence_refs(
    fields: Iterable[Any],
    *,
    relation_catalog: RelationCatalog,
) -> tuple[str, ...]:
    blocked_refs = blocked_fact_refs_by_read_and_field(relation_catalog)
    output: list[str] = []
    for field in fields:
        read_id = str(getattr(field, "read_id", "") or "").strip()
        field_id = str(getattr(field, "field_id", "") or "").strip()
        if not read_id or not field_id:
            continue
        output.extend(
            (
                read_field_evidence_ref(read_id=read_id, field_id=field_id),
                *blocked_refs.get((read_id, field_id), ()),
            )
        )
    return tuple(dict.fromkeys(output))


def source_binding_evidence_refs(
    candidates: Iterable[Any],
    *,
    relation_catalog: RelationCatalog,
) -> dict[str, tuple[str, ...]]:
    blocked_refs = blocked_fact_refs_by_read_and_field(relation_catalog)
    output: dict[str, tuple[str, ...]] = {}
    for candidate in candidates:
        source = getattr(candidate, "source", None)
        read_id = source.read_id if source is not None else ""
        if not read_id:
            continue
        payload = getattr(candidate, "payload", None)
        evidence_items = (
            payload.get("evidence_items") if isinstance(payload, dict) else ()
        )
        output.update(
            _evidence_refs_for_items(
                evidence_items,
                read_id=read_id,
                blocked_refs=blocked_refs,
            )
        )
    return output


def bound_source_evidence_refs(
    bound_sources: Iterable[Any],
    *,
    relation_catalog: RelationCatalog,
) -> dict[str, tuple[str, ...]]:
    blocked_refs = blocked_fact_refs_by_read_and_field(relation_catalog)
    output: dict[str, tuple[str, ...]] = {}
    for bound_source in bound_sources:
        source = getattr(bound_source, "source", None)
        read_id = source.read_id if source is not None else ""
        if not read_id:
            continue
        output.update(
            _evidence_refs_for_items(
                (
                    {
                        "evidence_id": item.evidence_id,
                        "field_id": item.field_id,
                    }
                    for item in getattr(bound_source, "evidence_items", ())
                ),
                read_id=read_id,
                blocked_refs=blocked_refs,
            )
        )
    return output


def _evidence_refs_for_items(
    evidence_items: Iterable[Any],
    *,
    read_id: str,
    blocked_refs: Mapping[tuple[str, str], tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        field_id = str(item.get("field_id") or "").strip()
        if not evidence_id or not field_id:
            continue
        refs = [
            read_field_evidence_ref(read_id=read_id, field_id=field_id),
            *blocked_refs.get((read_id, field_id), ()),
        ]
        output[evidence_id] = tuple(dict.fromkeys(refs))
    return output
