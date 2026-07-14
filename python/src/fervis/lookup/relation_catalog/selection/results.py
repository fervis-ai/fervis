"""Shared helpers for catalog-selection results."""

from __future__ import annotations

from fervis.lookup.relation_catalog import EndpointRead, RelationCatalog
from fervis.lookup.relation_catalog.selection.model import (
    RequestedFactCatalogSelection,
)


def selected_read_ids_from_fact_selections(
    selections: tuple[RequestedFactCatalogSelection, ...],
) -> tuple[str, ...]:
    return dedupe_read_ids(
        tuple(
            read_id
            for selection in selections
            for read_id in selection.selected_read_ids
        )
    )


def relation_catalog_for_read_ids(
    catalog: RelationCatalog,
    *,
    read_ids: tuple[str, ...],
) -> RelationCatalog:
    selected = set(read_ids)
    retained_read_ids = _reads_required_for_relational_closure(
        catalog,
        selected_read_ids=read_ids,
    )
    retained_reads = tuple(catalog.read(read_id) for read_id in retained_read_ids)
    retained_facts = tuple(
        fact for fact in catalog.facts if not fact.read_id or fact.read_id in selected
    )
    return RelationCatalog(
        reads=retained_reads,
        facts=retained_facts,
        candidate_key_authorities=catalog.candidate_key_authorities,
    )


def _reads_required_for_relational_closure(
    catalog: RelationCatalog,
    *,
    selected_read_ids: tuple[str, ...],
) -> tuple[str, ...]:
    reads_by_id = {read.id: read for read in catalog.reads}
    key_owner_ids: dict[tuple[str, str], list[str]] = {}
    for read in catalog.reads:
        for key in read.candidate_keys:
            key_owner_ids.setdefault((key.entity_kind, key.id), []).append(read.id)

    retained_ids = list(selected_read_ids)
    next_read_index = 0
    while next_read_index < len(retained_ids):
        read_id = retained_ids[next_read_index]
        next_read_index += 1
        read = reads_by_id[read_id]
        targets = _read_key_targets(read)
        for target in targets:
            for owner_id in key_owner_ids.get(target, ()):
                if owner_id not in retained_ids:
                    retained_ids.append(owner_id)
    return tuple(retained_ids)


def _read_key_targets(read: EndpointRead) -> tuple[tuple[str, str], ...]:
    reference_targets = (
        (reference.target_entity_kind, reference.target_key_id)
        for reference in read.entity_references
    )
    parameter_targets = (
        (param.entity_target.entity_kind, param.entity_target.key_id)
        for param in read.params
        if param.entity_target is not None
    )
    return tuple(dict.fromkeys((*reference_targets, *parameter_targets)))


def dedupe_read_ids(read_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for read_id in read_ids:
        if read_id in seen:
            continue
        seen.add(read_id)
        output.append(read_id)
    return tuple(output)
