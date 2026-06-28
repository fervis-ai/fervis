"""Shared helpers for catalog-selection results."""

from __future__ import annotations

from fervis.lookup.relation_catalog import RelationCatalog
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
    return RelationCatalog(
        reads=tuple(catalog.read(read_id) for read_id in read_ids),
        facts=tuple(
            fact
            for fact in catalog.facts
            if not fact.read_id or fact.read_id in selected
        ),
    )


def dedupe_read_ids(read_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for read_id in read_ids:
        if read_id in seen:
            continue
        seen.add(read_id)
        output.append(read_id)
    return tuple(output)
