"""Prepare deterministic recall for the read-eligibility model turn."""

from __future__ import annotations

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
    dedupe_read_ids,
    relation_catalog_for_read_ids,
    selected_read_ids_from_fact_selections,
)
from fervis.lookup.read_eligibility.source_groups import (
    read_ids_with_card_surface,
)


def prepare_catalog_selection_for_read_eligibility(
    *,
    catalog_selection: CatalogSelectionResult,
    full_catalog: RelationCatalog,
    max_reads_per_fact: int,
) -> CatalogSelectionResult:
    """Fill the read-eligibility surface from ranked positives after card pruning."""

    if max_reads_per_fact < 1:
        raise ValueError("read-eligibility recall requires positive max_reads_per_fact")
    surface_read_ids = read_ids_with_card_surface(full_catalog)
    selections = tuple(
        _prepared_fact_selection(
            selection,
            surface_read_ids=surface_read_ids,
            max_reads_per_fact=max_reads_per_fact,
        )
        for selection in catalog_selection.requested_fact_selections
    )
    selected_read_ids = selected_read_ids_from_fact_selections(selections)
    return CatalogSelectionResult(
        relation_catalog=relation_catalog_for_read_ids(
            full_catalog,
            read_ids=selected_read_ids,
        ),
        requested_fact_selections=selections,
        selected_read_ids=selected_read_ids,
    )


def _prepared_fact_selection(
    selection: RequestedFactCatalogSelection,
    *,
    surface_read_ids: frozenset[str],
    max_reads_per_fact: int,
) -> RequestedFactCatalogSelection:
    selected = _surface_selected_read_ids(
        selection,
        surface_read_ids=surface_read_ids,
        max_reads_per_fact=max_reads_per_fact,
    )
    unselected_positive = _remaining_unselected_positive_read_ids(
        selection,
        selected_read_ids=selected,
    )
    return RequestedFactCatalogSelection(
        requested_fact_id=selection.requested_fact_id,
        query_terms=selection.query_terms,
        rankings=_rankings_for_selected_reads(selection, selected_read_ids=selected),
        selected_read_ids=selected,
        unselected_positive_read_ids=unselected_positive,
    )


def _surface_selected_read_ids(
    selection: RequestedFactCatalogSelection,
    *,
    surface_read_ids: frozenset[str],
    max_reads_per_fact: int,
) -> tuple[str, ...]:
    retained_selected = _read_ids_present_on_surface(
        selection.selected_read_ids,
        surface_read_ids=surface_read_ids,
    )
    backfill = _read_ids_present_on_surface(
        selection.unselected_positive_read_ids,
        surface_read_ids=surface_read_ids,
    )
    return dedupe_read_ids((*retained_selected, *backfill))[:max_reads_per_fact]


def _read_ids_present_on_surface(
    read_ids: tuple[str, ...],
    *,
    surface_read_ids: frozenset[str],
) -> tuple[str, ...]:
    return tuple(read_id for read_id in read_ids if read_id in surface_read_ids)


def _rankings_for_selected_reads(
    selection: RequestedFactCatalogSelection,
    *,
    selected_read_ids: tuple[str, ...],
) -> tuple[CatalogSelectionRanking, ...]:
    rankings_by_read_id = {ranking.read_id: ranking for ranking in selection.rankings}
    return tuple(
        rankings_by_read_id.get(read_id)
        or CatalogSelectionRanking(read_id=read_id, score=0)
        for read_id in selected_read_ids
    )


def _remaining_unselected_positive_read_ids(
    selection: RequestedFactCatalogSelection,
    *,
    selected_read_ids: tuple[str, ...],
) -> tuple[str, ...]:
    selected = set(selected_read_ids)
    return tuple(
        read_id
        for read_id in selection.unselected_positive_read_ids
        if read_id not in selected
    )
