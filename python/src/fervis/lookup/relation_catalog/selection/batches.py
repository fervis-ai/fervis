"""Deterministic pagination of positive catalog-selection results."""

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection.model import (
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.relation_catalog.selection.results import (
    relation_catalog_for_read_ids,
)


def next_catalog_selection_batch(
    *,
    catalog_selection: CatalogSelectionResult,
    full_catalog: RelationCatalog,
    max_reads_per_fact: int,
) -> CatalogSelectionResult | None:
    if max_reads_per_fact < 1:
        raise ValueError("catalog recall requires a positive batch size")
    selections = tuple(
        _next_fact_batch(
            item, max_reads=max_reads_per_fact,
            reviewed_read_ids=frozenset(catalog_selection.selected_read_ids),
        )
        for item in catalog_selection.requested_fact_selections
    )
    selected_read_ids = _dedupe(
        read_id for item in selections for read_id in item.selected_read_ids
    )
    if not selected_read_ids:
        return None
    return CatalogSelectionResult(
        relation_catalog=relation_catalog_for_read_ids(
            full_catalog,
            read_ids=selected_read_ids,
        ),
        requested_fact_selections=selections,
        selected_read_ids=selected_read_ids,
    )


def combine_catalog_selection_batches(
    batches: tuple[CatalogSelectionResult, ...],
    *,
    full_catalog: RelationCatalog,
) -> CatalogSelectionResult:
    if not batches:
        raise ValueError("catalog recall requires at least one batch")
    fact_ids = tuple(
        item.requested_fact_id for item in batches[0].requested_fact_selections
    )
    reviewed_read_ids = frozenset(read_id for batch in batches for read_id in batch.selected_read_ids)
    selections = tuple(
        _combine_fact_batches(
            tuple(
                item
                for batch in batches
                for item in batch.requested_fact_selections
                if item.requested_fact_id == fact_id
            ),
            reviewed_read_ids=reviewed_read_ids,
        )
        for fact_id in fact_ids
    )
    selected_read_ids = _dedupe(
        read_id for item in selections for read_id in item.selected_read_ids
    )
    return CatalogSelectionResult(
        relation_catalog=relation_catalog_for_read_ids(
            full_catalog,
            read_ids=selected_read_ids,
        ),
        requested_fact_selections=selections,
        selected_read_ids=selected_read_ids,
    )


def _next_fact_batch(
    selection: RequestedFactCatalogSelection,
    *,
    max_reads: int,
    reviewed_read_ids: frozenset[str],
) -> RequestedFactCatalogSelection:
    # Every candidate in a read batch is assessed against every requested fact.
    pending = tuple(ref for ref in selection.unselected_positive_read_ids if ref not in reviewed_read_ids)
    selected = pending[:max_reads]
    remaining = pending[max_reads:]
    rankings_by_read = {item.read_id: item for item in selection.rankings}
    return RequestedFactCatalogSelection(
        requested_fact_id=selection.requested_fact_id,
        query_terms=selection.query_terms,
        rankings=tuple(
            rankings_by_read[read_id]
            for read_id in selected
            if read_id in rankings_by_read
        ),
        selected_read_ids=selected,
        unselected_positive_read_ids=remaining,
    )


def _combine_fact_batches(
    selections: tuple[RequestedFactCatalogSelection, ...],
    *,
    reviewed_read_ids: frozenset[str],
) -> RequestedFactCatalogSelection:
    first = selections[0]
    selected = _dedupe(
        read_id for item in selections
        for read_id in (*item.selected_read_ids, *item.unselected_positive_read_ids)
        if read_id in reviewed_read_ids
    )
    selected_set = set(selected)
    remaining = _dedupe(
        read_id
        for item in selections
        for read_id in item.unselected_positive_read_ids
        if read_id not in selected_set
    )
    rankings_by_read = {
        ranking.read_id: ranking for item in selections for ranking in item.rankings
    }
    return RequestedFactCatalogSelection(
        requested_fact_id=first.requested_fact_id,
        query_terms=first.query_terms,
        rankings=tuple(
            rankings_by_read[read_id]
            for read_id in selected
            if read_id in rankings_by_read
        ),
        selected_read_ids=selected,
        unselected_positive_read_ids=remaining,
    )


def _dedupe(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))
