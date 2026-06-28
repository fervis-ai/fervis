"""Apply read-retention results to deterministic catalog selection."""

from __future__ import annotations

from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
    relation_catalog_for_read_ids,
    selected_read_ids_from_fact_selections,
)
from fervis.lookup.read_eligibility.model import ReadEligibilityResult


def filter_catalog_selection_for_read_eligibility(
    *,
    catalog_selection: CatalogSelectionResult,
    read_eligibility: ReadEligibilityResult,
) -> CatalogSelectionResult:
    retained_by_fact = read_eligibility.retained_read_ids_by_requested_fact()
    selections = tuple(
        _filter_fact_selection(
            selection,
            retained_read_ids=retained_by_fact.get(selection.requested_fact_id, ()),
        )
        for selection in catalog_selection.requested_fact_selections
    )
    selected_read_ids = selected_read_ids_from_fact_selections(selections)
    return CatalogSelectionResult(
        relation_catalog=relation_catalog_for_read_ids(
            catalog_selection.relation_catalog,
            read_ids=selected_read_ids,
        ),
        requested_fact_selections=selections,
        selected_read_ids=selected_read_ids,
    )


def _filter_fact_selection(
    selection: RequestedFactCatalogSelection,
    *,
    retained_read_ids: tuple[str, ...],
) -> RequestedFactCatalogSelection:
    retained = set(retained_read_ids)
    selected = tuple(
        read_id for read_id in selection.selected_read_ids if read_id in retained
    )
    return RequestedFactCatalogSelection(
        requested_fact_id=selection.requested_fact_id,
        query_terms=selection.query_terms,
        rankings=tuple(item for item in selection.rankings if item.read_id in selected),
        selected_read_ids=selected,
        unselected_positive_read_ids=tuple(
            dict.fromkeys(
                (
                    *selection.unselected_positive_read_ids,
                    *(
                        read_id
                        for read_id in selection.selected_read_ids
                        if read_id not in retained
                    ),
                )
            )
        ),
    )
