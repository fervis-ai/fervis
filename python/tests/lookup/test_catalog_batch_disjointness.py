"""Each recalled read is assessed for every fact, once across batches."""
from fervis.lookup.relation_catalog import EndpointRead, RelationCatalog
from fervis.lookup.relation_catalog.selection.model import CatalogSelectionResult, RequestedFactCatalogSelection
from fervis.lookup.relation_catalog.selection.batches import next_catalog_selection_batch, combine_catalog_selection_batches


def test_overlapping_fact_rankings_do_not_repeat_already_reviewed_reads():
    full = RelationCatalog(reads=tuple(EndpointRead(value,value) for value in ('a','b','c','d')))
    first = CatalogSelectionResult(
        RelationCatalog(reads=full.reads[:2]),
        (
            RequestedFactCatalogSelection('f1',(),(),('a',),('b','c')),
            RequestedFactCatalogSelection('f2',(),(),('b',),('a','d')),
        ), ('a','b'),
    )
    second = next_catalog_selection_batch(catalog_selection=first,full_catalog=full,max_reads_per_fact=1)
    assert second is not None
    assert second.selected_read_ids == ('c','d')
    combined = combine_catalog_selection_batches((first,second),full_catalog=full)
    assert combined.selected_read_ids == ('a','b','c','d')
    assert all(not item.unselected_positive_read_ids for item in combined.requested_fact_selections)
    assert next_catalog_selection_batch(catalog_selection=combined,full_catalog=full,max_reads_per_fact=1) is None
