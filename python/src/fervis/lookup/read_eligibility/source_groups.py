"""Low-level API read row-source grouping for read eligibility."""

from __future__ import annotations

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    api_read_source_groups,
    build_row_source_catalog,
)


def read_card_source_groups_by_read(
    catalog: RelationCatalog,
) -> dict[str, tuple[tuple[RowSource, ...], ...]]:
    sources_by_read = _api_row_sources_by_read(catalog)
    groups_by_read = {
        read_id: api_read_source_groups(tuple(sources))
        for read_id, sources in sources_by_read.items()
    }
    return {read_id: groups for read_id, groups in groups_by_read.items() if groups}


def read_ids_with_card_surface(catalog: RelationCatalog) -> frozenset[str]:
    return frozenset(read_card_source_groups_by_read(catalog))


def _api_row_sources_by_read(catalog: RelationCatalog) -> dict[str, list[RowSource]]:
    output: dict[str, list[RowSource]] = {}
    for source in build_row_source_catalog(catalog).sources:
        if not source.read_id:
            continue
        output.setdefault(source.read_id, []).append(source)
    return output
