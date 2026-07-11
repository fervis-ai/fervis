"""Row-source lookup helpers."""

from __future__ import annotations

from fervis.lookup.answer_program.relations import Relation, SourceKind

from .builder import (
    _api_row_source_for_relation,
    memory_row_source_id,
)
from .model import CALENDAR_ROW_SOURCE_ID, RowSource, RowSourceCatalog


def row_source_for_relation(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
) -> RowSource:
    source = relation.source
    if source.kind == SourceKind.GENERATED_CALENDAR:
        return row_sources.source(CALENDAR_ROW_SOURCE_ID)
    if source.kind == SourceKind.MEMORY_READ:
        return row_sources.source(memory_row_source_id(source.memory_relation_id))
    if source.kind == SourceKind.API_READ:
        if source.row_source_id:
            selected = row_sources.source(source.row_source_id)
            if selected.read_id != source.read_id:
                raise KeyError(source.row_source_id)
            return selected
        return _api_row_source_for_relation(relation, row_sources=row_sources)
    raise KeyError(source.kind.value)
