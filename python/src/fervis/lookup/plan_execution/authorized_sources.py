"""Authorized execution-source transport for fact-plan verification/execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.relations import RelationSource, SourceKind


@dataclass(frozen=True)
class AuthorizedExecutionSources:
    """Executable catalog projection plus the API reads a plan may use."""

    relation_catalog: RelationCatalog = field(default_factory=RelationCatalog)
    api_read_ids: tuple[str, ...] = ()

    @classmethod
    def from_catalog_selection(
        cls,
        catalog_selection: CatalogSelectionResult | None,
    ) -> "AuthorizedExecutionSources":
        read_ids = _api_read_ids_from_catalog_selection(catalog_selection)
        return cls(
            relation_catalog=(
                catalog_selection.relation_catalog
                if catalog_selection is not None
                else RelationCatalog()
            ),
            api_read_ids=read_ids,
        )

    @classmethod
    def from_pipeline_sources(
        cls,
        *,
        full_catalog: RelationCatalog,
        catalog_selection: CatalogSelectionResult | None,
        relation_sources: Iterable[RelationSource],
    ) -> "AuthorizedExecutionSources":
        api_read_ids = _unique(
            (
                *_api_read_ids_from_catalog_selection(catalog_selection),
                *_api_read_ids_from_relation_sources(relation_sources),
            )
        )
        return cls(
            relation_catalog=_project_catalog(full_catalog, api_read_ids),
            api_read_ids=api_read_ids,
        )

    @classmethod
    def from_program(
        cls,
        *,
        full_catalog: RelationCatalog,
        program: AnswerProgram,
    ) -> "AuthorizedExecutionSources":
        api_read_ids = _unique(
            relation.source.read_id
            for relation in program.relations
            if relation.source.kind is SourceKind.API_READ
        )
        return cls(
            relation_catalog=_project_catalog(full_catalog, api_read_ids),
            api_read_ids=api_read_ids,
        )

    @property
    def allowed_read_ids(self) -> frozenset[str]:
        return frozenset(self.api_read_ids)


def _api_read_ids_from_catalog_selection(
    catalog_selection: CatalogSelectionResult | None,
) -> tuple[str, ...]:
    if catalog_selection is None:
        return ()
    read_ids: list[str] = list(catalog_selection.selected_read_ids)
    for selection in catalog_selection.requested_fact_selections:
        read_ids.extend(selection.selected_read_ids)
    return _unique(read_ids)


def _api_read_ids_from_relation_sources(
    sources: Iterable[RelationSource],
) -> tuple[str, ...]:
    return _unique(
        source.read_id
        for source in sources
        if source.kind == SourceKind.API_READ and source.read_id
    )


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def _project_catalog(
    catalog: RelationCatalog,
    read_ids: tuple[str, ...],
) -> RelationCatalog:
    selected = set(read_ids)
    return RelationCatalog(
        reads=tuple(read for read in catalog.reads if read.id in selected),
        facts=tuple(
            fact
            for fact in catalog.facts
            if not fact.read_id or fact.read_id in selected
        ),
    )
