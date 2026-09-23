"""Returned fields with the same local name retain their owning source."""

from dataclasses import replace

from fervis.lookup.available_sources import AvailableSourceCatalog, SourceContractSnapshot
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceField, RowSourceValueType, build_row_source_catalog,
)


def test_choice_surfaces_are_source_scoped_even_when_field_names_match():
    [base] = build_row_source_catalog(RelationCatalog()).sources
    sources = tuple(
        replace(base, id=source_id, fields=(RowSourceField(
            "status", "field.data.status", "status", RowSourceValueType.CHOICE, (),
            choices=choices,
        ),))
        for source_id, choices in (
            ("orders", ("COMPLETED", "CANCELED")),
            ("shifts", ("OPEN", "CLOSED")),
        )
    )
    catalog = AvailableSourceCatalog(SourceContractSnapshot.from_content("{}"), sources, ())
    surfaces = catalog.choice_surfaces
    assert len({surface.surface_ref for surface in surfaces}) == 2
    for source, surface in zip(sources, surfaces, strict=True):
        assert catalog.choice_surface(surface.surface_ref).source_ref == source.id
        assert surface.target_ref == "field.data.status"
        assert tuple(value.value for value in surface.values) == source.fields[0].choices

    bindings = catalog.field_bindings
    assert len({binding.ref for binding in bindings}) == 2
    for binding in bindings:
        assert catalog.field_binding(binding.ref) == binding
