"""Catalog row-path inference helpers."""

from __future__ import annotations

from fervis.lookup.relation_catalog.model import CatalogField, RowPath


def infer_field_row_path_id(
    field: CatalogField,
    *,
    row_paths: tuple[RowPath, ...],
) -> str:
    if field.row_path_id:
        return field.row_path_id
    if not row_paths:
        return ""
    matches = tuple(
        row_path
        for row_path in row_paths
        if field.path == row_path.path or field.path.startswith(f"{row_path.path}.")
    )
    if not matches:
        return ""
    return max(matches, key=lambda item: len(item.path)).id
