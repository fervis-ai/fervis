"""Read-eligibility row-path projection helpers."""

from __future__ import annotations

from fervis.lookup.relation_catalog import (
    EndpointRead,
    catalog_field_is_count_anchor,
)


def read_row_path_payloads(read: EndpointRead) -> list[dict[str, object]]:
    if read.row_paths:
        return [
            _row_path_payload(
                read,
                row_path_id=row_path.id,
                path=row_path.path,
                cardinality=row_path.cardinality.value,
                parent_path=row_path.parent_path,
            )
            for row_path in read.row_paths
        ]
    return [
        _row_path_payload(
            read,
            row_path_id="root",
            path="",
            cardinality="one",
            parent_path="",
        )
    ]


def _row_path_payload(
    read: EndpointRead,
    *,
    row_path_id: str,
    path: str,
    cardinality: str,
    parent_path: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "row_path_id": row_path_id,
        "path": path,
        "cardinality": cardinality,
        "parent_path": parent_path,
    }
    count_anchor_field_paths = [
        field.path
        for field in read.fields
        if (field.row_path_id or "root") == row_path_id
        and catalog_field_is_count_anchor(field)
    ]
    if count_anchor_field_paths:
        payload["count_anchor_field_paths"] = count_anchor_field_paths
    return payload
