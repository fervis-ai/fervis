"""Row-population evidence payloads for source-binding candidates."""

from ._shared import Any


def row_population_evidence_item(
    row_path_id: str,
    *,
    row_cardinality: str,
    row_source_id: str,
) -> dict[str, Any]:
    if not row_source_id:
        raise ValueError("row population evidence requires row source")
    return {
        "evidence_id": row_population_evidence_id(row_path_id),
        "field_id": row_path_id,
        "label": row_path_id,
        "row_path_id": row_path_id,
        "row_cardinality": row_cardinality,
        "row_source_id": row_source_id,
        "type": "row_population",
    }


def row_population_evidence_id(row_path_id: str) -> str:
    return f"row_population.{row_path_id}"
