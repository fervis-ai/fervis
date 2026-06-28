"""Shared executable-support value objects for planning boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fervis.lookup.relation_catalog import (
    source_field_has_primary_stable_identity,
)


@dataclass(frozen=True)
class RowPopulationBasis:
    row_source_id: str
    row_path_id: str
    row_cardinality: str

    def __post_init__(self) -> None:
        if not self.row_source_id:
            raise ValueError("row population basis requires row source")
        if not self.row_path_id:
            raise ValueError("row population basis requires row path")
        if self.row_cardinality != "many":
            raise ValueError("row population basis requires many rows")


@dataclass(frozen=True)
class CountBasis:
    record_id_field_id: str = ""
    row_population: RowPopulationBasis | None = None

    def __post_init__(self) -> None:
        if bool(self.record_id_field_id) == bool(self.row_population):
            raise ValueError("count basis requires exactly one basis")


def count_basis_payload(count_basis: CountBasis) -> dict[str, str]:
    if count_basis.row_population is not None:
        basis = count_basis.row_population
        return {
            "kind": "row_population",
            "row_source_id": basis.row_source_id,
            "row_path_id": basis.row_path_id,
            "row_cardinality": basis.row_cardinality,
        }
    return {
        "kind": "field",
        "record_id_field_id": count_basis.record_id_field_id,
    }


def count_basis_metric_key(count_basis: Mapping[str, Any]) -> str:
    if count_basis.get("kind") == "field":
        return str(count_basis["record_id_field_id"])
    if count_basis.get("kind") == "row_population":
        row_path_id = str(count_basis.get("row_path_id") or "")
        return f"{row_path_id}_rows" if row_path_id else "rows"
    raise ValueError("unsupported count basis")


def compiled_count_basis_payload(count_basis: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(count_basis.get("kind") or "")
    if kind == "field":
        record_id_field_id = str(count_basis.get("record_id_field_id") or "")
        if not record_id_field_id:
            raise ValueError("field count basis requires record id field")
        return {
            "record_id_field_id": record_id_field_id,
            "row_population_basis": {},
        }
    if kind == "row_population":
        row_source_id = str(count_basis.get("row_source_id") or "")
        row_path_id = str(count_basis.get("row_path_id") or "")
        row_cardinality = str(count_basis.get("row_cardinality") or "")
        if not row_source_id or not row_path_id or row_cardinality != "many":
            raise ValueError("row population count basis requires many row source")
        return {
            "record_id_field_id": "",
            "row_population_basis": {
                "row_source_id": row_source_id,
                "row_path_id": row_path_id,
                "row_cardinality": row_cardinality,
            },
        }
    raise ValueError("unsupported count basis")


def count_basis_for_evidence_item(
    item: Any,
    *,
    field: Any | None,
) -> CountBasis | None:
    """Return the executable count basis represented by one evidence item."""

    field_id = str(getattr(item, "field_id", "") or "")
    if not field_id:
        return None
    if source_field_has_primary_stable_identity(field):
        return CountBasis(record_id_field_id=field_id)
    if (
        str(getattr(item, "type", "") or "") == "row_population"
        and str(getattr(item, "row_cardinality", "") or "") == "many"
    ):
        return CountBasis(
            row_population=RowPopulationBasis(
                row_source_id=str(getattr(item, "row_source_id", "") or ""),
                row_path_id=field_id,
                row_cardinality=str(getattr(item, "row_cardinality", "") or ""),
            )
        )
    return None


def count_metric_payload_for_evidence_item(
    item: Any,
    *,
    field: Any | None,
) -> dict[str, Any] | None:
    count_basis = count_basis_for_evidence_item(item, field=field)
    if count_basis is None:
        return None
    return {
        "kind": "count_records",
        "count_basis": count_basis_payload(count_basis),
    }


def unique_count_metric_payloads(
    metrics: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for metric in metrics:
        basis = metric.get("count_basis")
        if not isinstance(basis, dict):
            continue
        key = _count_basis_dedupe_key(basis)
        if key in seen:
            continue
        seen.add(key)
        output.append(metric)
    return tuple(output)


def count_basis_matches_evidence_item(
    count_basis: Mapping[str, Any],
    item: Any,
) -> bool:
    kind = str(count_basis.get("kind") or "")
    if kind == "field":
        return str(getattr(item, "field_id", "") or "") == str(
            count_basis.get("record_id_field_id") or ""
        )
    if kind == "row_population":
        return (
            str(getattr(item, "type", "") or "") == "row_population"
            and str(getattr(item, "field_id", "") or "")
            == str(count_basis.get("row_path_id") or "")
            and str(getattr(item, "row_source_id", "") or "")
            == str(count_basis.get("row_source_id") or "")
        )
    return False


def count_basis_excluded_field_id(count_basis: Mapping[str, Any]) -> str:
    if count_basis.get("kind") == "field":
        return str(count_basis.get("record_id_field_id") or "")
    return ""


def count_basis_meaning(count_basis: Mapping[str, Any]) -> str:
    if count_basis.get("kind") == "field":
        return f"count({count_basis.get('record_id_field_id')})"
    if count_basis.get("kind") == "row_population":
        return f"count_rows({count_basis.get('row_path_id')})"
    raise ValueError("unsupported count basis")


def _count_basis_dedupe_key(count_basis: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(count_basis.get("kind") or ""),
        str(
            count_basis.get("record_id_field_id")
            or count_basis.get("row_source_id")
            or ""
        ),
        str(count_basis.get("row_path_id") or ""),
    )


@dataclass(frozen=True)
class ScopedRowPredicate:
    source_candidate_id: str
    row_path_id: str
    field_id: str
    field_path: str
    type: str
    allowed_values: tuple[str, ...]
    operator: str = "in"

    @property
    def predicate_id(self) -> str:
        return ".".join(
            (
                "rp",
                self.source_candidate_id,
                "row",
                self.row_path_id or "root",
                self.field_id,
            )
        )

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "field_id": self.field_id,
            "field_path": self.field_path,
            "row_path_id": self.row_path_id or "root",
            "type": self.type,
            "operator": self.operator,
            "allowed_values": list(self.allowed_values),
            "default": "all_values",
        }

    def __post_init__(self) -> None:
        if not self.source_candidate_id:
            raise ValueError("row predicate requires source candidate")
        if not self.field_id:
            raise ValueError("row predicate requires field")
        if not self.field_path:
            raise ValueError("row predicate requires field path")
        if not self.allowed_values:
            raise ValueError("row predicate requires allowed values")
