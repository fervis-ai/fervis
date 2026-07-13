"""Shared executable-support value objects for planning boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Mapping, Protocol, TypeAlias

CountBasisPayload: TypeAlias = dict[str, str]
CountMetricValue: TypeAlias = str | tuple[str, ...] | CountBasisPayload
CountMetricPayload: TypeAlias = dict[str, CountMetricValue]


class CountEvidence(Protocol):
    @property
    def field_id(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def row_cardinality(self) -> str: ...

    @property
    def row_source_id(self) -> str: ...


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
    row_population: RowPopulationBasis


def count_basis_payload(count_basis: CountBasis) -> CountBasisPayload:
    basis = count_basis.row_population
    return {
        "kind": "row_population",
        "row_source_id": basis.row_source_id,
        "row_path_id": basis.row_path_id,
        "row_cardinality": basis.row_cardinality,
    }


def parse_count_basis(count_basis: Mapping[str, str]) -> CountBasis:
    kind = str(count_basis.get("kind") or "")
    if kind == "row_population":
        row_source_id = str(count_basis.get("row_source_id") or "")
        row_path_id = str(count_basis.get("row_path_id") or "")
        row_cardinality = str(count_basis.get("row_cardinality") or "")
        if not row_source_id or not row_path_id or row_cardinality != "many":
            raise ValueError("row population count basis requires many row source")
        return CountBasis(
            row_population=RowPopulationBasis(
                row_source_id=row_source_id,
                row_path_id=row_path_id,
                row_cardinality=row_cardinality,
            )
        )
    raise ValueError("unsupported count basis")


def count_basis_for_evidence_item(
    item: CountEvidence,
) -> CountBasis | None:
    """Return the executable count basis represented by one evidence item."""

    if item.type == "row_population" and item.row_cardinality == "many":
        return CountBasis(
            row_population=RowPopulationBasis(
                row_source_id=item.row_source_id,
                row_path_id=item.field_id,
                row_cardinality=item.row_cardinality,
            )
        )
    return None


def count_metric_payload_for_evidence_item(
    item: CountEvidence,
) -> CountMetricPayload | None:
    count_basis = count_basis_for_evidence_item(item)
    if count_basis is None:
        return None
    return {
        "kind": "count_records",
        "count_basis": count_basis_payload(count_basis),
    }


def unique_count_metric_payloads(
    metrics: list[CountMetricPayload],
) -> tuple[CountMetricPayload, ...]:
    output: list[CountMetricPayload] = []
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
    count_basis: Mapping[str, str],
    item: CountEvidence,
) -> bool:
    kind = str(count_basis.get("kind") or "")
    if kind == "row_population":
        return (
            item.type == "row_population"
            and item.field_id == count_basis.get("row_path_id", "")
            and item.row_source_id == count_basis.get("row_source_id", "")
        )
    return False


def count_basis_meaning(count_basis: Mapping[str, str]) -> str:
    if count_basis.get("kind") == "row_population":
        return f"count_rows({count_basis.get('row_path_id')})"
    raise ValueError("unsupported count basis")


def _count_basis_dedupe_key(
    count_basis: Mapping[str, str],
) -> tuple[str, str, str]:
    return (
        str(count_basis.get("kind") or ""),
        str(
            count_basis.get("row_source_id") or ""
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
