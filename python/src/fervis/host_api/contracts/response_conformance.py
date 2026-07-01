"""Framework-neutral response contract conformance checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .endpoint import EndpointContract

ConformanceStatus = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True)
class DeclaredResponseShape:
    root_cardinality: str
    array_field_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservedResponseShape:
    root_shape: str


@dataclass(frozen=True)
class ResponseConformanceResult:
    endpoint_name: str
    path_template: str
    status: ConformanceStatus
    reason: str
    message: str


def declared_response_shape(contract: EndpointContract) -> DeclaredResponseShape:
    return DeclaredResponseShape(
        root_cardinality=str(contract.response_cardinality or "one"),
        array_field_paths=tuple(
            path
            for field in contract.response_fields
            if _is_top_level_array_field(path := str(field.path or field.name or ""))
            and str(field.type or "").lower() == "array"
        ),
    )


def observed_response_shape(body: Any) -> ObservedResponseShape:
    if isinstance(body, list):
        return ObservedResponseShape(root_shape="array")
    if isinstance(body, dict):
        return ObservedResponseShape(root_shape="object")
    if body is None:
        return ObservedResponseShape(root_shape="empty")
    return ObservedResponseShape(root_shape="scalar")


def check_response_conformance(
    contract: EndpointContract,
    body: Any,
) -> ResponseConformanceResult:
    declared = declared_response_shape(contract)
    observed = observed_response_shape(body)
    if observed.root_shape not in {"array", "object"}:
        return _failed(
            contract,
            reason="unsupported_response_shape",
            message=(
                f"GET {contract.path_template} returned JSON {observed.root_shape}, "
                "but Fervis lookup reads require a JSON object or array."
            ),
        )
    if declared.root_cardinality == "many":
        if observed.root_shape == "array":
            return _passed(contract)
        if observed.root_shape == "object":
            if not declared.array_field_paths:
                return _failed(
                    contract,
                    reason="cardinality_mismatch",
                    message=(
                        f"GET {contract.path_template} is declared as many rows, "
                        "but returned a JSON object without a declared array field."
                    ),
                )
            return _check_declared_array_fields(
                contract,
                body,
                declared.array_field_paths,
            )
        return _failed(
            contract,
            reason="cardinality_mismatch",
            message=(
                f"GET {contract.path_template} is declared as many rows, but "
                "returned a JSON object."
            ),
        )
    if observed.root_shape == "array":
        return _failed(
            contract,
            reason="cardinality_mismatch",
            message=(
                f"GET {contract.path_template} is declared as one object, but "
                "returned a JSON array."
            ),
        )
    return _check_declared_array_fields(contract, body, declared.array_field_paths)


def _check_declared_array_fields(
    contract: EndpointContract,
    body: Any,
    array_field_paths: tuple[str, ...],
) -> ResponseConformanceResult:
    if not array_field_paths:
        return _passed(contract)
    if not isinstance(body, dict):
        return _passed(contract)
    for path in array_field_paths:
        value = body.get(path)
        if isinstance(value, list):
            continue
        return _failed(
            contract,
            reason="cardinality_mismatch",
            message=(
                f"GET {contract.path_template} declares response field {path} "
                "as an array, but the returned JSON value is not an array."
            ),
        )
    return _passed(contract)


def _is_top_level_array_field(path: str) -> bool:
    return bool(path) and "." not in path


def _passed(contract: EndpointContract) -> ResponseConformanceResult:
    return ResponseConformanceResult(
        endpoint_name=contract.endpoint_name,
        path_template=contract.path_template,
        status="passed",
        reason="shape_matches",
        message=f"GET {contract.path_template} matches its declared response shape.",
    )


def _failed(
    contract: EndpointContract,
    *,
    reason: str,
    message: str,
) -> ResponseConformanceResult:
    return ResponseConformanceResult(
        endpoint_name=contract.endpoint_name,
        path_template=contract.path_template,
        status="failed",
        reason=reason,
        message=message,
    )
