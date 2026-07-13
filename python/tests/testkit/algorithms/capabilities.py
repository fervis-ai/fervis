from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.host_api.contracts.capabilities import capabilities_from_schema

from tests.testkit.assertions import subset_mismatches


@dataclass(frozen=True)
class _Param:
    name: str
    type: str


@dataclass(frozen=True)
class _Field:
    name: str
    path: str


@dataclass(frozen=True)
class _KeyComponent:
    field_path: str


@dataclass(frozen=True)
class _CandidateKey:
    components: tuple[_KeyComponent, ...]


def run_capabilities_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    capabilities = capabilities_from_schema(
        path_params=tuple(
            _param(item) for item in input_payload.get("path_params", ())
        ),
        query_params=tuple(
            _param(item) for item in input_payload.get("query_params", ())
        ),
        response_fields=tuple(
            _field(item) for item in input_payload.get("response_fields", ())
        ),
        candidate_keys=tuple(
            _candidate_key(item) for item in input_payload.get("candidate_keys", ())
        ),
    )
    expected = payload["expect"]["result_contains"]
    probes = expected.get("has") or {}
    result = {"has": {key: capabilities.has(*key.split(":", 1)) for key in probes}}
    return subset_mismatches(actual=result, expected_subset=expected)


def _param(payload: dict[str, Any]) -> _Param:
    return _Param(
        name=str(payload["name"]),
        type=str(payload.get("type") or "string"),
    )


def _field(payload: dict[str, Any]) -> _Field:
    return _Field(
        name=str(payload["name"]),
        path=str(payload.get("path") or payload["name"]),
    )


def _candidate_key(payload: dict[str, Any]) -> _CandidateKey:
    return _CandidateKey(
        components=tuple(
            _KeyComponent(field_path=str(item["field_path"]))
            for item in payload.get("components", ())
        )
    )
