from __future__ import annotations

import pytest

from tests.testkit.adapters import host_api
from tests.testkit.algorithms import business_time, outcomes, relation_catalog
from tests.testkit.assertions import rejection_mismatches


def _rejection_expect(code: str) -> dict[str, object]:
    return {
        "result_contains": {
            "status": "rejected",
            "code": code,
        }
    }


def test_rejection_oracle_requires_stable_expected_code() -> None:
    assert rejection_mismatches(
        actual_code="invalid_time_intent",
        expected={"result_contains": {"status": "rejected"}},
    ) == ["expect.result_contains.code: required for rejection"]


def test_business_time_rejection_does_not_swallow_runtime_error(monkeypatch) -> None:
    def crash(*args, **kwargs):
        raise RuntimeError("unrelated crash")

    monkeypatch.setattr(business_time, "resolve_time", crash)

    with pytest.raises(RuntimeError, match="unrelated crash"):
        business_time.run_business_time_case(
            {
                "input": {},
                "expect": _rejection_expect("invalid_time_intent"),
            }
        )


def test_host_api_rejection_does_not_swallow_runtime_error(monkeypatch) -> None:
    def crash(*args, **kwargs):
        raise RuntimeError("unrelated crash")

    monkeypatch.setattr(host_api, "relation_catalog_from_endpoint_contracts", crash)

    with pytest.raises(RuntimeError, match="unrelated crash"):
        host_api.run_host_api_projection_case(
            {
                "input": {"contracts": []},
                "expect": _rejection_expect("invalid_relation_catalog"),
            }
        )


def test_relation_catalog_rejection_does_not_swallow_runtime_error(
    monkeypatch,
) -> None:
    monkeypatch.setattr(relation_catalog, "catalog_from_payload", lambda payload: object())

    def crash(*args, **kwargs):
        raise RuntimeError("unrelated crash")

    monkeypatch.setattr(relation_catalog, "parse_relation_catalog", crash)

    with pytest.raises(RuntimeError, match="unrelated crash"):
        relation_catalog.run_relation_catalog_case(
            {
                "input": {"catalog": {}},
                "expect": _rejection_expect("invalid_relation_catalog"),
            }
        )


def test_outcomes_rejection_does_not_swallow_runtime_error(monkeypatch) -> None:
    def crash(*args, **kwargs):
        raise RuntimeError("unrelated crash")

    monkeypatch.setattr(outcomes, "_classify_answer", crash)

    with pytest.raises(RuntimeError, match="unrelated crash"):
        outcomes.run_outcomes_classify_case(
            {
                "input": {"mode": "answer"},
                "expect": _rejection_expect("invalid_result_projection"),
            }
        )
