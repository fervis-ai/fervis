from __future__ import annotations

from fervis.host_api.contracts import (
    EndpointContract,
    ResponseFieldContract,
    check_response_conformance,
)


def test_response_conformance_accepts_declared_one_observed_object() -> None:
    result = check_response_conformance(
        _contract(response_cardinality="one"),
        {"id": "order_1"},
    )

    assert result.status == "passed"


def test_response_conformance_rejects_declared_one_observed_array() -> None:
    result = check_response_conformance(
        _contract(response_cardinality="one"),
        [{"id": "order_1"}],
    )

    assert result.status == "failed"
    assert result.reason == "cardinality_mismatch"
    assert "declared as one object" in result.message
    assert "returned a JSON array" in result.message


def test_response_conformance_accepts_declared_many_observed_array() -> None:
    result = check_response_conformance(
        _contract(response_cardinality="many"),
        [{"id": "order_1"}],
    )

    assert result.status == "passed"


def test_response_conformance_accepts_declared_many_envelope_array() -> None:
    result = check_response_conformance(
        _contract(
            response_cardinality="many",
            response_fields=(
                ResponseFieldContract(name="results", path="results", type="array"),
                ResponseFieldContract(name="id", path="results.id", type="string"),
            ),
        ),
        {"results": [{"id": "order_1"}]},
    )

    assert result.status == "passed"


def test_response_conformance_rejects_declared_many_observed_object() -> None:
    result = check_response_conformance(
        _contract(response_cardinality="many"),
        {"id": "order_1"},
    )

    assert result.status == "failed"
    assert result.reason == "cardinality_mismatch"
    assert "declared as many rows" in result.message
    assert "returned a JSON object" in result.message


def test_response_conformance_rejects_scalar_json_response_shape() -> None:
    result = check_response_conformance(
        _contract(response_cardinality="one"),
        42,
    )

    assert result.status == "failed"
    assert result.reason == "unsupported_response_shape"
    assert "JSON scalar" in result.message
    assert "lookup reads require a JSON object or array" in result.message


def test_response_conformance_accepts_declared_envelope_array() -> None:
    result = check_response_conformance(
        _contract(
            response_cardinality="one",
            response_fields=(
                ResponseFieldContract(name="data", path="data", type="array"),
                ResponseFieldContract(name="id", path="data.id", type="string"),
            ),
        ),
        {"data": [{"id": "order_1"}]},
    )

    assert result.status == "passed"


def test_response_conformance_rejects_declared_envelope_array_observed_object() -> None:
    result = check_response_conformance(
        _contract(
            response_cardinality="one",
            response_fields=(
                ResponseFieldContract(name="data", path="data", type="array"),
                ResponseFieldContract(name="id", path="data.id", type="string"),
            ),
        ),
        {"data": {"id": "order_1"}},
    )

    assert result.status == "failed"
    assert result.reason == "cardinality_mismatch"
    assert "data" in result.message
    assert "not an array" in result.message


def _contract(
    *,
    response_cardinality: str,
    response_fields: tuple[ResponseFieldContract, ...] = (
        ResponseFieldContract(name="id", path="id", type="string"),
    ),
) -> EndpointContract:
    return EndpointContract(
        endpoint_name="list_orders",
        url_name="list_orders",
        method="GET",
        path_template="/api/orders/",
        docstring="",
        view_class="app:list_orders",
        response_fields=response_fields,
        response_schema_source="openapi",
        response_cardinality=response_cardinality,
    )
