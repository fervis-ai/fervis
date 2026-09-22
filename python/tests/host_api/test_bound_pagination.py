"""An inferred traversal is explicit and checked against declared API structure."""

from dataclasses import replace

import pytest

from fervis.host_api.adapters.get_execution import (
    prepare_get_endpoint,
    execute_prepared_get,
)
from fervis.host_api.contracts import (
    EndpointContract,
    ParameterContract,
    ResponseFieldContract,
    PaginationContract,
    PaginationKind,
)
from fervis.host_api.contracts.response_page import ResponsePage, ResponseFormat
from fervis.host_api.contracts.ports import EndpointExecutionError


def contract():
    return EndpointContract(
        "records",
        "records",
        "GET",
        "/records",
        "Records with numbered slices.",
        "Records",
        query_params=(
            ParameterContract("segment", "integer", default=1),
            ParameterContract("width", "integer", default=2),
        ),
        response_fields=(
            ResponseFieldContract("records", "array", "records"),
            ResponseFieldContract("id", "integer", "records.id"),
            ResponseFieldContract("matched", "integer", "matched"),
        ),
        response_cardinality="collection",
    )


def policy(**changes):
    pagination = PaginationContract(
        PaginationKind.PAGE_NUMBER,
        "segment",
        "width",
        "records",
        2,
        2,
        total_path="matched",
    )
    pagination = replace(pagination, **changes)
    return {"mode": "all_pages", "pagination_contract": pagination.to_public_dict()}


@pytest.mark.parametrize("total", [4, 5])
def test_bound_pagination_collects_every_page_with_existing_get_kernel(total):
    source = contract()
    traversal = policy()
    prepared = prepare_get_endpoint(
        source, path_params={}, query_params={}, page_policy=traversal
    )
    calls = []

    def page(url, args):
        calls.append((url, dict(args)))
        start = (args["segment"] - 1) * args["width"]
        return ResponsePage(
            200,
            {
                "records": [
                    {"id": i} for i in range(start, min(start + args["width"], total))
                ],
                "matched": total,
            },
            ResponseFormat.JSON,
        )

    result = execute_prepared_get(
        contract=source, prepared=prepared, page_policy=traversal, get_page=page
    )
    assert result.response_body["data"] == [{"id": i} for i in range(total)]
    assert result.truncated is False
    assert calls == [
        ("/records", {"segment": i, "width": 2}) for i in range(1, total // 2 + 2)
    ]
    assert source.pagination is None


@pytest.mark.parametrize(
    "changes",
    [
        {"position_query_param": "foreign"},
        {"page_size_query_param": "foreign"},
        {"position_query_param": "width"},
        {"results_path": "matched"},
        {"total_path": "records.id"},
    ],
)
def test_bound_pagination_rejects_foreign_or_wrong_typed_fields_before_read(changes):
    with pytest.raises((ValueError, EndpointExecutionError)):
        prepare_get_endpoint(
            contract(), path_params={}, query_params={}, page_policy=policy(**changes)
        )


@pytest.mark.parametrize("total, more", [(2, True), (3, False)])
def test_conflicting_completion_evidence_cannot_certify_a_population(total, more):
    source = replace(
        contract(),
        response_fields=(
            *contract().response_fields,
            ResponseFieldContract("more", "boolean", "more"),
        ),
    )
    traversal = policy(continuation_path="more")
    prepared = prepare_get_endpoint(
        source, path_params={}, query_params={}, page_policy=traversal
    )
    calls = []

    def page(url, args):
        calls.append(args)
        return ResponsePage(
            200, {"records": [{"id": 1}, {"id": 2}], "matched": total, "more": more}
        )

    with pytest.raises(EndpointExecutionError, match="completion evidence"):
        execute_prepared_get(
            contract=source, prepared=prepared, page_policy=traversal, get_page=page
        )
    assert len(calls) == 1
