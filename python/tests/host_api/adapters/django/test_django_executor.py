import pytest

from fervis.host_api.adapters.get_execution import (
    PreparedGet,
    execute_all_pages,
    execute_prepared_get,
)
from fervis.host_api.contracts import (
    EndpointContract,
    PaginationContract,
    PaginationKind,
)
from fervis.host_api.contracts.ports import EndpointExecutionError


class _Client:
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []

    def get(self, url, params):
        self.requests.append((url, dict(params)))
        return 200, self.pages.pop(0)


def _contract() -> EndpointContract:
    return EndpointContract(
        endpoint_name="list_records",
        url_name="records",
        method="GET",
        path_template="/v1/records/",
        docstring="Records.",
        view_class="RecordList",
        pagination=PaginationContract(
            kind=PaginationKind.OFFSET,
            position_query_param="after",
            page_size_query_param="batch",
            results_path="records",
            page_size=20,
            max_page_size=100,
            continuation_path="meta.more",
        ),
    )


def test_all_pages_execution_not_truncated_when_last_page_hits_page_cap():
    client = _Client(
        (
            {"records": [{"id": 1}], "meta": {"more": True}},
            {"records": [{"id": 2}], "meta": {"more": False}},
        )
    )

    result = execute_all_pages(
        contract=_contract(),
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"page_size": 1, "max_pages": 2},
        get_page=client.get,
    )

    assert result.page_count == 2
    assert result.truncated is False
    assert client.requests == [
        ("/v1/records/", {"batch": 1, "after": 0}),
        ("/v1/records/", {"batch": 1, "after": 1}),
    ]


def test_single_page_policy_performs_one_read_and_preserves_incompleteness():
    client = _Client(
        (
            {"records": [{"id": 1}], "meta": {"more": True}},
            {"records": [{"id": 2}], "meta": {"more": False}},
        )
    )

    result = execute_prepared_get(
        contract=_contract(),
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"mode": "single_page", "page_size": 1},
        get_page=client.get,
    )

    assert result.response_body == {
        "records": [{"id": 1}],
        "meta": {"more": True},
    }
    assert result.page_count == 1
    assert result.truncated is True
    assert client.requests == [
        ("/v1/records/", {"batch": 1, "after": 0}),
    ]


def test_total_only_pagination_requires_a_terminal_page():
    contract = EndpointContract(
        endpoint_name="list_records",
        url_name="records",
        method="GET",
        path_template="/v1/records/",
        docstring="Records.",
        view_class="RecordList",
        pagination=PaginationContract(
            kind=PaginationKind.PAGE_NUMBER,
            position_query_param="page",
            page_size_query_param="size",
            results_path="records",
            total_path="total",
            page_size=1,
            max_page_size=100,
        ),
    )
    client = _Client(
        (
            {"records": [{"id": 1}], "total": 2},
            {"records": [{"id": 2}], "total": 2},
            {"records": [], "total": 2},
        )
    )

    result = execute_all_pages(
        contract=contract,
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"page_size": 1, "max_pages": 3},
        get_page=client.get,
    )

    assert result.response_body["data"] == [{"id": 1}, {"id": 2}]
    assert result.page_count == 3
    assert result.truncated is False


def test_repeated_total_only_page_never_becomes_complete_from_row_count():
    contract = EndpointContract(
        endpoint_name="list_records",
        url_name="records",
        method="GET",
        path_template="/v1/records/",
        docstring="Records.",
        view_class="RecordList",
        pagination=PaginationContract(
            kind=PaginationKind.PAGE_NUMBER,
            position_query_param="page",
            page_size_query_param="size",
            results_path="records",
            total_path="total",
            page_size=1,
            max_page_size=100,
        ),
    )
    client = _Client(
        (
            {"records": [{"id": 1}], "total": 2},
            {"records": [{"id": 1}], "total": 2},
        )
    )

    result = execute_all_pages(
        contract=contract,
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"page_size": 1, "max_pages": 2},
        get_page=client.get,
    )

    assert result.page_count == 2
    assert result.truncated is True


def test_all_pages_execution_preserves_has_more_when_page_cap_truncates():
    client = _Client(
        (
            {"records": [{"id": 1}], "meta": {"more": True}},
            {"records": [{"id": 2}], "meta": {"more": True}},
        )
    )

    result = execute_all_pages(
        contract=_contract(),
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"page_size": 1, "max_pages": 2},
        get_page=client.get,
    )

    assert {
        "page_count": result.page_count,
        "truncated": result.truncated,
        "has_more": result.response_body["pagination"]["has_more"],
    } == {
        "page_count": 2,
        "truncated": True,
        "has_more": True,
    }


def test_all_pages_execution_stops_at_the_row_budget():
    client = _Client(
        (
            {
                "records": [{"id": 1}, {"id": 2}],
                "meta": {"more": True},
            },
        )
    )

    result = execute_all_pages(
        contract=_contract(),
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"page_size": 2, "max_rows": 2},
        get_page=client.get,
    )

    assert result.truncated is True
    assert result.response_body["data"] == [{"id": 1}, {"id": 2}]
    assert len(client.requests) == 1


def test_all_pages_execution_rejects_non_row_collection_response():
    client = _Client(({"detail": "schema changed"},))

    with pytest.raises(EndpointExecutionError, match="row path"):
        execute_all_pages(
            contract=_contract(),
            prepared=PreparedGet(url="/v1/records/", query_params={}),
            page_policy={"page_size": 1, "max_pages": 1},
            get_page=client.get,
        )
