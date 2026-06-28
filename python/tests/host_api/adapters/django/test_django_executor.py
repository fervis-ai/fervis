import pytest

from fervis.host_api.adapters.get_execution import (
    PreparedGet,
    execute_all_pages,
)
from fervis.host_api.contracts import EndpointContract
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
    )


def test_all_pages_execution_not_truncated_when_last_page_hits_page_cap():
    client = _Client(
        (
            {"data": [{"id": 1}], "pagination": {"has_more": True}},
            {"data": [{"id": 2}], "pagination": {"has_more": False}},
        )
    )

    result = execute_all_pages(
        contract=_contract(),
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"limit": 1, "max_pages": 2},
        get_page=client.get,
    )

    assert result.page_count == 2
    assert result.truncated is False


def test_all_pages_execution_preserves_has_more_when_page_cap_truncates():
    client = _Client(
        (
            {"data": [{"id": 1}], "pagination": {"has_more": True}},
            {"data": [{"id": 2}], "pagination": {"has_more": True}},
        )
    )

    result = execute_all_pages(
        contract=_contract(),
        prepared=PreparedGet(url="/v1/records/", query_params={}),
        page_policy={"limit": 1, "max_pages": 2},
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


def test_all_pages_execution_rejects_non_row_collection_response():
    client = _Client(({"detail": "schema changed"},))

    with pytest.raises(EndpointExecutionError, match="row list"):
        execute_all_pages(
            contract=_contract(),
            prepared=PreparedGet(url="/v1/records/", query_params={}),
            page_policy={"limit": 1, "max_pages": 1},
            get_page=client.get,
        )
