from rest_framework.mixins import ListModelMixin
from rest_framework.pagination import LimitOffsetPagination

from fervis.host_api.adapters.django.pagination import pagination_contract
from fervis.host_api.contracts import PaginationContract, PaginationKind


def test_django_pagination_uses_the_host_paginator_contract() -> None:
    class HostPagination(LimitOffsetPagination):
        default_limit = 25
        max_limit = 75
        limit_query_param = "batch"
        offset_query_param = "after"

        def get_paginated_response_schema(
            self,
            schema: dict[str, object],
        ) -> dict[str, object]:
            return {
                "type": "object",
                "properties": {
                    "records": schema,
                    "meta": {
                        "type": "object",
                        "properties": {"more": {"type": "boolean"}},
                    },
                },
            }

    class HostView(ListModelMixin):
        pagination_class = HostPagination

    assert pagination_contract(HostView) == PaginationContract(
        kind=PaginationKind.OFFSET,
        position_query_param="after",
        page_size_query_param="batch",
        results_path="records",
        page_size=25,
        max_page_size=75,
        continuation_path="meta.more",
    )
