from rest_framework.mixins import ListModelMixin
from rest_framework.pagination import LimitOffsetPagination
from rest_framework import generics, serializers

from fervis.host_api.adapters.django.catalog import _build_contract
from fervis.host_api.adapters.django.pagination import (
    complete_single_response,
    pagination_contract,
)
from fervis.host_api.contracts import PaginationContract, PaginationKind
from fervis.lookup.relation_catalog.from_host_api import relation_catalog_from_endpoint_contracts
from fervis.lookup.source_reads.pagination_discovery import PaginationDiscoveryRequest


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


def test_framework_default_list_without_paginator_certifies_one_complete_response():
    class StandardListView(ListModelMixin):
        pagination_class = None

    class CustomListView(StandardListView):
        def list(self, request, *args, **kwargs):
            return None

    assert complete_single_response(StandardListView, get_action="list")
    assert not complete_single_response(CustomListView, get_action="list")
    assert not complete_single_response(StandardListView, get_action="retrieve")


def test_django_catalog_carries_framework_complete_response_authority():
    class ItemSerializer(serializers.Serializer):
        identifier = serializers.IntegerField()

    class ItemList(generics.ListAPIView):
        serializer_class = ItemSerializer
        pagination_class = None

    contract = _build_contract(
        path="items/", url_name="item-list", view_class=ItemList, converters={}
    )
    assert contract.complete_single_response
    catalog = relation_catalog_from_endpoint_contracts((contract,))
    assert catalog.reads[0].complete_single_response
    assert PaginationDiscoveryRequest(catalog, (catalog.reads[0].id,)).targets == ()
