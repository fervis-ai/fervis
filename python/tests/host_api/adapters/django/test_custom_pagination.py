"""Custom APIView pagination requires an explicit, validated response contract."""
import pytest
from rest_framework.views import APIView
from fervis.host_api.adapters.django.pagination import pagination_contract
from fervis.host_api.contracts import PaginationKind


def test_custom_view_can_declare_its_actual_pagination_mechanics():
    class Feed(APIView):
        fervis_pagination = {
            'kind': 'offset', 'positionQueryParam': 'offset', 'pageSizeQueryParam': 'limit',
            'resultsPath': 'data', 'pageSize': 50, 'maxPageSize': 200,
            'totalPath': 'pagination.count', 'continuationPath': 'pagination.has_more',
        }
    contract = pagination_contract(Feed)
    assert contract is not None
    assert contract.kind is PaginationKind.OFFSET
    assert contract.results_path == 'data'
    assert contract.total_path == 'pagination.count'
    assert contract.page_size == 50


def test_invalid_custom_pagination_cannot_silently_become_an_unpaginated_read():
    class Feed(APIView):
        fervis_pagination = {'kind': 'offset'}
    with pytest.raises(ValueError, match='pagination'):
        pagination_contract(Feed)


def test_custom_pagination_projects_rows_under_the_declared_envelope():
    from rest_framework import serializers
    from fervis.host_api.adapters.django.catalog import _build_contract
    from fervis.lookup.relation_catalog.from_host_api import relation_catalog_from_endpoint_contracts
    from fervis.lookup.relation_catalog import RowCardinality

    class RowSerializer(serializers.Serializer):
        id = serializers.UUIDField()

    class Feed(APIView):
        serializer_class = RowSerializer
        fervis_response_cardinality = 'many'
        fervis_pagination = {
            'kind':'offset', 'positionQueryParam':'offset', 'pageSizeQueryParam':'limit',
            'resultsPath':'data', 'pageSize':50, 'maxPageSize':200, 'totalPath':'pagination.count',
        }
    contract = _build_contract(path='feed/', url_name='feed-list', view_class=Feed, converters={})
    read, = relation_catalog_from_endpoint_contracts((contract,)).reads
    assert {path.id:path.cardinality for path in read.row_paths} == {
        'root': RowCardinality.ONE, 'data': RowCardinality.MANY,
    }
