"""Configured source ownership is distinct from an app's module/URL namespace."""
from types import SimpleNamespace
from django.urls import include, path
from rest_framework import serializers
from rest_framework.views import APIView
from fervis.host_api.adapters.django import catalog
from fervis.project.source_scope import DjangoSourceScope


class RowSerializer(serializers.Serializer):
    name = serializers.CharField()


class RowsView(APIView):
    serializer_class = RowSerializer
    fervis_response_cardinality = 'many'
    def get(self, request):
        raise AssertionError('Catalog construction must not execute reads.')


def test_root_router_preserves_the_configured_source_name(monkeypatch):
    monkeypatch.setattr(RowsView, '__module__', 'library_api.views')
    patterns = [path('', include([path('records/', RowsView.as_view(), name='records')]))]
    monkeypatch.setattr(catalog, 'get_resolver', lambda: SimpleNamespace(url_patterns=patterns))
    catalog.clear_endpoint_contract_cache()
    try:
        contracts = catalog.get_endpoint_contracts(sources=(DjangoSourceScope(
            name='public-library', app_modules=('library_api',), path_prefixes=('/',)),))
        assert len(contracts) == 1
        assert contracts[0].catalog_endpoint.source_namespace_path[0] == 'public-library'
    finally:
        catalog.clear_endpoint_contract_cache()


def test_empty_configured_catalog_is_not_reported_as_ready():
    from fervis.project.catalog_command import _catalog_payload
    from fervis.project.integration import DjangoAppSource
    source = DjangoAppSource(name='empty', app_modules=('library_api',), path_prefixes=('/',))
    loaded = SimpleNamespace(config=SimpleNamespace(sources=(source,)))
    result = _catalog_payload(loaded, ())
    assert result['status'] == 'blocked'
    assert result['blocked_sources'] == [{'source': 'empty', 'reason': 'no_read_endpoints'}]
