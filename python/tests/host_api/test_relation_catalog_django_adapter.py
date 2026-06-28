from fervis.lookup.relation_catalog import RelationCatalogProvider
from fervis.host_api.adapters.django.adapter import DjangoHostApiAdapter
from fervis.host_api.contracts import (
    CatalogEndpointContract,
    EndpointContract,
    ResponseFieldContract,
    make_catalog_endpoint_key,
)
from fervis.project.source_scope import DjangoSourceScope


def test_django_adapter_projects_endpoint_contracts_to_relation_catalog(monkeypatch):
    expected_sources = (
        DjangoSourceScope(
            name="records",
            app_modules=("tests",),
            path_prefixes=("/records/",),
        ),
    )

    def fake_contracts(*, sources=()):
        assert sources == expected_sources
        endpoint = _record_endpoint_contract()
        return (endpoint,)

    import fervis.host_api.adapters.django.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "get_endpoint_contracts", fake_contracts)
    adapter: RelationCatalogProvider = DjangoHostApiAdapter(sources=expected_sources)

    catalog = adapter.build_relation_catalog()

    read = catalog.read("list_records")
    assert read.endpoint_name == "list_records"
    assert read.catalog_endpoint is not None
    assert {
        "catalog_endpoint_key": read.catalog_endpoint.catalog_endpoint_key,
        "endpoint_name": read.catalog_endpoint.endpoint_name,
        "framework_kind": read.catalog_endpoint.framework_kind,
        "source_namespace_kind": read.catalog_endpoint.source_namespace_kind,
        "source_namespace_path": read.catalog_endpoint.source_namespace_path,
        "route_method": read.catalog_endpoint.route_method,
        "route_path_template": read.catalog_endpoint.route_path_template,
        "route_name": read.catalog_endpoint.route_name,
        "api_schema_operation_id": read.catalog_endpoint.api_schema_operation_id,
        "handler_ref": read.catalog_endpoint.handler_ref,
        "domain_resource_names": read.catalog_endpoint.domain_resource_names,
    } == {
        "catalog_endpoint_key": make_catalog_endpoint_key(_record_endpoint_contract()),
        "endpoint_name": "list_records",
        "framework_kind": "django_drf",
        "source_namespace_kind": "django_app",
        "source_namespace_path": ("tests",),
        "route_method": "GET",
        "route_path_template": "/records/",
        "route_name": "records",
        "api_schema_operation_id": "",
        "handler_ref": "tests.RecordView",
        "domain_resource_names": ("record",),
    }


def _record_endpoint_contract() -> EndpointContract:
    return EndpointContract(
        endpoint_name="list_records",
        url_name="records",
        method="GET",
        path_template="/records/",
        docstring="",
        view_class="tests.RecordView",
        resource_names=("record",),
        response_fields=(
            ResponseFieldContract(name="id", type="string", path="id"),
        ),
        catalog_endpoint=CatalogEndpointContract(
            framework_kind="django_drf",
            source_namespace_kind="django_app",
            source_namespace_path=("tests",),
            handler_ref="tests.RecordView",
            route_name="records",
            domain_resource_names=("record",),
        ),
    )
