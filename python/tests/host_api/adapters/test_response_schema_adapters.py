from fervis.host_api.adapters.jsonapi_schema import enrich_contract_from_jsonapi_resource
from fervis.host_api.adapters.marshmallow_schema import (
    response_fields_from_marshmallow_schema,
)
from fervis.host_api.contracts import EndpointContract


class _Field:
    load_only = False
    metadata: dict[str, str] = {}


class _Integer(_Field):
    pass


class _QuantityField(_Integer):
    pass


class _Schema:
    many = False
    fields = {"quantity": _QuantityField()}


class _ResourceDetail:
    schema = _Schema


def test_marshmallow_subclass_keeps_its_declared_base_type() -> None:
    fields = response_fields_from_marshmallow_schema(_Schema())

    assert [(field.path, field.type) for field in fields] == [
        ("quantity", "integer")
    ]


def test_jsonapi_detail_declares_one_data_object() -> None:
    contract = enrich_contract_from_jsonapi_resource(
        EndpointContract(
            endpoint_name="get_item",
            url_name="get_item",
            method="GET",
            path_template="/items/{item_id}/",
            docstring="Item detail.",
            view_class="ItemDetail",
        ),
        view=_ResourceDetail,
    )

    assert contract.response_cardinality == "one"
    assert (contract.response_fields[0].path, contract.response_fields[0].type) == (
        "data",
        "object",
    )
