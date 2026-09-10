from fervis.host_api.contracts import (
    EndpointContract,
    PaginationContract,
    PaginationKind,
    ResponseFieldContract,
)
from fervis.lookup.relation_catalog import RowCardinality
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)


def test_paginated_many_response_has_one_envelope_and_many_result_rows() -> None:
    contract = EndpointContract(
        endpoint_name="list_records",
        url_name="records",
        method="GET",
        path_template="/records/",
        docstring="Records.",
        view_class="RecordsView",
        resource_names=("record",),
        response_cardinality="many",
        response_fields=(
            ResponseFieldContract(name="data", path="data", type="array"),
            ResponseFieldContract(name="id", path="data.id", type="string"),
        ),
        pagination=PaginationContract(
            kind=PaginationKind.OFFSET,
            position_query_param="offset",
            page_size_query_param="limit",
            results_path="data",
            page_size=50,
            max_page_size=200,
            continuation_path="pagination.has_more",
        ),
    )

    [read] = relation_catalog_from_endpoint_contracts((contract,)).reads
    paths = {item.id: item for item in read.row_paths}

    assert paths["root"].cardinality is RowCardinality.ONE
    assert paths["data"].cardinality is RowCardinality.MANY


def test_array_inside_flattened_object_has_a_declared_row_parent():
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog

    fields = (
        ResponseFieldContract(name="period", path="period", type="object"),
        ResponseFieldContract(name="id", path="period.id", type="string"),
        ResponseFieldContract(name="policy", path="period.policy", type="object"),
        ResponseFieldContract(name="prizes", path="period.policy.prizes", type="array"),
        ResponseFieldContract(
            name="amount", path="period.policy.prizes.amount", type="number"
        ),
    )

    def catalog(response_fields):
        return relation_catalog_from_endpoint_contracts(
            (
                EndpointContract(
                    endpoint_name="get_period",
                    url_name="period",
                    method="GET",
                    path_template="/period/",
                    docstring="Period and its policy prizes.",
                    view_class="PeriodView",
                    resource_names=("period",),
                    response_cardinality="one",
                    response_fields=response_fields,
                ),
            )
        )

    normal = catalog(fields)
    sources = build_api_row_source_catalog(normal)
    prizes = next(
        source
        for source in sources.sources
        if source.row_path == "period.policy.prizes"
    )
    assert prizes.parent_row_path == "period"
    assert prizes.parent_row_cardinality is RowCardinality.ONE
    reordered = catalog(tuple(reversed(fields)))
    assert reordered.reads[0].row_paths == normal.reads[0].row_paths
    assert {field.path: field.row_path_id for field in reordered.reads[0].fields} == {
        field.path: field.row_path_id for field in normal.reads[0].fields
    }
