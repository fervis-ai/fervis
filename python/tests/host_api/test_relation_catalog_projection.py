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
