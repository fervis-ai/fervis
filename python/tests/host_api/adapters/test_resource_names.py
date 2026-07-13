from fervis.host_api.adapters.resource_names import endpoint_resource_names


def test_declared_resource_names_are_preserved_without_morphology() -> None:
    names = endpoint_resource_names(
        tags=("series", "analysis", "status"),
        operation_id="list_news",
        path_template="/api/news/",
    )

    assert names == ("series", "analysis", "status", "news")
