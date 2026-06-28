from tests.lookup.orchestrator._imports import *  # noqa: F403


def _catalog(*reads: EndpointRead) -> RelationCatalog:
    return RelationCatalog(reads=reads)


def _variant_read(read_id: str, *, include_name: bool) -> EndpointRead:
    fields = [
        CatalogField(
            ref="field.variant_id",
            path="data.variant_id",
            row_path_id="data",
            type="string",
            identity=IdentityMetadata(
                entity_ref="variant",
                identity_field="variant_id",
                primary_key=True,
                display_fields=("field.variant_name",),
            ),
        )
    ]
    if include_name:
        fields.append(
            CatalogField(
                ref="field.variant_name",
                path="data.variant_name",
                row_path_id="data",
                type="string",
            )
        )
    return EndpointRead(
        id=read_id,
        endpoint_name=read_id,
        resource_names=("variant",),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=tuple(fields),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _metric_read(read_id: str) -> EndpointRead:
    return EndpointRead(
        id=read_id,
        endpoint_name=read_id,
        resource_names=("metric",),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref="field.metric_total",
                path="data.metric_total",
                row_path_id="data",
                type="number",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _inventory_read(read_id: str) -> EndpointRead:
    return EndpointRead(
        id=read_id,
        endpoint_name=read_id,
        resource_names=("inventory",),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref="field.inventory_count",
                path="data.inventory_count",
                row_path_id="data",
                type="number",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _clarification_read() -> EndpointRead:
    return EndpointRead(
        id="clarification_read",
        endpoint_name="clarification_read",
        resource_names=("clarification",),
        params=(
            CatalogParam(
                ref="clarification_read.query.selector",
                name="selector",
                source=ParamSource.QUERY,
                type="string",
                required=True,
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _metric_catalog() -> RelationCatalog:
    return _catalog(
        EndpointRead(
            id="metric_read",
            endpoint_name="metric_read",
            resource_names=("metric",),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.location_id",
                    path="data.location_id",
                    row_path_id="data",
                    type="uuid",
                    identity=IdentityMetadata(
                        entity_ref="location",
                        identity_field="location_id",
                        primary_key=True,
                        stable=True,
                        display_fields=("field.location_name",),
                    ),
                ),
                CatalogField(
                    ref="field.location_name",
                    path="data.location_name",
                    row_path_id="data",
                    type="string",
                ),
                CatalogField(
                    ref="field.metric_total",
                    path="data.metric_total",
                    row_path_id="data",
                    type="number",
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        )
    )


def _sales_and_store_catalog() -> RelationCatalog:
    return _catalog(
        EndpointRead(
            id="list_sale_list",
            endpoint_name="list_sale_list",
            resource_names=("sale",),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            params=(
                CatalogParam(
                    ref="list_sale_list.query.start_date",
                    name="start_date",
                    source=ParamSource.QUERY,
                    type="date",
                ),
                CatalogParam(
                    ref="list_sale_list.query.end_date",
                    name="end_date",
                    source=ParamSource.QUERY,
                    type="date",
                ),
            ),
            fields=(
                CatalogField(
                    ref="field.data.sale_id",
                    path="data.sale_id",
                    row_path_id="data",
                    type="string",
                    identity=IdentityMetadata(
                        entity_ref="sale",
                        primary_key=True,
                        stable=True,
                    ),
                ),
                CatalogField(
                    ref="field.data.location_id",
                    path="data.location_id",
                    row_path_id="data",
                    type="string",
                ),
                CatalogField(
                    ref="field.data.amount",
                    path="data.amount",
                    row_path_id="data",
                    type="number",
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
        EndpointRead(
            id="list_store_list",
            endpoint_name="list_store_list",
            resource_names=("store",),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.data.store_id",
                    path="data.store_id",
                    row_path_id="data",
                    type="string",
                    identity=IdentityMetadata(
                        entity_ref="store",
                        primary_key=True,
                        stable=True,
                    ),
                ),
                CatalogField(
                    ref="field.data.name",
                    path="data.name",
                    row_path_id="data",
                    type="string",
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
