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
        candidate_keys=(
            CandidateKey(
                id="primary_key",
                entity_kind="variant",
                components=(
                    CandidateKeyComponent(
                        id="variant_id",
                        field_ref="field.variant_id",
                    ),
                ),
                primary=True,
                context_field_refs=("field.variant_name",) if include_name else (),
            ),
        ),
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
            candidate_keys=(
                CandidateKey(
                    id="location_key",
                    entity_kind="location",
                    components=(
                        CandidateKeyComponent(
                            id="location_id",
                            field_ref="field.location_id",
                        ),
                    ),
                    primary=True,
                    context_field_refs=("field.location_name",),
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
            candidate_keys=(
                CandidateKey(
                    id="primary_key",
                    entity_kind="sale",
                    components=(
                        CandidateKeyComponent(
                            id="sale_id",
                            field_ref="field.data.sale_id",
                        ),
                    ),
                    primary=True,
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
                ),
                CatalogField(
                    ref="field.data.name",
                    path="data.name",
                    row_path_id="data",
                    type="string",
                ),
            ),
            candidate_keys=(
                CandidateKey(
                    id="primary_key",
                    entity_kind="store",
                    components=(
                        CandidateKeyComponent(
                            id="store_id",
                            field_ref="field.data.store_id",
                        ),
                    ),
                    primary=True,
                    context_field_refs=("field.data.name",),
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
