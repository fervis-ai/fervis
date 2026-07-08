from fervis.lookup.fact_plan.row_sources import build_row_source_catalog
from fervis.lookup.fact_plan.row_sources.model import RowSourceValueType
from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogParam,
    EndpointRead,
    IdentityMetadata,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)


def test_row_source_catalog_allows_any_api_param_type_for_identity_route():
    staff_read = EndpointRead(
        id="get_staff_detail",
        endpoint_name="get_staff_detail",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="get_staff_detail.path.staff_id",
                name="staff_id",
                source=ParamSource.PATH,
                type="any",
                required=True,
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.ONE),),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.full_name",),
                ),
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
        ),
    )

    catalog = build_row_source_catalog(RelationCatalog(reads=(staff_read,)))

    source = _source_for_read(catalog, "get_staff_detail")
    assert source.params[0].type == RowSourceValueType.ANY


def test_row_source_catalog_allows_any_response_field_type():
    staff_read = EndpointRead(
        id="list_staff",
        endpoint_name="list_staff",
        resource_names=("staff",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.full_name",),
                ),
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="any",
            ),
        ),
    )

    catalog = build_row_source_catalog(RelationCatalog(reads=(staff_read,)))

    source = _source_for_read(catalog, "list_staff")
    full_name = source.field("full_name")
    assert full_name.type == RowSourceValueType.ANY
    assert full_name.can_carry_lookup_text


def _source_for_read(catalog, read_id: str):
    for source in catalog.sources:
        if source.read_id == read_id:
            return source
    raise AssertionError(f"missing row source for {read_id}")
