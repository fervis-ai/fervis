from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.read_eligibility.semantic import (
    ReadRequirementAssessment,
    SemanticReadDecision,
    SemanticReadEligibilityResult,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog import (
    CatalogField,
    CandidateKey,
    CandidateKeyComponent,
    EndpointRead,
    RowCardinality,
    RowPath,
)
from tests.lookup.grounding._fixtures import _area_read, _location_with_area_read


def test_available_sources_are_a_bounded_projection_with_declared_relations() -> None:
    row_sources = build_row_source_catalog(
        RelationCatalog(reads=(_location_with_area_read(), _area_read()))
    )
    selected = tuple(
        source
        for source in row_sources.sources
        if source.read_id in {"list_location_list", "list_area_list"}
    )
    result = SemanticReadEligibilityResult(
        read_assessments=tuple(
            ReadRequirementAssessment(
                requested_fact_id="fact_1",
                candidate_ref=source.read_id,
                source_refs=(source.id,),
                read_id=source.read_id,
                relevant_field_refs=tuple(
                    field.field_ref for field in source.fields
                ),
                assessment_basis="The source contributes declared evidence.",
                decision=SemanticReadDecision.RETAIN,
            )
            for source in selected
        ),
        identity_outcomes=(),
    )

    catalog = build_available_source_catalog(
        row_sources,
        read_eligibility=result,
    )
    repeated = build_available_source_catalog(
        row_sources,
        read_eligibility=result,
    )

    assert catalog == repeated
    assert {source.id for source in catalog.sources} == {
        source.id for source in selected
    }
    assert all(
        item.left_source_ref in {source.id for source in selected}
        and item.right_source_ref in {source.id for source in selected}
        for item in catalog.relation_evidence
    )
    assert len(catalog.relation_evidence) == 1
    [relation] = catalog.relation_evidence
    assert relation.left_field_refs != relation.right_field_refs

    projected = catalog.select(
        source_refs=frozenset((relation.left_source_ref,)),
        relation_evidence_refs=frozenset(),
    )
    assert [source.id for source in projected.sources] == [relation.left_source_ref]
    assert projected.relation_evidence == ()
    assert {
        item.source_ref for item in projected.identity_evidence
    } == {relation.left_source_ref}


def test_retained_read_fields_project_to_their_shallowest_row_sources() -> None:
    read = _nested_sales_read()
    row_sources = build_row_source_catalog(RelationCatalog(reads=(read,)))
    read_sources = tuple(
        source for source in row_sources.sources if source.read_id == read.id
    )
    result = SemanticReadEligibilityResult(
        read_assessments=(
            ReadRequirementAssessment(
                requested_fact_id="fact_1",
                candidate_ref=read.id,
                source_refs=tuple(source.id for source in read_sources),
                read_id=read.id,
                relevant_field_refs=tuple(field.ref for field in read.fields),
                assessment_basis="The read contributes sale and item fields.",
                decision=SemanticReadDecision.RETAIN,
            ),
        ),
        identity_outcomes=(),
    )

    catalog = build_available_source_catalog(row_sources, read_eligibility=result)

    assert {source.row_path for source in catalog.sources} == {"data", "data.items"}
    sale_source = next(source for source in catalog.sources if source.row_path == "data")
    assert sale_source.candidate_keys[0].entity_kind == "sale"


def _nested_sales_read() -> EndpointRead:
    return EndpointRead(
        id="list_sale_list",
        endpoint_name="list_sale_list",
        row_paths=(
            RowPath(id="sales", path="data", cardinality=RowCardinality.MANY),
            RowPath(
                id="items",
                path="data.items",
                parent_path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.sale_id",
                path="data.sale_id",
                row_path_id="sales",
                type="uuid",
            ),
            CatalogField(
                ref="field.data.amount",
                path="data.amount",
                row_path_id="sales",
                type="decimal",
            ),
            CatalogField(
                ref="field.data.items.sale_item_id",
                path="data.items.sale_item_id",
                row_path_id="items",
                type="uuid",
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
    )
