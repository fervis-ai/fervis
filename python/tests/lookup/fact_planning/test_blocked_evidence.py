from fervis.lookup.answer_program.relations import SourceKind
from fervis.lookup.fact_planning.blocked_evidence import (
    bound_source_evidence_refs,
    canonical_blocked_evidence_refs,
)
from fervis.lookup.relation_catalog import (
    CatalogField,
    EndpointRead,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.source_binding.compiler_ir import DraftRelationSource
from fervis.lookup.source_binding.model import (
    AnswerPopulation,
    BoundSource,
    SourceEvidenceItem,
)


def test_bound_row_population_handle_resolves_to_catalog_evidence() -> None:
    bound_source = BoundSource(
        id="sb_1",
        requested_fact_id="fact_1",
        answer_population=AnswerPopulation(
            population_binding_id="population_1",
            intent_text="payment rows",
            match_basis_explanation="The source returns payment rows.",
        ),
        source=DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id="payments",
        ),
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="row_population.payments",
                field_id="data",
                type="row_population",
                row_source_id="payments.data",
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="payments",
                endpoint_name="list_payments",
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="payments.field.id",
                        path="data.id",
                        row_path_id="data",
                        type="string",
                    ),
                ),
            ),
        ),
    )

    evidence = bound_source_evidence_refs(
        (bound_source,),
        relation_catalog=catalog,
    )

    assert evidence == {"row_population.payments": ("payments",)}


def test_selected_strategy_is_not_misrepresented_as_catalog_evidence() -> None:
    evidence = canonical_blocked_evidence_refs(
        (
            "requested_fact:fact_1",
            "source_strategy.fact_1.aggregate_scalar.source_2",
            "row_population.payments",
        ),
        source_evidence_refs={"row_population.payments": ("payments",)},
        requested_fact_ids=("fact_1",),
        non_catalog_evidence_refs=(
            "source_strategy.fact_1.aggregate_scalar.source_2",
        ),
    )

    assert evidence == ("payments",)
