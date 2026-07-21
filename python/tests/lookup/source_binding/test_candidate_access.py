from fervis.lookup.source_binding.candidates.contracts import FieldEvidence
from fervis.lookup.source_binding.candidates.model import SourceCandidate
from fervis.lookup.source_binding.parser.candidate_access import candidate_source_fields


def test_required_fields_do_not_cross_the_selected_row_source() -> None:
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        evidence_items=(
            FieldEvidence(
                evidence_id="source_1.data.sale_id",
                field_id="sale_id",
                type="uuid",
                row_path_id="data",
                row_source_id="rows.data",
            ),
            FieldEvidence(
                evidence_id="source_1.data_items.sale_item_id",
                field_id="sale_item_id",
                type="uuid",
                row_path_id="data_items",
                row_source_id="rows.data_items",
            ),
        ),
    )

    fields = candidate_source_fields(
        candidate,
        row_source_id="rows.data",
        required_field_ids=("sale_id", "sale_item_id"),
    )

    assert tuple(field.field_id for field in fields) == ("sale_id",)
