from __future__ import annotations

from typing import Any

from fervis.lookup.fact_planning.grouped_ranked_choices import (
    grouped_ranked_choice_payload,
)
from fervis.lookup.source_binding.compiler_ir import DraftRelationSource
from fervis.lookup.answer_program.relations import SourceKind
from fervis.lookup.source_binding import (
    AnswerPopulation,
    BoundSource,
    CandidateKeyEvidence,
    EntityEvidenceComponent,
    EntityReferenceEvidence,
    SourceEvidenceItem,
    SourceField,
    SourceFulfillment,
    SourceMetricFitBasis,
)

from tests.testkit.assertions import subset_mismatches


def run_grouped_ranked_choices_case(payload: dict[str, Any]) -> list[str]:
    source = _bound_source(payload["input"]["source"])
    choices = grouped_ranked_choice_payload(
        (source,),
        requested_fact_id=str(payload["input"].get("requested_fact_id") or "fact_1"),
        plan_shape=str(payload["input"].get("plan_shape") or "ranked_aggregate"),
    )
    metrics = [
        metric
        for choice in choices
        for metric in choice.get("metric_candidates") or ()
        if isinstance(metric, dict)
    ]
    return subset_mismatches(
        actual={
            "group_field_ids": sorted(
                {
                    str(field_id)
                    for choice in choices
                    for group in (choice.get("group"),)
                    if isinstance(group, dict)
                    for field_id in group.get("field_ids") or ()
                }
            ),
            "metric_field_ids": sorted(
                {
                    str(metric.get("field_id") or "")
                    for metric in metrics
                    if metric.get("kind") == "aggregate_field"
                }
            ),
            "metric_kinds": sorted(
                {str(metric.get("kind") or "") for metric in metrics}
            ),
            "count_basis": [
                metric["count_basis"]
                for metric in metrics
                if metric.get("kind") == "count_records"
                and isinstance(metric.get("count_basis"), dict)
            ],
            "metric_allowed_functions": {
                str(metric.get("kind") or ""): list(
                    metric.get("allowed_functions") or ()
                )
                for metric in metrics
            },
            "metric_source_binding_bases": {
                str(metric.get("field_id") or ""): metric.get("source_binding_basis")
                for metric in metrics
                if metric.get("source_binding_basis")
            },
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _bound_source(payload: dict[str, Any]) -> BoundSource:
    return BoundSource(
        id=str(payload.get("id") or "sb_1"),
        requested_fact_id=str(payload.get("requested_fact_id") or "fact_1"),
        answer_population=AnswerPopulation(
            population_binding_id="population_1",
            intent_text="requested rows",
            match_basis_explanation="Conformance source rows match the requested fact.",
        ),
        source=DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id=str(payload.get("read_id") or "read_1"),
            row_source_id=str(payload.get("row_source_id") or ""),
        ),
        cardinality=str(payload.get("cardinality") or "many"),
        available_field_ids=tuple(
            str(item["field_id"]) for item in payload.get("fields") or ()
        ),
        available_fields=tuple(
            SourceField(
                field_id=str(item["field_id"]),
                type=str(item.get("type") or ""),
                roles=tuple(item.get("roles") or ()),
                row_cardinality=str(item.get("row_cardinality") or ""),
            )
            for item in payload.get("fields") or ()
        ),
        evidence_items=tuple(
            SourceEvidenceItem(
                evidence_id=str(item["evidence_id"]),
                field_id=str(item.get("field_id") or ""),
                type=str(item.get("type") or ""),
                row_cardinality=str(item.get("row_cardinality") or ""),
                row_source_id=str(item.get("row_source_id") or ""),
            )
            for item in payload.get("evidence") or ()
        ),
        fulfillments=tuple(
            SourceFulfillment(
                requested_fact_id=str(
                    item.get("requested_fact_id")
                    or payload.get("requested_fact_id")
                    or "fact_1"
                ),
                answer_output_id=str(item.get("answer_output_id") or "answer_1"),
                match_basis_explanation="Conformance fulfillment.",
                fulfillment_support_set_id=str(item.get("support_id") or "support_1"),
                entity_evidence=_entity_evidence(item.get("entity_evidence")),
                value_evidence_ids=tuple(item.get("value_evidence_ids") or ()),
                metric_measure_evidence_ids=tuple(
                    item.get("metric_measure_evidence_ids") or ()
                ),
                row_count_basis_evidence_ids=tuple(
                    item.get("row_count_basis_evidence_ids") or ()
                ),
                metric_fit_bases=tuple(
                    SourceMetricFitBasis(
                        evidence_id=str(basis["evidence_id"]),
                        metric_meaning=str(basis["metric_meaning"]),
                        fit_basis=str(basis["fit_basis"]),
                    )
                    for basis in item.get("metric_fit_bases") or ()
                ),
            )
            for item in payload.get("fulfillments") or ()
        ),
    )


def _entity_evidence(
    payload: Any,
) -> CandidateKeyEvidence | EntityReferenceEvidence | None:
    if not isinstance(payload, dict):
        return None
    components = tuple(
        EntityEvidenceComponent(
            component_id=str(item["component_id"]),
            field_evidence_id=str(item["field_evidence_id"]),
            field_id=str(item["field_id"]),
        )
        for item in payload.get("components") or ()
    )
    common = {
        "evidence_id": str(payload.get("evidence_id") or "entity_1"),
        "components": components,
        "row_source_id": str(payload.get("row_source_id") or "row_source_1"),
        "row_path_id": str(payload.get("row_path_id") or "data"),
    }
    if payload.get("type") == "candidate_key":
        return CandidateKeyEvidence(
            key_id=str(payload["key_id"]),
            entity_kind=str(payload["entity_kind"]),
            **common,
        )
    if payload.get("type") == "entity_reference":
        return EntityReferenceEvidence(
            reference_id=str(payload["reference_id"]),
            target_key_id=str(payload["target_key_id"]),
            target_entity_kind=str(payload["target_entity_kind"]),
            **common,
        )
    raise ValueError("unsupported entity evidence")
