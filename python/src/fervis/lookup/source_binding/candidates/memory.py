"""Memory and prior-reference source candidates."""

from fervis.lookup.fact_plan.row_sources import memory_row_source_id

from ._shared import RowCardinality
from fervis.lookup.source_binding.candidates.contracts import CandidateField, JsonObject, JsonValue


def _memory_candidate_payloads(memory_inputs: JsonObject) -> list[JsonObject]:
    output: list[JsonObject] = []
    for relation in _objects(memory_inputs.get("memoryRelations")):
        relation_id = str(relation.get("id") or "")
        if not relation_id:
            continue
        field_payloads = [
            _memory_field(field).payload() for field in _objects(relation.get("fields"))
        ]
        fields: list[JsonValue] = list(field_payloads)
        result_grain: JsonObject = {
            "row_path_id": "root",
            "row_source_id": memory_row_source_id(relation_id),
            "cardinality": _memory_relation_cardinality(relation),
            "evidence_items": fields,
        }
        result_grains: list[JsonValue] = [result_grain]
        output.append(
            {
                "source_candidate_id": relation_id,
                "kind": "prior_answer_rows",
                "memory_relation_id": relation_id,
                "fields": fields,
                "result_grains": result_grains,
                **_memory_relation_cardinality_payload(relation),
            }
        )
    return output


def _memory_field(field: JsonObject) -> CandidateField:
    field_id = str(field.get("id") or "")
    return CandidateField(
        field_id=field_id,
        type=str(field.get("type") or ""),
        field_ref=str(field.get("sourceField") or ""),
        path=field_id,
        row_path_id="root",
    )


def _memory_relation_cardinality_payload(relation: JsonObject) -> dict[str, str]:
    return {"cardinality": _memory_relation_cardinality(relation)}


def _memory_relation_cardinality(relation: JsonObject) -> str:
    completeness = relation.get("completeness")
    if not isinstance(completeness, dict):
        return RowCardinality.MANY.value
    row_count = completeness.get("rowCount")
    if not isinstance(row_count, int):
        return RowCardinality.MANY.value
    return RowCardinality.ONE.value if row_count <= 1 else RowCardinality.MANY.value


def _source_contexts_for_fact(
    requested_fact_id: str,
    *,
    api_sources: list[JsonObject],
) -> list[JsonObject]:
    if not api_sources:
        return []
    source_options: list[JsonValue] = list(api_sources)
    return [
        {
            "context_id": f"requested_fact:{requested_fact_id}:current_question_api_reads",
            "kind": "current_question_api_reads",
            "ordering_rationale": (
                "catalog-selected API reads in deterministic selection order"
            ),
            "source_options": source_options,
        }
    ]


def _has_answer_evidence_fields(candidate: JsonObject) -> bool:
    fields = _objects(candidate.get("fields"))
    if not fields:
        return bool(candidate.get("value_id"))
    param_ids = {
        str(param.get("param_id") or "") for param in _objects(candidate.get("params"))
    } | {
        str(param.get("param_id") or "")
        for param in _objects(candidate.get("bound_params"))
    }
    return any(
        field_id and field_id not in param_ids
        for field in fields
        for field_id in (str(field.get("field_id") or field.get("id") or ""),)
    )


def _objects(value: JsonValue | None) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))
