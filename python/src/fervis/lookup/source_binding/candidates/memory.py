"""Memory and prior-reference source candidates."""

from fervis.lookup.fact_plan.row_sources import memory_row_source_id

from ._shared import Any, RowCardinality


def _memory_candidate_payloads(memory_inputs: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for relation in memory_inputs.get("memoryRelations", ()) or ():
        if not isinstance(relation, dict):
            continue
        relation_id = str(relation.get("id") or "")
        if not relation_id:
            continue
        fields = list(relation.get("fields") or ())
        output.append(
            {
                "source_candidate_id": relation_id,
                "kind": "prior_answer_rows",
                "memory_relation_id": relation_id,
                "fields": fields,
                "result_grains": [
                    {
                        "row_path_id": "root",
                        "row_source_id": memory_row_source_id(relation_id),
                        "cardinality": _memory_relation_cardinality(relation),
                        "evidence_items": fields,
                    }
                ],
                **_memory_relation_cardinality_payload(relation),
            }
        )
    return output


def _memory_relation_cardinality_payload(relation: dict[str, Any]) -> dict[str, str]:
    return {"cardinality": _memory_relation_cardinality(relation)}


def _memory_relation_cardinality(relation: dict[str, Any]) -> str:
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
    api_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not api_sources:
        return []
    return [
        {
            "context_id": f"requested_fact:{requested_fact_id}:current_question_api_reads",
            "kind": "current_question_api_reads",
            "ordering_rationale": (
                "catalog-selected API reads in deterministic selection order"
            ),
            "source_options": api_sources,
        }
    ]


def _has_answer_evidence_fields(candidate: dict[str, Any]) -> bool:
    fields = tuple(
        field for field in candidate.get("fields") or () if isinstance(field, dict)
    )
    if not fields:
        return bool(candidate.get("value_id"))
    param_ids = {
        str(param.get("param_id") or "")
        for param in candidate.get("params") or ()
        if isinstance(param, dict)
    } | {
        str(param.get("param_id") or "")
        for param in candidate.get("bound_params") or ()
        if isinstance(param, dict)
    }
    return any(
        field_id and field_id not in param_ids
        for field in fields
        for field_id in (str(field.get("field_id") or field.get("id") or ""),)
    )
