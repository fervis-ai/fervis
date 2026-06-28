"""Evidence field payloads for source-binding candidates."""

from ._shared import Any, RowSource, RowSourceField
from .row_population import row_population_evidence_item


def _read_field_payloads(sources: tuple[RowSource, ...]) -> list[dict[str, Any]]:
    fields: list[RowSourceField] = []
    seen: set[str] = set()
    for source in sources:
        for field in source.fields:
            if field.id in seen:
                continue
            seen.add(field.id)
            fields.append(field)
    return [
        _without_empty(
            {
                "field_id": field.id,
                "field_ref": field.field_ref,
                "path": field.path,
                "response_path": field.response_path,
                "type": field.type,
                "roles": [role.value for role in field.allowed_roles],
                **(
                    {"label": field.label}
                    if field.label and field.label != field.id
                    else {}
                ),
                **_identity_payload(field),
            }
        )
        for field in fields
    ]


def _identity_payload(field: RowSourceField) -> dict[str, Any]:
    if field.identity is None:
        return {}
    identity = {
        "entity_ref": field.identity.entity_ref,
        "identity_field": field.identity.identity_field,
        "primary_key": field.identity.primary_key,
        "stable": field.identity.stable,
    }
    return {"identity": _without_empty(identity)}


def _without_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items() if value not in (None, "", [], ())
    }


def _candidate_with_evidence_items(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("source_candidate_id") or "")
    if not candidate_id:
        return candidate
    output = dict(candidate)
    if output.get("result_grains"):
        result_grains, evidence_items = _candidate_result_grains_with_evidence_items(
            output,
            candidate_id=candidate_id,
        )
        if result_grains:
            output["result_grains"] = result_grains
        if evidence_items:
            output["evidence_items"] = evidence_items
        return output
    evidence_items = _candidate_evidence_items(output, candidate_id=candidate_id)
    if evidence_items:
        output["evidence_items"] = evidence_items
    return output


def _candidate_result_grains_with_evidence_items(
    candidate: dict[str, Any],
    *,
    candidate_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_grains: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    for grain in candidate.get("result_grains") or ():
        if not isinstance(grain, dict):
            continue
        grain_id = str(grain.get("grain_id") or grain.get("row_path_id") or "root")
        grain_items: list[dict[str, Any]] = []
        for field in grain.get("evidence_items") or ():
            if not isinstance(field, dict):
                continue
            field_id = str(field.get("field_id") or field.get("id") or "")
            if not field_id:
                continue
            evidence_item = _evidence_item_for_field(
                field,
                evidence_id=f"{candidate_id}.{grain_id}.{field_id}",
            )
            if grain_id and not evidence_item.get("row_path_id"):
                evidence_item["row_path_id"] = grain_id
            grain_items.append(evidence_item)
            evidence_items.append(evidence_item)
        row_population_item = row_population_evidence_item(
            grain_id,
            row_cardinality=str(grain.get("cardinality") or ""),
            row_source_id=str(grain.get("row_source_id") or ""),
        )
        grain_items.append(row_population_item)
        evidence_items.append(row_population_item)
        result_grains.append({**grain, "evidence_items": grain_items})
    return result_grains, evidence_items


def _candidate_evidence_items(
    candidate: dict[str, Any],
    *,
    candidate_id: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, field in enumerate(candidate.get("fields") or (), start=1):
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("field_id") or field.get("id") or "")
        if not field_id:
            continue
        items.append(
            _evidence_item_for_field(
                field,
                evidence_id=f"{candidate_id}_evidence_{index}",
            )
        )
    if items:
        return items
    value_id = str(candidate.get("value_id") or "")
    if not value_id:
        return []
    value_item = {"evidence_id": f"{candidate_id}_value", "value_id": value_id}
    value_type = str(candidate.get("type") or candidate.get("literal_type") or "")
    if value_type:
        value_item["type"] = value_type
    answer_output_ids = tuple(
        str(item)
        for item in candidate.get("answer_output_ids") or ()
        if str(item).strip()
    )
    if answer_output_ids:
        value_item["answer_output_ids"] = list(answer_output_ids)
    prior_answer_output_ids = tuple(
        str(item)
        for item in candidate.get("prior_answer_output_ids") or ()
        if str(item).strip()
    )
    if prior_answer_output_ids:
        value_item["prior_answer_output_ids"] = list(prior_answer_output_ids)
    return [value_item]


def _evidence_item_for_field(
    field: dict[str, Any],
    *,
    evidence_id: str,
) -> dict[str, Any]:
    evidence_item = {
        "evidence_id": evidence_id,
        "field_id": str(field.get("field_id") or field.get("id") or ""),
    }
    field_ref = str(field.get("field_ref") or "")
    if field_ref:
        evidence_item["field_ref"] = field_ref
    path = str(field.get("path") or "")
    if path:
        evidence_item["path"] = path
    response_path = str(field.get("response_path") or "")
    if response_path:
        evidence_item["response_path"] = response_path
    row_cardinality = str(field.get("row_cardinality") or "")
    if row_cardinality:
        evidence_item["row_cardinality"] = row_cardinality
    row_path_id = str(field.get("row_path_id") or "")
    if row_path_id:
        evidence_item["row_path_id"] = row_path_id
    row_source_id = str(field.get("row_source_id") or "")
    if row_source_id:
        evidence_item["row_source_id"] = row_source_id
    label = str(field.get("label") or "")
    if label:
        evidence_item["label"] = label
    description = str(field.get("description") or "")
    if description:
        evidence_item["description"] = description
    field_type = str(field.get("type") or "")
    if field_type:
        evidence_item["type"] = field_type
    roles = [str(role) for role in field.get("roles") or () if str(role)]
    if roles:
        evidence_item["roles"] = roles
    identity = field.get("identity")
    if isinstance(identity, dict) and identity:
        evidence_item["identity"] = dict(identity)
    answer_output_ids = tuple(
        str(item) for item in field.get("answer_output_ids") or () if str(item).strip()
    )
    if answer_output_ids:
        evidence_item["answer_output_ids"] = list(answer_output_ids)
    prior_answer_output_ids = tuple(
        str(item)
        for item in field.get("prior_answer_output_ids") or ()
        if str(item).strip()
    )
    if prior_answer_output_ids:
        evidence_item["prior_answer_output_ids"] = list(prior_answer_output_ids)
    return evidence_item
