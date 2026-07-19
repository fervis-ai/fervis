"""Value and generated-calendar source candidates."""

from ._shared import Any, FactValue
from .api_sources import _source_parameter_payloads


def _calendar_candidate_payload(
    payload: dict[str, Any],
    *,
    available_values: tuple[FactValue, ...],
) -> dict[str, Any]:
    calendar_id = str(payload.get("calendar_id") or "")
    output = {
        **payload,
        "source_candidate_id": calendar_id,
    }
    applied_param_ids = {
        param_id
        for item in output.get("applied_filters") or ()
        if isinstance(item, dict)
        for param_id in _applied_filter_param_ids(item)
    }
    output["params"] = _source_parameter_payloads(
        tuple(
            item
            for item in output.get("params") or ()
            if isinstance(item, dict)
            and str(item.get("param_id") or "") not in applied_param_ids
        ),
        available_values=available_values,
    )
    if not output["params"]:
        output.pop("params", None)
    return output


def _applied_filter_param_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    param_ids = payload.get("param_ids")
    if isinstance(param_ids, list):
        return tuple(str(item) for item in param_ids if str(item))
    param_id = str(payload.get("param_id") or "")
    return (param_id,) if param_id else ()


def _value_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    value_id = str(payload.get("value_id") or "")
    value_type = str(payload.get("literal_type") or "")
    return {
        **payload,
        "kind": "value",
        "source_candidate_id": value_id,
        **({"type": value_type} if value_type else {}),
    }


def _memory_value_candidate_payloads(
    memory_inputs: dict[str, Any],
    *,
    source_linked: bool = False,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in memory_inputs.get("memoryValues", ()) or ():
        if not isinstance(value, dict):
            continue
        value_id = str(value.get("id") or "")
        if not value_id:
            continue
        has_source_link = bool(
            value.get("sourceRelationId") or value.get("sourceRowId")
        )
        if has_source_link != source_linked:
            continue
        output.append(
            {
                "source_candidate_id": value_id,
                "kind": "value",
                "value_id": value_id,
                "type": value.get("type"),
                "value": value.get("value"),
                "source_relation_id": value.get("sourceRelationId"),
                "source_row_id": value.get("sourceRowId"),
                "source_row_grain": value.get("sourceRowGrain"),
                "source_field_id": value.get("sourceFieldId"),
                "prior_answer_output_ids": value.get("priorAnswerOutputIds"),
            }
        )
    return output
