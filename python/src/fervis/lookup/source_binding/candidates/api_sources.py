"""API-read source candidates for source binding."""

from typing import TypedDict

from fervis.lookup.source_binding.candidates.contracts import JsonValue

from ._shared import Any, FactValue
from .params import (
    _choice_labels,
    _param_bind_options,
    _param_binding_values,
    _param_omit_option,
)


class _DefaultParamBinding(TypedDict):
    param_id: str
    value: JsonValue
    source: str


def _api_candidate_payload(
    payload: dict[str, Any],
    *,
    available_values: tuple[FactValue, ...],
) -> dict[str, Any]:
    applied_param_ids = {
        param_id
        for item in payload.get("applied_filters") or ()
        if isinstance(item, dict)
        for param_id in _applied_filter_param_ids(item)
    }
    output = {
        **payload,
        "kind": "new_api_read",
        "source_candidate_id": _api_source_candidate_id(payload),
    }
    if output.get("applied_filters"):
        output["applied_filters"] = [
            _prompt_projection_applied_filter(item)
            for item in output["applied_filters"]
            if isinstance(item, dict)
        ]
    if applied_param_ids:
        output["params"] = [
            item
            for item in output.get("params") or ()
            if str(item.get("param_id") or "") not in applied_param_ids
        ]
    params = tuple(
        item for item in output.get("params") or () if isinstance(item, dict)
    )
    default_bindings = _source_default_param_bindings(params)
    if default_bindings:
        output["bound_params"] = [
            *list(output.get("bound_params") or ()),
            *default_bindings,
        ]
    output["params"] = _bindable_param_payloads(
        tuple(
            item
            for item in params
            if str(item.get("param_id") or "")
            not in {binding["param_id"] for binding in default_bindings}
        ),
        available_values=available_values,
    )
    return output


def _source_default_param_bindings(
    params: tuple[dict[str, Any], ...],
) -> tuple[_DefaultParamBinding, ...]:
    bindings: list[_DefaultParamBinding] = []
    for param in params:
        param_id = str(param.get("param_id") or "")
        default_value = param.get("default")
        if not param_id or default_value is None or not _should_bind_source_default(param):
            continue
        binding: _DefaultParamBinding = {
            "param_id": str(param.get("param_id") or ""),
            "value": default_value,
            "source": "source_default",
        }
        bindings.append(binding)
    return tuple(bindings)


def _should_bind_source_default(param: dict[str, Any]) -> bool:
    if param.get("required") is True:
        return True
    return str(param.get("param_semantics") or "") == "response_shape"


def _api_source_candidate_id(payload: dict[str, Any]) -> str:
    read_id = str(payload.get("read_id") or "").strip()
    row_path_id = str(payload.get("row_path_id") or "").strip()
    row_source_id = str(payload.get("row_source_id") or "").strip()
    read_row_source_count = int(payload.get("read_row_source_count") or 0)
    if not read_id:
        return ""
    if read_row_source_count <= 1:
        return read_id
    suffix = row_path_id or "root"
    if not row_source_id:
        return f"{read_id}.{suffix}"
    return f"{read_id}.{suffix}.{row_source_id}"


def _bindable_param_payloads(
    params: tuple[dict[str, Any], ...],
    *,
    available_values: tuple[FactValue, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for param in params:
        binding_values = _param_binding_values(param, available_values=available_values)
        if not binding_values and not param.get("choices"):
            continue
        item = dict(param)
        choices = item.get("choices")
        if isinstance(choices, list) and choices:
            item["choice_labels"] = _choice_labels(item)
        if binding_values:
            item["binding_values"] = binding_values
        bind_options = _param_bind_options(item)
        if bind_options:
            item["bind_options"] = bind_options
            item["omit_option"] = _param_omit_option(
                item,
                bind_options=bind_options,
            )
        output.append(item)
    return output


def _applied_filter_param_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    param_ids = payload.get("param_ids")
    if isinstance(param_ids, list):
        return tuple(str(item) for item in param_ids if str(item))
    param_id = str(payload.get("param_id") or "")
    if param_id:
        return (param_id,)
    return ()


def _prompt_projection_applied_filter(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output.pop("param_id", None)
    output.pop("param_ids", None)
    return output
