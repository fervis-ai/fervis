"""Parameter decision options for source-binding candidates."""

import hashlib

from ._shared import (
    Any,
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralType,
    LiteralValuePayload,
    TimeValuePayload,
    ValueKind,
    canonical_param_value,
    re,
)
from fervis.lookup.source_binding.candidates.contracts import EntityTarget, parse_entity_target


def _candidate_with_param_decision_options(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return candidate
    params = candidate.get("params")
    if not isinstance(params, list):
        return candidate
    output = dict(candidate)
    output["params"] = [
        _param_with_decision_options(
            source_candidate_id=source_candidate_id,
            param=param,
        )
        for param in params
        if isinstance(param, dict)
    ]
    return output


def _candidate_with_param_population_contracts(
    candidate: dict[str, Any],
    *,
    requested_facts: tuple[Any, ...] = (),
) -> dict[str, Any]:
    params = candidate.get("params")
    if not isinstance(params, list):
        return candidate
    output = dict(candidate)
    output["params"] = [
        _param_with_population_contract(
            param=param,
            evidence_items=tuple(
                item
                for item in candidate.get("evidence_items") or ()
                if isinstance(item, dict)
            ),
            requested_facts=requested_facts,
        )
        for param in params
        if isinstance(param, dict)
    ]
    return output


def _param_with_decision_options(
    *,
    source_candidate_id: str,
    param: dict[str, Any],
) -> dict[str, Any]:
    param_id = str(param.get("param_id") or "")
    if not param_id:
        return param
    output = dict(param)
    decision_options: list[dict[str, str]] = []
    for bind_option in output.get("bind_options") or ():
        if not isinstance(bind_option, dict):
            continue
        value = str(bind_option.get("value") or "")
        if not value:
            continue
        value_component = str(bind_option.get("value_component") or "")
        decision_option = {
            "meaning": str(bind_option.get("meaning") or ""),
            "decision": "bind",
            "param_id": param_id,
            "value": value,
            "param_decision_id": _param_decision_id(
                source_candidate_id=source_candidate_id,
                param_id=param_id,
                decision="bind",
                value=value,
                value_component=value_component,
            ),
        }
        if value_component:
            decision_option["value_component"] = value_component
        decision_options.append(decision_option)
    omit_option = output.get("omit_option")
    if isinstance(omit_option, dict):
        decision = str(omit_option.get("decision") or "")
        if decision:
            option: dict[str, str] = {
                "meaning": str(omit_option.get("meaning") or ""),
                "decision": decision,
                "param_id": param_id,
            }
            default_value = str(omit_option.get("default_value") or "")
            if default_value:
                option["value"] = default_value
            option["param_decision_id"] = _param_decision_id(
                source_candidate_id=source_candidate_id,
                param_id=param_id,
                decision=decision,
                value=default_value,
            )
            decision_options.append(option)
    output.pop("bind_options", None)
    output.pop("omit_option", None)
    if decision_options:
        output["decision_surface"] = "single_decision"
        output["decision_options"] = decision_options
    return output


def _param_with_population_contract(
    *,
    param: dict[str, Any],
    evidence_items: tuple[dict[str, Any], ...],
    requested_facts: tuple[Any, ...],
) -> dict[str, Any]:
    if _param_has_omit_decision(param):
        return param
    if not _param_has_population_contract(param):
        return param
    param_id = str(param.get("param_id") or "")
    if not param_id:
        return param
    output = dict(param)
    contract: dict[str, Any] = {
        "axis_kind": "endpoint_param_value",
        "omission_behavior": _param_omission_behavior(
            output,
            requested_facts=requested_facts,
        ),
    }
    evidence_item = _returned_field_evidence(
        evidence_items=evidence_items,
        field_id=param_id,
    )
    if evidence_item:
        contract["axis_kind"] = "returned_field_value"
        contract["axis_field"] = {
            "field_id": str(evidence_item.get("field_id") or param_id),
            "evidence_id": str(evidence_item.get("evidence_id") or ""),
        }
    output["population_contract"] = contract
    return output


def _param_has_omit_decision(param: dict[str, Any]) -> bool:
    return any(
        isinstance(option, dict) and option.get("decision") == "omit"
        for option in param.get("decision_options") or ()
    )


def _param_has_population_contract(param: dict[str, Any]) -> bool:
    choices = param.get("choices")
    return bool(
        param.get("required") is not True
        and (
            (isinstance(choices, list) and any(str(choice) for choice in choices))
            or (
                str(param.get("type") or "") == "boolean"
                and _param_source_is_query(param)
            )
        )
    )


def _param_omission_behavior(
    param: dict[str, Any],
    *,
    requested_facts: tuple[Any, ...],
) -> dict[str, Any]:
    param_id = str(param.get("param_id") or "")
    bind_options = _param_bind_options(param)
    default_value = canonical_param_value(param.get("default"))
    if default_value:
        default_label = _param_option_label(default_value, bind_options=bind_options)
        effect_prefix = f"Omitting {param_id} uses the default value {default_label}"
        return {
            "kind": "uses_default",
            "default_value": default_value,
            "default_label": default_label,
            "omission_consequence_by_requested_fact": (
                _omission_consequences_by_requested_fact(
                    consequence_prefix=effect_prefix,
                    requested_facts=requested_facts,
                )
            ),
        }
    labels = tuple(
        str(item.get("label") or item.get("value") or "")
        for item in bind_options
        if str(item.get("label") or item.get("value") or "")
    )
    if labels:
        effect_prefix = (
            f"Omitting {param_id} includes records across {_joined_labels(labels)}"
        )
        return {
            "kind": "all_values",
            "omission_consequence_by_requested_fact": (
                _omission_consequences_by_requested_fact(
                    consequence_prefix=effect_prefix,
                    requested_facts=requested_facts,
                )
            ),
        }
    effect_prefix = f"Omitting {param_id} does not apply this endpoint constraint"
    return {
        "kind": "unbounded",
        "omission_consequence_by_requested_fact": (
            _omission_consequences_by_requested_fact(
                consequence_prefix=effect_prefix,
                requested_facts=requested_facts,
            )
        ),
    }


def _omission_consequences_by_requested_fact(
    *,
    consequence_prefix: str,
    requested_facts: tuple[Any, ...],
) -> dict[str, str]:
    return {
        fact_id: (
            f"{consequence_prefix} for the answer to requested_fact "
            f"{fact_id}: {description}"
        )
        for fact in requested_facts
        for fact_id in (str(getattr(fact, "id", "") or ""),)
        for description in (str(getattr(fact, "description", "") or ""),)
        if fact_id and description
    }


def _returned_field_evidence(
    *,
    evidence_items: tuple[dict[str, Any], ...],
    field_id: str,
) -> dict[str, Any]:
    for item in evidence_items:
        if str(item.get("field_id") or "") == field_id and str(
            item.get("evidence_id") or ""
        ):
            return item
    return {}


def _joined_labels(labels: tuple[str, ...]) -> str:
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _param_decision_id(
    *,
    source_candidate_id: str,
    param_id: str,
    decision: str,
    value: str = "",
    value_component: str = "",
) -> str:
    parts = (
        "param_decision",
        _symbol(source_candidate_id),
        _symbol(param_id),
        _symbol(decision),
        _id_symbol(value) if value else "",
        _symbol(value_component) if value_component else "",
    )
    return ".".join(part for part in parts if part)


def _id_symbol(value: object) -> str:
    text = str(value).strip()
    symbol = _symbol(text)
    if symbol != text.lower():
        return f"{symbol}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"
    return symbol


def _symbol(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower())
    text = text.strip("_")
    return text or "value"


def _param_supports_static_boolean_options(param: dict[str, Any]) -> bool:
    return _param_source_is_query(param)


def _param_source_is_query(param: dict[str, Any]) -> bool:
    source = str(param.get("source") or "").strip().lower()
    return source == "query"


def _param_bind_options(param: dict[str, Any]) -> list[dict[str, str]]:
    choices = param.get("choices")
    if isinstance(choices, list) and choices:
        labels = _choice_labels(param)
        return [
            {
                "value": str(choice),
                "label": str(labels.get(str(choice)) or str(choice)),
                "meaning": (
                    f"Filtering {str(param.get('param_id') or '')} to "
                    f"{str(labels.get(str(choice)) or str(choice))} means only "
                    f"records where {str(param.get('param_id') or '')} is "
                    f"{str(labels.get(str(choice)) or str(choice))}."
                ),
            }
            for choice in choices
            if str(choice)
        ]
    binding_values = param.get("binding_values")
    if isinstance(binding_values, list) and binding_values:
        return [
            {
                "value": str(item.get("value") or ""),
                "label": str(item.get("label") or item.get("value") or ""),
                "meaning": _binding_value_meaning(param=param, item=item),
                **(
                    {"value_component": str(item.get("value_component") or "")}
                    if str(item.get("value_component") or "")
                    else {}
                ),
            }
            for item in binding_values
            if isinstance(item, dict) and str(item.get("value") or "")
        ]
    return []


def _binding_value_meaning(
    *,
    param: dict[str, Any],
    item: dict[str, Any],
) -> str:
    param_id = str(param.get("param_id") or "")
    label = str(item.get("label") or item.get("value") or "")
    component = str(item.get("value_component") or "")
    if component:
        return (
            f"Filtering {param_id} to the {component} of {label} means only "
            f"records where {param_id} is that {component} date/time."
        )
    return (
        f"Filtering {param_id} to {label} means only "
        f"records where {param_id} is {label}."
    )


def _param_omit_option(
    param: dict[str, Any],
    *,
    bind_options: list[dict[str, str]],
) -> dict[str, str]:
    param_id = str(param.get("param_id") or "")
    if param.get("required") is True:
        return {}
    default_value = canonical_param_value(param.get("default"))
    if default_value:
        default_label = _param_option_label(default_value, bind_options=bind_options)
        return {
            "decision": "use_default",
            "meaning": (
                f"Using the default for {param_id} means {param_id} is {default_label}."
            ),
            "default_value": default_value,
            "default_label": default_label,
        }
    labels = tuple(
        str(item.get("label") or item.get("value") or "")
        for item in bind_options
        if str(item.get("label") or item.get("value") or "")
    )
    if _param_has_exhaustive_static_boolean_options(param):
        return {
            "decision": "omit",
            "meaning": (
                f"Omitting {param_id} includes records across {_joined_labels(labels)}."
            ),
        }
    return {}


def _param_has_exhaustive_static_boolean_options(param: dict[str, Any]) -> bool:
    binding_values = tuple(
        item for item in param.get("binding_values") or () if isinstance(item, dict)
    )
    values = {str(item.get("value") or "") for item in binding_values}
    sources = {str(item.get("source") or "") for item in binding_values}
    return (
        not param.get("choices")
        and str(param.get("type") or "") == "boolean"
        and _param_supports_static_boolean_options(param)
        and values == {"true", "false"}
        and sources == {"static_choice"}
    )


def _param_option_label(
    value: str,
    *,
    bind_options: list[dict[str, str]],
) -> str:
    for item in bind_options:
        if str(item.get("value") or "") == value:
            return str(item.get("label") or value)
    return value


def _choice_labels(param: dict[str, Any]) -> dict[str, str]:
    existing = param.get("choice_labels")
    labels = dict(existing) if isinstance(existing, dict) else {}
    return {
        str(choice): str(labels.get(str(choice)) or _choice_label(str(choice)))
        for choice in param.get("choices") or ()
        if str(choice)
    }


def _choice_label(value: str) -> str:
    normalized = value.replace("_", " ").replace("-", " ").strip().lower()
    if not normalized:
        return value
    return " ".join(item.capitalize() for item in normalized.split())


def _param_binding_values(
    param: dict[str, Any],
    *,
    available_values: tuple[FactValue, ...],
) -> list[dict[str, str]]:
    param_type = str(param.get("type") or "")
    if param_type == "boolean" and _param_supports_static_boolean_options(param):
        return [
            {"value": "true", "label": "true", "source": "static_choice"},
            {"value": "false", "label": "false", "source": "static_choice"},
        ]
    entity_target = param.get("entity_target")
    if isinstance(entity_target, dict):
        target = parse_entity_target(entity_target)
        return [
            {
                "value": value.id,
                "label": value.label or value.id,
                "source": "available_value",
            }
            for value in available_values
            if _value_matches_entity_target(value, target=target)
        ]
    if param_type in {"date", "datetime"}:
        return [
            item
            for value in available_values
            if value.kind == ValueKind.TIME
            for item in _time_binding_values(value)
        ]
    if param_type in {"integer", "number", "decimal", "float", "string"}:
        return [
            {
                "value": value.id,
                "label": value.label or value.id,
                "source": "available_value",
            }
            for value in available_values
            if _value_matches_literal_param(value, param_type=param_type)
        ]
    return []


def _time_binding_values(
    value: FactValue,
) -> tuple[dict[str, str], ...]:
    payload = value.payload
    if not isinstance(payload, TimeValuePayload):
        return ()
    base = {
        "value": value.id,
        "label": value.label or value.id,
        "source": "available_value",
    }
    allowed_components = ["start", "end"]
    if payload.resolved_start and payload.resolved_start == payload.resolved_end:
        allowed_components.append("instant")
    output = [
        {
            **base,
            "value_component": component,
            "component_label": component,
        }
        for component in allowed_components
    ]
    return tuple(output)


def _value_matches_entity_target(
    value: FactValue,
    *,
    target: EntityTarget,
) -> bool:
    if value.kind == ValueKind.IDENTITY and isinstance(
        value.payload, IdentityValuePayload
    ):
        return (
            value.payload.entity_kind == target.entity_kind
            and value.payload.key_id == target.key_id
            and target.component_id
            in {component.component_id for component in value.payload.key.components}
        )
    if value.kind == ValueKind.IDENTITY_SET and isinstance(
        value.payload, IdentitySetValuePayload
    ):
        return (
            value.payload.entity_kind == target.entity_kind
            and value.payload.key_id == target.key_id
            and all(
                target.component_id
                in {component.component_id for component in key.components}
                for key in value.payload.keys
            )
        )
    return False


def _value_matches_literal_param(value: FactValue, *, param_type: str) -> bool:
    if value.kind != ValueKind.LITERAL or not isinstance(
        value.payload, LiteralValuePayload
    ):
        return False
    if param_type in {"integer", "number", "decimal", "float"}:
        return value.payload.literal_type == LiteralType.NUMBER
    if param_type == "string":
        return value.payload.literal_type == LiteralType.STRING
    return False
