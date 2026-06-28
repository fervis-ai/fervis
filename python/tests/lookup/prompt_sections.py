from __future__ import annotations

import json
from typing import Any
from xml.etree import ElementTree


def prompt_section_payload(prompt: str, label: str) -> dict[str, Any]:
    raw = prompt_section_text(prompt, label)
    if raw.lstrip().startswith("<"):
        return _xml_section_payload(raw)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} must be an object")
    return payload


def prompt_section_text(prompt: str, label: str) -> str:
    marker = f"{label}:\n"
    if marker not in prompt:
        raise AssertionError(f"prompt missing section: {label}")
    start = prompt.index(marker) + len(marker)
    end = prompt.find("\n\n", start)
    if end < 0:
        end = len(prompt)
    return prompt[start:end]


def _xml_section_payload(raw: str) -> dict[str, Any]:
    root = ElementTree.fromstring(raw)
    if root.tag == "candidate_api_reads":
        return {
            "requested_fact_read_candidates": [
                {
                    "requested_fact_id": str(group.get("id") or ""),
                    "read_candidates": [
                        _api_read_payload(child)
                        for child in group
                        if child.tag == "api_read"
                    ],
                }
                for group in root
                if group.tag == "requested_fact"
            ]
        }
    if root.tag == "plan_selection_source_strategies":
        return {
            "requested_fact_source_strategies": [
                {
                    "requested_fact_id": str(group.get("id") or ""),
                    "source_strategies": [
                        _source_strategy_payload(child)
                        for child in group
                        if child.tag == "source_strategy"
                    ],
                }
                for group in root
                if group.tag == "requested_fact"
            ]
        }
    if root.tag == "candidate_evidence_sources":
        payload: dict[str, Any] = {
            "requested_fact_sources": [
                {
                    "requested_fact_id": str(group.get("id") or ""),
                    "source_contexts": [
                        {
                            "context_id": str(context.get("id") or ""),
                            "kind": str(context.get("kind") or ""),
                            "source_options": [
                                _source_payload(child) for child in context
                            ],
                        }
                        for context in group
                        if context.tag == "source_context"
                    ],
                }
                for group in root
                if group.tag == "requested_fact"
            ]
        }
        for tag, key in (
            ("memory_sources", "memory_source_candidates"),
            ("utility_sources", "utility_source_candidates"),
            ("value_sources", "value_source_candidates"),
        ):
            group = root.find(tag)
            if group is not None:
                payload[key] = [_source_payload(child) for child in group]
        return payload
    raise AssertionError(f"unsupported XML prompt section: {root.tag}")


def _source_strategy_payload(element: ElementTree.Element) -> dict[str, Any]:
    output = {
        "source_strategy_id": str(element.get("id") or ""),
        "plan_shape": str(element.get("plan_shape") or ""),
    }
    answer_outputs = _split_words(element.get("answer_outputs"))
    if answer_outputs:
        output["required_answer_output_ids"] = answer_outputs
    output["source_members"] = [_source_payload(child) for child in element]
    return output


def _source_payload(element: ElementTree.Element) -> dict[str, Any]:
    if element.tag == "api_read":
        return _api_read_payload(element)
    output = {
        "source_candidate_id": str(element.get("id") or ""),
        "kind": str(element.get("kind") or ""),
        "value_id": str(element.get("value") or ""),
        "source_relation_id": str(element.get("relation") or ""),
        "memory_relation_id": str(element.get("memory_relation") or ""),
        "source_field_id": str(element.get("field") or ""),
        "calendar_id": str(element.get("calendar") or ""),
        "cardinality": str(element.get("cardinality") or ""),
    }
    description = _child_text(element, "description")
    if description:
        output["description"] = description
    fields = _flat_fields_payload(element.find("fields"))
    if fields:
        output["fields"] = fields
    evidence = _flat_fields_payload(
        element.find("evidence_items"),
        item_tag="evidence",
        attr_map={"name": "field_id", "id": "evidence_id", "path": "field_path"},
    )
    if evidence:
        output["evidence_items"] = evidence
    population_bindings = _generic_children_payloads(element.find("population_bindings"))
    if population_bindings:
        output["population_bindings"] = population_bindings
    fulfillment_choices = _fulfillment_choices_payload(
        element.find("fulfillment_choices")
    )
    if fulfillment_choices:
        output["fulfillment_choices"] = fulfillment_choices
        output["fulfillment_support_sets"] = fulfillment_choices
    return {key: value for key, value in output.items() if value not in ("", [], {})}


def _api_read_payload(element: ElementTree.Element) -> dict[str, Any]:
    output: dict[str, Any] = {
        "source_candidate_id": str(element.get("id") or ""),
        "read_id": str(element.get("read") or ""),
    }
    for attr, key in (
        ("kind", "kind"),
        ("endpoint", "endpoint_name"),
        ("row_source", "row_source_id"),
        ("memory_relation", "memory_relation_id"),
        ("resources", "resource_names"),
    ):
        value = str(element.get(attr) or "")
        if value:
            output[key] = _split_words(value) if key == "resource_names" else value
    description = _child_text(element, "description")
    if description:
        output["description"] = description
        output["docstring"] = description
    input_params = _input_params_payload(element.find("input_params"))
    if input_params:
        output["input_params"] = input_params
    binding_params = _binding_params_payload(element.find("binding_params"))
    if binding_params:
        output["params"] = binding_params
    response = element.find("response")
    if response is not None:
        output["response_rows"] = [
            row
            for child in response
            if child.tag == "row"
            for row in _response_row_payloads(child)
        ]
    row_predicates = _row_predicates_payload(element.find("row_predicates"))
    if row_predicates:
        output["row_predicates"] = row_predicates
    applied_filters = _generic_children_payloads(element.find("applied_filters"))
    if applied_filters:
        output["applied_filters"] = applied_filters
    population_roles = _population_roles_payload(element.find("population_roles"))
    if population_roles:
        output["population_roles"] = population_roles
    population_bindings = _generic_children_payloads(element.find("population_bindings"))
    if population_bindings:
        output["population_bindings"] = population_bindings
    fulfillment_choices = _fulfillment_choices_payload(
        element.find("fulfillment_choices")
    )
    if fulfillment_choices:
        output["fulfillment_choices"] = fulfillment_choices
    return output


def _row_predicates_payload(
    element: ElementTree.Element | None,
) -> list[dict[str, Any]]:
    if element is None:
        return []
    output = []
    for child in element:
        if child.tag != "predicate":
            continue
        item = _attrs(child, attr_map={"id": "predicate_id", "field": "field_id"})
        values = [
            str(value.text or "")
            for values_node in child
            if values_node.tag == "values"
            for value in values_node
            if value.tag == "value" and str(value.text or "")
        ]
        if values:
            item["allowed_values"] = values
        output.append(item)
    return output


def _input_params_payload(element: ElementTree.Element | None) -> list[dict[str, Any]]:
    if element is None:
        return []
    output = []
    for child in element:
        if child.tag != "param":
            continue
        item = _attrs(child)
        if "required" in item:
            item["required"] = _bool_value(item["required"])
        choices = [
            str(choice.get("value") or "")
            for choices_node in child
            if choices_node.tag == "choices"
            for choice in choices_node
            if choice.tag == "choice" and str(choice.get("value") or "")
        ]
        if choices:
            item["choices"] = choices
        output.append(item)
    return output


def _binding_params_payload(element: ElementTree.Element | None) -> list[dict[str, Any]]:
    if element is None:
        return []
    output = []
    for child in element:
        if child.tag != "param":
            continue
        item = _attrs(child, attr_map={"id": "param_id"})
        if "required" in item:
            item["required"] = _bool_value(item["required"])
        choices_node = child.find("choices")
        choices = [
            str(choice.get("value") or "")
            for choice in choices_node or ()
            if choice.tag == "choice" and str(choice.get("value") or "")
        ]
        if choices:
            item["choices"] = choices
        binding_values_node = child.find("binding_values")
        binding_values = [
            _attrs(value)
            for value in binding_values_node or ()
            if value.tag == "value"
        ]
        if binding_values:
            item["binding_values"] = binding_values
        decision_options_node = child.find("decision_options")
        decision_options = [
            _attrs(option, attr_map={"id": "param_decision_id"})
            for option in decision_options_node or ()
            if option.tag == "option"
        ]
        if decision_options:
            item["decision_options"] = decision_options
        population_contract = _population_contract_payload(
            child.find("population_contract")
        )
        if population_contract:
            item["population_contract"] = population_contract
        profiles_node = child.find("normal_instance_role_profiles")
        profiles = [
            _normal_instance_profile_payload(profile)
            for profile in profiles_node or ()
            if profile.tag == "profile"
        ]
        if profiles:
            item["normal_instance_role_profiles"] = profiles
        output.append(item)
    return output


def _normal_instance_profile_payload(
    element: ElementTree.Element,
) -> dict[str, Any]:
    output = _attrs(
        element,
        attr_map={"subject": "subject_text"},
    )
    match_policy = _child_text(element, "match_policy")
    if match_policy:
        output["match_policy"] = match_policy
    roles_node = element.find("excluded_state_roles")
    excluded_roles = [
        {
            "role": str(role.get("role") or ""),
            "role_definition": str(role.get("definition") or ""),
        }
        for role in roles_node or ()
        if role.tag == "excluded_state"
    ]
    if excluded_roles:
        output["excluded_state_roles"] = excluded_roles
    return output


def _population_contract_payload(
    element: ElementTree.Element | None,
) -> dict[str, Any]:
    if element is None:
        return {}
    output = _attrs(element)
    axis_field = element.find("axis_field")
    if axis_field is not None:
        output["axis_field"] = _attrs(axis_field)
    omission = element.find("omission_behavior")
    if omission is not None:
        omission_payload = _attrs(omission)
        consequences = [
            _attrs(child)
            for child in omission
            if child.tag == "requested_fact_effect"
        ]
        if consequences:
            omission_payload["omission_consequence_by_requested_fact"] = consequences
        output["omission_behavior"] = omission_payload
    return output


def _response_row_payloads(
    element: ElementTree.Element,
    *,
    parent_path: str = "",
) -> list[dict[str, Any]]:
    row = _attrs(element)
    if parent_path:
        row["parent_path"] = parent_path
    fields = [
        _attrs(child, attr_map={"name": "field_id"})
        for child in element
        if child.tag == "field"
    ]
    row["fields"] = fields
    output = [row]
    row_path = str(row.get("path") or "")
    for child in element:
        if child.tag == "row":
            output.extend(_response_row_payloads(child, parent_path=row_path))
    return output


def _population_roles_payload(
    element: ElementTree.Element | None,
) -> list[dict[str, Any]]:
    if element is None:
        return []
    return [
        _attrs(
            child,
            attr_map={
                "id": "role_id",
                "row_path": "row_path_id",
                "kind": "role_kind",
                "text": "role_text",
            },
        )
        for child in element
        if child.tag == "role"
    ]


def _flat_fields_payload(
    element: ElementTree.Element | None,
    *,
    item_tag: str = "field",
    attr_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if element is None:
        return []
    names = {"name": "field_id", **(attr_map or {})}
    return [_attrs(child, attr_map=names) for child in element if child.tag == item_tag]


def _fulfillment_choices_payload(
    element: ElementTree.Element | None,
) -> list[dict[str, Any]]:
    if element is None:
        return []
    return [
        {
            **_attrs(
                choice,
                attr_map={
                    "id": "fulfillment_choice_id",
                    "answer_output": "answer_output_id",
                },
            ),
            "fulfillment_slots": _fulfillment_choice_slots(choice),
        }
        for choice in element
        if choice.tag == "choice"
    ]


def _fulfillment_choice_slots(element: ElementTree.Element) -> list[dict[str, Any]]:
    legacy_slots = [
        _fulfillment_slot_payload(slot) for slot in element if slot.tag == "slot"
    ]
    evidence_slots = [
        _fulfillment_evidence_payload(evidence)
        for evidence in element
        if evidence.tag == "evidence"
    ]
    return [*legacy_slots, *evidence_slots]


def _fulfillment_evidence_payload(element: ElementTree.Element) -> dict[str, Any]:
    kind = str(element.get("kind") or "")
    output = {
        "fulfillment_slot_id": str(element.get("evidence_id") or ""),
    }
    evidence_key = _fulfillment_evidence_key(kind)
    if evidence_key:
        output[evidence_key] = [
            _attrs(
                element,
                attr_map={
                    "evidence_id": "evidence_id",
                    "field": "field_id",
                    "row_path": "row_path_id",
                },
            )
        ]
    return output


def _fulfillment_evidence_key(kind: str) -> str:
    return {
        "scope": "scope_evidence",
        "metric": "metric_measure_evidence",
        "row_count_basis": "row_count_basis_evidence",
        "group_key": "group_key_evidence",
    }.get(kind, "")


def _fulfillment_slot_payload(element: ElementTree.Element) -> dict[str, Any]:
    output = _attrs(
        element,
        attr_map={
            "id": "fulfillment_slot_id",
            "answer_output": "answer_output_id",
            "basis": "compatibility_basis",
        },
    )
    for node_name, output_key, item_name in (
        ("scope_evidence", "scope_evidence", "scope"),
        ("metric_evidence", "metric_measure_evidence", "metric"),
        ("row_count_basis_evidence", "row_count_basis_evidence", "row_count_basis"),
        ("group_key_evidence", "group_key_evidence", "group_key"),
    ):
        container = element.find(node_name)
        items = [
            _attrs(
                item,
                attr_map={
                    "id": "evidence_id",
                    "field": "field_id",
                    "row_path": "row_path_id",
                },
            )
            for item in container or ()
            if item.tag == item_name
        ]
        if items:
            output[output_key] = items
    return output


def _generic_children_payloads(
    element: ElementTree.Element | None,
) -> list[dict[str, Any]]:
    if element is None:
        return []
    return [_attrs(child) for child in element]


def _attrs(
    element: ElementTree.Element,
    *,
    attr_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    names = attr_map or {}
    return {
        names.get(key, key): value
        for key, value in element.attrib.items()
        if value != ""
    }


def _bool_value(value: object) -> bool:
    return str(value).casefold() == "true"


def _split_words(value: object) -> list[str]:
    return [item for item in str(value or "").split() if item]


def _child_text(element: ElementTree.Element, tag: str) -> str:
    child = element.find(tag)
    if child is None:
        return ""
    return str(child.text or "").strip()
