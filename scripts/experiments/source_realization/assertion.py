"""Assert selected row, field, and relationship outcomes without rebuilding inputs."""
from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    expected_maps = {
        "set_bindings": ("expected_row_refs", "rows_ref"),
        "fact_bindings": ("expected_field_refs", "field_ref"),
        "association_bindings": ("expected_relation_refs", "realization_ref"),
    }
    if not any(context.get(key) for key, _ in expected_maps.values()) and not (
        context.get("expected_field_pairs") or context.get("expected_carrier_options")
        or context.get("expected_proxy_fields")
    ):
        return ["source realization assertion requires expected outcomes"]
    errors = []
    for section, (expected_key, field) in expected_maps.items():
        actual = arguments.get(section)
        if not isinstance(actual, dict):
            errors.append(f"{section} is missing")
            continue
        for ref, expected in context.get(expected_key, {}).items():
            values = actual.get(ref)
            if not isinstance(values, list):
                errors.append(f"realization is missing: {ref}")
                continue
            observed = [value.get(field) for value in values if isinstance(value, dict)]
            if observed != expected:
                errors.append(f"{ref} selects {observed!r}; expected {expected!r}")
    for ref, expected in context.get("expected_field_pairs", {}).items():
        bindings = (arguments.get("association_bindings") or {}).get(ref)
        if not isinstance(bindings, list):
            errors.append(f"observed association is missing: {ref}")
            continue
        actual = [
            [
                (pair.get("from_field_ref"), pair.get("to_field_ref"))
                for pair in binding.get("field_pairs", [])
            ]
            for binding in bindings
            if isinstance(binding, dict)
        ]
        if actual != [[tuple(pair) for pair in values] for values in expected]:
            errors.append(f"{ref} selects incompatible observed equality fields")
    for ref, expected in context.get("expected_proxy_fields", {}).items():
        values = (arguments.get("set_bindings") or {}).get(ref)
        selected = [
            value.get("reference_proxy_field_ref")
            for value in values or () if isinstance(value, dict)
        ]
        if selected != expected:
            errors.append(f"{ref} does not project the required proxy value")
    for set_ref, options in context.get("expected_carrier_options", {}).items():
        set_bindings = (arguments.get("set_bindings") or {}).get(set_ref)
        if not isinstance(set_bindings, list) or len(set_bindings) != 1:
            errors.append(f"{set_ref} lacks one selected carrier")
            continue
        selected = set_bindings[0]
        if not isinstance(selected, dict):
            errors.append(f"{set_ref} has an invalid carrier")
            continue
        fields = options.get(selected.get("rows_ref"))
        if not isinstance(fields, dict):
            errors.append(f"{set_ref} selects a carrier outside the expected population")
            continue
        branch_id = selected.get("branch_id")
        for fact_ref, expected_fields in fields.items():
            selected_facts = (arguments.get("fact_bindings") or {}).get(fact_ref)
            actual_fields = [
                fact.get("field_ref")
                for fact in selected_facts or ()
                if isinstance(fact, dict) and fact.get("branch_id") == branch_id
            ]
            if actual_fields != expected_fields:
                errors.append(f"{fact_ref} is not owned by the selected carrier")
    return errors
