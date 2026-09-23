from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.values import ValueProjectionKind
from fervis.lookup.source_binding.param_binding_sets import (
    RelationInputOrigin,
    alternate_param_binding_sets,
    coalesce_equivalent_param_binding_sets,
    combine_param_binding_sets,
    intersect_param_binding_sets,
    parameter_binding_sets,
)
from fervis.lookup.source_binding.param_values import fact_value_parameter_projection
from tests.testkit.answer_program_contracts import fact_value_from_payload
from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)


def run_parameter_binding_alternatives_case(payload: dict[str, Any]) -> list[str]:
    try:
        groups = tuple(
            _parameter_group(item)
            for item in payload["input"].get("parameter_groups") or ()
        )
        groups += tuple(
            _projected_parameter_constraints(item)
            for item in payload["input"].get("projected_parameter_constraints") or ()
        )
        constraints = payload["input"].get("same_target_constraints", ())
        groups += tuple(
            intersect_param_binding_sets(
                tuple(
                    _parameter_group(
                        {
                            "target_id": item["target_id"],
                            "type": item["type"],
                            "alternative_values": values,
                            "value_id_prefix": f"constraint_{index}_",
                        }
                    )
                    for index, values in enumerate(
                        item["allowed_values_by_constraint"], start=1
                    )
                )
            )
            for item in constraints
        )
        binding_sets = combine_param_binding_sets(groups)
        if any(not group for group in groups):
            raise ValueError("parameter constraints conflict")
    except ValueError:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected", expected=payload["expect"]
            )
        raise
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    actual = {
        "binding_sets": [
            {
                binding.param_id: {
                    "value": binding.compiler_value,
                    "proof_refs": list(binding.proof_refs),
                }
                for binding in binding_set
            }
            for binding_set in binding_sets
        ]
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _parameter_group(item: dict[str, Any]):
    return alternate_param_binding_sets(
        coalesce_equivalent_param_binding_sets(
            parameter_binding_sets(
                param_id=str(item["target_id"]),
                value=value,
                parameter_type=str(item["type"]),
                origin_kind=RelationInputOrigin.PLAN_CONTROL,
                proof_refs=(f"application:{index}",),
                value_id=(
                    f"{item['value_id_prefix']}{index}"
                    if item.get("value_id_prefix")
                    else ""
                ),
            )
            for index, value in enumerate(item["alternative_values"], start=1)
        )
    )


def _projected_parameter_constraints(item: dict[str, Any]):
    groups = tuple(
        parameter_binding_sets(
            param_id=str(item["target_id"]),
            value=fact_value_parameter_projection(
                fact_value_from_payload(
                    value_payload,
                    value_id=str(value_payload.get("id") or f"value_{index}"),
                ),
                projection=ValueProjectionKind(str(item["projection"])),
                component_id=(
                    str(item["component_id"]) if item.get("component_id") else None
                ),
                type_name=str(item["type"]),
                choices=tuple(str(value) for value in item.get("choices") or ()),
            ),
            parameter_type=str(item["type"]),
            origin_kind=RelationInputOrigin.QUESTION_INPUT,
            value_id=str(value_payload.get("id") or f"value_{index}"),
            proof_refs=tuple(str(ref) for ref in value_payload.get("proof_refs") or ()),
        )
        for index, value_payload in enumerate(item["values"], start=1)
    )
    distinct = coalesce_equivalent_param_binding_sets(groups)
    return intersect_param_binding_sets(distinct)


__all__ = ["run_parameter_binding_alternatives_case"]
