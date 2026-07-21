"""Validate bounded answer-output-owned recall for a grouped payroll question."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    facts = arguments.get("requested_fact_resource_name_matches") or []
    if len(facts) != 1:
        return [f"expected one requested fact, got {len(facts)}"]
    rows = facts[0].get("answer_output_resource_lineage") or []
    errors: list[str] = []
    ids = [row.get("answer_output_id") for row in rows]
    if ids != ["group_key", "answer_1", "answer_2"]:
        errors.append(f"unexpected output coverage: {ids!r}")
    measure_terms = set(
        next(
            (
                row.get("matching_resource_names") or []
                for row in rows
                if row.get("answer_output_id") == "answer_2"
            ),
            [],
        )
    )
    if not measure_terms & {"shift compensation", "shift compensations"}:
        errors.append("payroll measure omitted shift-compensation resources")
    return errors
