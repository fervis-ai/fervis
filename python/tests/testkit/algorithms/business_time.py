from __future__ import annotations

from datetime import date
from typing import Any

from fervis.lookup.grounding.time_resolution import resolve_time

from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)


def run_business_time_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        result = resolve_time(
            str(input_payload.get("expression") or ""),
            intent=input_payload.get("intent"),
            anchor_date=_date_value(input_payload.get("anchor_date")),
            timezone=str(input_payload.get("timezone") or "UTC"),
        )
    except Exception as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    return subset_mismatches(
        actual=result,
        expected_subset=payload["expect"]["result_contains"],
    )


def _date_value(value: Any) -> date | str | None:
    if value is None:
        return None
    return str(value)
