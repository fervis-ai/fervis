from __future__ import annotations

from typing import Any

from fervis.lookup.grounding.model import (
    GroundedValueCertification,
    GroundedValueCertificationMethod,
)

from tests.testkit.assertions import subset_mismatches


def run_grounding_contract_case(payload: dict[str, Any]) -> list[str]:
    actual = {
        "certifications": [
            GroundedValueCertification(
                value_id=str(item["value_id"]),
                method=GroundedValueCertificationMethod(str(item["method"])),
                authority_refs=tuple(
                    str(ref) for ref in item.get("authority_refs") or ()
                ),
                lineage_refs=tuple(str(ref) for ref in item.get("lineage_refs") or ()),
            ).to_payload()
            for item in payload["input"].get("certifications") or ()
        ]
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )
