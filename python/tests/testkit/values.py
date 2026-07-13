from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.values import FactValue


def fact_value_from_payload(payload: dict[str, Any]) -> FactValue:
    if payload.get("kind") == "identity":
        return FactValue.identity(
            id=str(payload["id"]),
            known_input_id=str(payload.get("known_input_id") or ""),
            entity_kind=str(payload["entity_kind"]),
            key_id=str(payload["key_id"]),
            key_component_id=str(payload["key_component_id"]),
            value=str(payload["value"]),
            display_value=str(payload.get("display_value") or ""),
            proof_refs=tuple(payload.get("proof_refs") or ()),
            applies_to_requested_fact_ids=tuple(
                payload.get("applies_to_requested_fact_ids") or ()
            ),
        )
    raise ValueError(f"unsupported fact value kind: {payload.get('kind')!r}")
