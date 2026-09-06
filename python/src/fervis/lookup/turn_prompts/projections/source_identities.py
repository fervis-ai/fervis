"""Model-facing projection of declared source identity authority."""

from __future__ import annotations

def source_identity_evidence_prompt_items(
    catalog,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "identity_ref": item.identity_ref,
            "source_ref": item.source_ref,
            "identity_kind": item.kind.value,
            "entity_kind": item.entity_kind,
            "key_id": item.key_id,
            "field_refs": list(item.field_refs),
        }
        for item in catalog.identity_evidence
    )


__all__ = ["source_identity_evidence_prompt_items"]
