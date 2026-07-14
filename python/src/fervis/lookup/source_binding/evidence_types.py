"""Type predicates for source-binding evidence items."""

from __future__ import annotations

from fervis.lookup.source_binding.candidates.contracts import FieldEvidence


def evidence_item_can_measure(evidence_item: FieldEvidence) -> bool:
    return evidence_item.type.lower() in {
        "decimal",
        "float",
        "integer",
        "number",
        "numeric",
    }
