from __future__ import annotations

from pathlib import Path

import fervis
from fervis.lookup.memory.projection import project_conversation_memory_cards
from fervis.memory.addresses import (
    FactAddress,
    RelationSourceKind,
)
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)


def test_fervis_core_conversation_resolution_uses_synthetic_framework_memory_not_retail_ops():
    artifact = build_fact_artifact(
        artifact_id="synthetic_turn",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.relation(
                address="relation.synthetic_rows",
                source={"kind": RelationSourceKind.API_READ.value},
                row_addresses=("row.synthetic",),
            ),
            FactAddress.row(
                address="row.synthetic", relation="relation.synthetic_rows"
            ),
        ),
    )

    projection = project_conversation_memory_cards(
        {"factArtifacts": [artifact.to_dict()]},
        current_question="How about those records?",
    )

    assert projection.cards
    assert (
        "retail_ops"
        not in repr([card.to_model_dict() for card in projection.cards]).casefold()
    )


def test_conversation_resolution_no_semantic_oracle_suffix_or_domain_heuristics():
    root = Path(fervis.__file__).resolve().parent
    assert root.is_dir()
    matches = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for banned in ('endswith("_id")', "endswith('_id')"):
            if banned in text:
                matches.append(f"{path}:{banned}")

    assert matches == []
