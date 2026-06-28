import pytest

from fervis.memory.addresses import (
    FactAddress,
    FactAddressKind,
)
from fervis.memory.artifacts import FactOutcome
from fervis.memory.projection import fact_artifacts_from_context


def test_fact_address_rejects_incomplete_public_variant():
    with pytest.raises(ValueError, match="entity fact address requires"):
        FactAddress(
            address="entity.staff.missing_identity",
            kind=FactAddressKind.ENTITY,
            resource="staff",
        )


def test_fact_address_rejects_untyped_relation_source():
    with pytest.raises(ValueError, match="valid source.kind"):
        FactAddress.relation(
            address="relation.unknown",
            source={"kind": "untyped", "endpointName": "list_unknown"},
        )


def test_fact_address_rejects_non_terminal_outcome_address():
    with pytest.raises(ValueError, match="terminal outcome"):
        FactAddress.outcome(
            address="outcome.answered",
            terminal=FactOutcome.ANSWERED.value,
        )


def test_fact_artifacts_from_context_accepts_requested_fact_without_addresses():
    artifacts = fact_artifacts_from_context(
        {
            "factArtifacts": [
                {
                    "artifactId": "memory_requested_fact_1",
                    "outcome": "needs_clarification",
                }
            ]
        }
    )

    assert len(artifacts) == 1
    assert artifacts[0].addresses == ()
