from __future__ import annotations

import pytest

from fervis.lineage.enums import ProofEdgeRole, ProofNodeKind
from fervis.lineage.payloads.execution_proof_graph import (
    EXECUTION_PROOF_GRAPH_SCHEMA,
    EXECUTION_PROOF_GRAPH_SCHEMA_REV,
    ProofGraphPayload,
    ProofGraphPayloadContribution,
    ProofGraphPayloadEdge,
    ProofGraphPayloadNode,
    read_execution_proof_graph_payload,
)
from fervis.lineage.proof_projection import project_proof_payload


def test_execution_proof_graph_payload_rejects_unknown_node_kind() -> None:
    with pytest.raises(ValueError, match="unsupported proof graph node kind"):
        read_execution_proof_graph_payload(
            payload_schema=EXECUTION_PROOF_GRAPH_SCHEMA,
            payload_schema_rev=EXECUTION_PROOF_GRAPH_SCHEMA_REV,
            payload_json={
                "nodes": [{"id": "n1", "kind": "unknown_kind"}],
                "edges": [],
            },
        )


def test_execution_proof_graph_payload_rev_1_accepts_population_choice_node() -> None:
    payload = read_execution_proof_graph_payload(
        payload_schema=EXECUTION_PROOF_GRAPH_SCHEMA,
        payload_schema_rev=EXECUTION_PROOF_GRAPH_SCHEMA_REV,
        payload_json={
            "nodes": [
                {
                    "id": "population_choice:source_1:row_predicate:status",
                    "kind": "population_choice",
                    "label": "Included status values [OPEN]",
                }
            ],
            "edges": [],
        },
    )

    assert payload.nodes[0].kind.value == "population_choice"


def test_execution_proof_graph_payload_rejects_unknown_edge_role() -> None:
    with pytest.raises(ValueError, match="unsupported proof graph edge role"):
        read_execution_proof_graph_payload(
            payload_schema=EXECUTION_PROOF_GRAPH_SCHEMA,
            payload_schema_rev=EXECUTION_PROOF_GRAPH_SCHEMA_REV,
            payload_json={
                "nodes": [
                    {"id": "n1", "kind": "relation"},
                    {"id": "n2", "kind": "answer_output"},
                ],
                "edges": [
                    {"source": "n1", "target": "n2", "role": "unknown_role"},
                ],
            },
        )


def test_proof_projection_does_not_keep_sibling_contribution_by_shared_proof_ref() -> (
    None
):
    payload = ProofGraphPayload(
        nodes=(
            ProofGraphPayloadNode(
                id="relation:first",
                kind=ProofNodeKind.RELATION,
                proof_refs=("source_read:shared",),
            ),
            ProofGraphPayloadNode(
                id="relation:second",
                kind=ProofNodeKind.RELATION,
                proof_refs=("source_read:shared",),
            ),
            ProofGraphPayloadNode(
                id="answer_output:fact:first",
                kind=ProofNodeKind.ANSWER_OUTPUT,
            ),
            ProofGraphPayloadNode(
                id="answer_output:fact:second",
                kind=ProofNodeKind.ANSWER_OUTPUT,
            ),
        ),
        edges=(
            ProofGraphPayloadEdge(
                source="relation:first",
                target="answer_output:fact:first",
                role=ProofEdgeRole.PRODUCES,
            ),
            ProofGraphPayloadEdge(
                source="relation:second",
                target="answer_output:fact:second",
                role=ProofEdgeRole.PRODUCES,
            ),
        ),
        contributions=(
            ProofGraphPayloadContribution(
                origin="explicit",
                label="first branch",
                node_refs=("relation:first",),
                proof_refs=("source_read:shared",),
            ),
            ProofGraphPayloadContribution(
                origin="explicit",
                label="second branch",
                node_refs=("relation:second",),
                proof_refs=("source_read:shared",),
            ),
        ),
    )

    projected = project_proof_payload(
        payload,
        target_node_ids=("answer_output:fact:first",),
    )

    assert [item.label for item in projected.contributions] == ["first branch"]
