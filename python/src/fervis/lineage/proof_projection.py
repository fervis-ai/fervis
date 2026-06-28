"""Reachability projection for execution proof payloads."""

from __future__ import annotations

from fervis.lineage.payloads.execution_proof_graph import (
    ProofGraphPayload,
    ProofGraphPayloadContribution,
)


def project_proof_payload(
    payload: ProofGraphPayload,
    *,
    target_node_ids: tuple[str, ...],
) -> ProofGraphPayload:
    if not target_node_ids:
        return payload
    included_node_ids = _upstream_node_ids(payload, target_node_ids)
    included_nodes = tuple(node for node in payload.nodes if node.id in included_node_ids)
    included_edges = tuple(
        edge
        for edge in payload.edges
        if edge.source in included_node_ids and edge.target in included_node_ids
    )
    included_proof_refs = {
        proof_ref for node in included_nodes for proof_ref in node.proof_refs
    }
    return ProofGraphPayload(
        nodes=included_nodes,
        edges=included_edges,
        contributions=tuple(
            contribution
            for contribution in payload.contributions
            if _contribution_reaches_projection(
                contribution,
                node_ids=included_node_ids,
                proof_refs=included_proof_refs,
            )
        ),
    )


def _upstream_node_ids(
    payload: ProofGraphPayload,
    target_node_ids: tuple[str, ...],
) -> set[str]:
    known_node_ids = {node.id for node in payload.nodes}
    included = {node_id for node_id in target_node_ids if node_id in known_node_ids}
    changed = True
    while changed:
        changed = False
        for edge in payload.edges:
            if edge.target in included and edge.source not in included:
                included.add(edge.source)
                changed = True
    return included


def _contribution_reaches_projection(
    contribution: ProofGraphPayloadContribution,
    *,
    node_ids: set[str],
    proof_refs: set[str],
) -> bool:
    contribution_node_refs = set(contribution.node_refs)
    if contribution_node_refs:
        return bool(contribution_node_refs & node_ids)
    return bool(set(contribution.proof_refs) & proof_refs)
