"""Execution-proof interpretation for lineage views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fervis.lineage.enums import (
    ContributionOrigin,
    ProofEdgeRole,
    ProofNodeKind,
)
from fervis.lineage.payloads.execution_proof_graph import (
    ProofGraphPayload,
    ProofGraphPayloadEdge,
)


@dataclass(frozen=True)
class ProofContributionSummary:
    origin: ContributionOrigin
    label: str
    node_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofEndpointArgSummary:
    handle: str
    arg_name: str
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofAppliedInputSummary:
    handle: str
    kind: ProofNodeKind
    label: str
    action: str
    proof_refs: tuple[str, ...] = ()


class ProofSourceRead(Protocol):
    source_read_id: str
    args: dict[str, object]


def proof_source_read_ids(payload: ProofGraphPayload) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                proof_ref.removeprefix("source_read:")
                for node in payload.nodes
                for proof_ref in node.proof_refs
                if proof_ref.startswith("source_read:")
            }
        )
    )


def proof_endpoint_args(
    payload: ProofGraphPayload, *, source_reads: tuple[ProofSourceRead, ...]
) -> tuple[ProofEndpointArgSummary, ...]:
    return tuple(
        ProofEndpointArgSummary(
            handle=node.id,
            arg_name=proof_node_label(node.id),
            values=(_render_value(node.value),) if node.value is not None else (),
        )
        for node in payload.nodes
        if node.kind is ProofNodeKind.ENDPOINT_ARG
    )


def proof_applied_inputs(
    payload: ProofGraphPayload,
) -> tuple[ProofAppliedInputSummary, ...]:
    return tuple(
        ProofAppliedInputSummary(
            handle=node.id,
            kind=node.kind,
            label=node.label or proof_node_label(node.id),
            action=_applied_action(node.kind),
            proof_refs=node.proof_refs,
        )
        for node in payload.nodes
        if node.kind
        in {
            ProofNodeKind.ENDPOINT_ARG,
            ProofNodeKind.POPULATION_CHOICE,
            ProofNodeKind.ROW_FILTER,
            ProofNodeKind.OPERATION_INPUT,
        }
        and (node.label or node.value is not None)
    )


def proof_contributions(
    payload: ProofGraphPayload,
) -> tuple[ProofContributionSummary, ...]:
    return tuple(
        ProofContributionSummary(
            origin=ContributionOrigin(item.origin),
            label=item.label,
            node_refs=item.node_refs,
            proof_refs=item.proof_refs,
        )
        for item in payload.contributions
    )


def proof_evidence_handle_labels(payload: ProofGraphPayload) -> tuple[str, ...]:
    return tuple(proof_node_debug_label(node.id) for node in payload.nodes)


def proof_computation_summaries(
    edges: tuple[ProofGraphPayloadEdge, ...],
) -> tuple[str, ...]:
    output: list[str] = []
    if any(
        edge.role is ProofEdgeRole.INPUT and _is_operation_edge(edge) for edge in edges
    ):
        output.append("source rows were used as computation input")
    if any(
        edge.role is ProofEdgeRole.PRODUCES and _is_operation_edge(edge)
        for edge in edges
    ):
        output.append("the computation produced derived rows")
    if any(
        edge.role is ProofEdgeRole.PRODUCES and edge.target.startswith("answer_output:")
        for edge in edges
    ):
        output.append("derived rows produced the answer output")
    return tuple(output)


def proof_computation_link_labels(
    edges: tuple[ProofGraphPayloadEdge, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{edge.source} -> {edge.target} ({edge.role.value})" for edge in edges
    )


def proof_node_label(node_id: str) -> str:
    return node_id.rsplit(".", 1)[-1].rsplit(":", 1)[-1]


def proof_node_debug_label(node_id: str) -> str:
    if node_id.startswith("endpoint_arg:"):
        return f"applied {proof_node_label(node_id)}"
    return node_id


def _applied_action(kind: ProofNodeKind) -> str:
    if kind is ProofNodeKind.ENDPOINT_ARG:
        return "was used as an endpoint argument."
    if kind is ProofNodeKind.POPULATION_CHOICE:
        return "was reviewed as a population choice."
    if kind is ProofNodeKind.ROW_FILTER:
        return "was applied as a row filter."
    if kind is ProofNodeKind.OPERATION_INPUT:
        return "was used as a computation input."
    return "was applied."


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _is_operation_edge(edge: ProofGraphPayloadEdge) -> bool:
    return edge.source.startswith("operation:") or edge.target.startswith("operation:")
