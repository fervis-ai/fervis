"""Execution proof graph payload contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from fervis.lineage.enums import ProofEdgeRole, ProofNodeKind

TEnum = TypeVar("TEnum", bound=StrEnum)


EXECUTION_PROOF_GRAPH_SCHEMA = "fervis.execution_proof_graph"
EXECUTION_PROOF_GRAPH_SCHEMA_REV = 1


@dataclass(frozen=True)
class ProofGraphPayloadNode:
    id: str
    kind: ProofNodeKind
    proof_refs: tuple[str, ...] = ()
    label: str = ""
    value: Any = None
    operator: str = ""


@dataclass(frozen=True)
class ProofGraphPayloadEdge:
    source: str
    target: str
    role: ProofEdgeRole


@dataclass(frozen=True)
class ProofGraphPayloadContribution:
    origin: str
    label: str
    node_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofGraphPayload:
    nodes: tuple[ProofGraphPayloadNode, ...]
    edges: tuple[ProofGraphPayloadEdge, ...]
    contributions: tuple[ProofGraphPayloadContribution, ...] = ()


def execution_proof_graph_payload(graph: Any) -> dict[str, Any]:
    payload = {
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind.value,
                "proof_refs": list(node.proof_refs),
                **_optional_node_payload(node),
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "role": edge.role.value,
            }
            for edge in graph.edges
        ],
    }
    contributions = tuple(getattr(graph, "contributions", ()) or ())
    if contributions:
        payload["contributions"] = [
            {
                "origin": item.origin.value,
                "label": item.label,
                "node_refs": list(item.node_refs),
                "proof_refs": list(item.proof_refs),
            }
            for item in contributions
        ]
    return payload


def read_execution_proof_graph_payload(
    *,
    payload_schema: str,
    payload_schema_rev: int,
    payload_json: object,
) -> ProofGraphPayload:
    if payload_schema != EXECUTION_PROOF_GRAPH_SCHEMA:
        raise ValueError(f"unsupported proof graph payload schema {payload_schema!r}")
    if payload_schema_rev != EXECUTION_PROOF_GRAPH_SCHEMA_REV:
        raise ValueError(
            f"unsupported proof graph payload revision {payload_schema_rev!r}"
        )
    if not isinstance(payload_json, dict):
        raise ValueError("proof graph payload must be an object")
    nodes = payload_json.get("nodes")
    edges = payload_json.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("proof graph payload requires nodes and edges lists")
    contributions = payload_json.get("contributions", [])
    if not isinstance(contributions, list):
        raise ValueError("proof graph payload contributions must be a list")
    return ProofGraphPayload(
        nodes=tuple(_read_node(item) for item in nodes),
        edges=tuple(_read_edge(item) for item in edges),
        contributions=tuple(_read_contribution(item) for item in contributions),
    )


def _optional_node_payload(node: object) -> dict[str, Any]:
    output: dict[str, Any] = {}
    label = str(getattr(node, "label", "") or "")
    operator = str(getattr(node, "operator", "") or "")
    value = getattr(node, "value", None)
    if label:
        output["label"] = label
    if operator:
        output["operator"] = operator
    if value is not None:
        output["value"] = value
    return output


def _read_node(item: object) -> ProofGraphPayloadNode:
    if not isinstance(item, dict):
        raise ValueError("proof graph node must be an object")
    node_id = _required_text(item, "id")
    kind = _enum_value(
        ProofNodeKind,
        _required_text(item, "kind"),
        "proof graph node kind",
    )
    proof_refs = item.get("proof_refs", [])
    if not isinstance(proof_refs, list):
        raise ValueError("proof graph node proof_refs must be a list")
    return ProofGraphPayloadNode(
        id=node_id,
        kind=kind,
        proof_refs=tuple(str(ref) for ref in proof_refs if str(ref)),
        label=str(item.get("label") or ""),
        value=item.get("value"),
        operator=str(item.get("operator") or ""),
    )


def _read_edge(item: object) -> ProofGraphPayloadEdge:
    if not isinstance(item, dict):
        raise ValueError("proof graph edge must be an object")
    return ProofGraphPayloadEdge(
        source=_required_text(item, "source"),
        target=_required_text(item, "target"),
        role=_enum_value(
            ProofEdgeRole,
            _required_text(item, "role"),
            "proof graph edge role",
        ),
    )


def _read_contribution(item: object) -> ProofGraphPayloadContribution:
    if not isinstance(item, dict):
        raise ValueError("proof graph contribution must be an object")
    node_refs = item.get("node_refs", [])
    proof_refs = item.get("proof_refs", [])
    if not isinstance(node_refs, list):
        raise ValueError("proof graph contribution node_refs must be a list")
    if not isinstance(proof_refs, list):
        raise ValueError("proof graph contribution proof_refs must be a list")
    return ProofGraphPayloadContribution(
        origin=_required_text(item, "origin"),
        label=_required_text(item, "label"),
        node_refs=tuple(str(ref) for ref in node_refs if str(ref)),
        proof_refs=tuple(str(ref) for ref in proof_refs if str(ref)),
    )


def _required_text(item: dict[str, object], key: str) -> str:
    value = str(item.get(key) or "")
    if not value:
        raise ValueError(f"proof graph payload requires {key}")
    return value


def _enum_value(enum_type: type[TEnum], value: str, label: str) -> TEnum:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {label} {value!r}") from exc
