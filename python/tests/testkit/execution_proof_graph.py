"""Current execution-proof fixtures that still exercise the raw lineage contract."""

from __future__ import annotations

from typing import Any

from fervis.lineage.payloads.execution_proof_graph import (
    EXECUTION_PROOF_GRAPH_SCHEMA,
    EXECUTION_PROOF_GRAPH_SCHEMA_REV,
)
from fervis.lineage.recorder import ExecutionProofGraphWrite


def proof_node(
    node_id: str,
    kind: str,
    *,
    proof_refs: tuple[str, ...] = (),
    row_tests: tuple[dict[str, str], ...] = (),
    condition_tests: tuple[dict[str, str], ...] = (),
    **fields: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": kind,
        "proof_refs": list(proof_refs),
        "population_coverage": {
            "row_tests": list(row_tests),
            "condition_tests": list(condition_tests),
        },
        **fields,
    }


def proof_graph_payload(
    *,
    nodes: tuple[dict[str, Any], ...],
    edges: tuple[dict[str, str], ...] = (),
    contributions: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodes": list(nodes),
        "edges": list(edges),
    }
    if contributions:
        payload["contributions"] = list(contributions)
    return payload


def proof_graph_write(
    *,
    proof_graph_id: str,
    run_id: str,
    fact_result_id: str,
    compile_step_id: str,
    execute_step_id: str,
    payload_json: dict[str, Any],
) -> ExecutionProofGraphWrite:
    return ExecutionProofGraphWrite(
        **proof_graph_record(
            proof_graph_id=proof_graph_id,
            run_id=run_id,
            fact_result_id=fact_result_id,
            compile_step_id=compile_step_id,
            execute_step_id=execute_step_id,
            payload_json=payload_json,
        )
    )


def proof_graph_record(
    *,
    proof_graph_id: str,
    run_id: str,
    fact_result_id: str,
    compile_step_id: str,
    execute_step_id: str,
    payload_json: dict[str, Any],
) -> dict[str, Any]:
    return {
        "proof_graph_id": proof_graph_id,
        "run_id": run_id,
        "fact_result_id": fact_result_id,
        "compile_step_id": compile_step_id,
        "execute_step_id": execute_step_id,
        "payload_schema": EXECUTION_PROOF_GRAPH_SCHEMA,
        "payload_schema_rev": EXECUTION_PROOF_GRAPH_SCHEMA_REV,
        "payload_json": payload_json,
    }
