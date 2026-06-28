"""Execution proof inputs projected for fact-plan verification."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.plan_execution.compiled_execution import (
    CompiledFactExecution,
)


@dataclass(frozen=True)
class ExecutionProofContext:
    endpoint_arg_scope_refs: dict[str, frozenset[str]]
    operation_refs: dict[str, frozenset[str]]
    row_filter_scope_refs: dict[str, frozenset[str]]

    @classmethod
    def from_compiled_execution(
        cls,
        compiled: CompiledFactExecution,
    ) -> "ExecutionProofContext":
        return cls(
            endpoint_arg_scope_refs=compiled.endpoint_arg_scope_refs,
            operation_refs={
                operation_id: frozenset(proof_refs)
                for operation_id, proof_refs in compiled.operation_proof_refs.items()
            },
            row_filter_scope_refs=compiled.row_filter_scope_refs,
        )
