"""Execution proof inputs projected for answer-program verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ExecutionProofSource(Protocol):
    @property
    def endpoint_arg_scope_refs(self) -> dict[str, frozenset[str]]: ...

    @property
    def operation_proof_refs(self) -> dict[str, tuple[str, ...]]: ...


@dataclass(frozen=True)
class ExecutionProofContext:
    endpoint_arg_scope_refs: dict[str, frozenset[str]]
    operation_refs: dict[str, frozenset[str]]

    @classmethod
    def empty(cls) -> "ExecutionProofContext":
        return cls(
            endpoint_arg_scope_refs={},
            operation_refs={},
        )

    @classmethod
    def from_materialized_execution(
        cls,
        materialized: ExecutionProofSource,
    ) -> "ExecutionProofContext":
        return cls(
            endpoint_arg_scope_refs=materialized.endpoint_arg_scope_refs,
            operation_refs={
                operation_id: frozenset(proof_refs)
                for operation_id, proof_refs in materialized.operation_proof_refs.items()
            },
        )
