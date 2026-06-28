"""Runtime data for deterministic relation operation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.lookup.fact_plan.operations import Operation
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.outcomes.errors import ExecutionIssue
from fervis.lookup.outcomes.model import Undefined


@dataclass(frozen=True)
class ScalarInput:
    id: str
    value: object
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationEngineInput:
    relations: tuple[RelationRows, ...] = ()
    operations: tuple[Operation, ...] = ()
    scalar_inputs: tuple[ScalarInput, ...] = ()
    operation_proof_refs: Mapping[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class RelationEngineOutput:
    relations: tuple[RelationRows, ...] = ()
    scalars: Mapping[str, object] | None = None
    scalar_proofs: Mapping[str, tuple[str, ...]] | None = None
    undefined: Undefined | None = None
    issue: ExecutionIssue | None = None

    def relation(self, relation_id: str) -> RelationRows:
        for relation in self.relations:
            if relation.id == relation_id:
                return relation
        raise RelationEngineError(f"unknown relation {relation_id}")
