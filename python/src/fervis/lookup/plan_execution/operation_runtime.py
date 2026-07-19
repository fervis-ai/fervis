"""Runtime data for deterministic relation operation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AntiJoinSpec,
    ComputeSpec,
    CrossJoinSpec,
    FilterSpec,
    JoinSpec,
    OrderSpec,
    ProjectSpec,
    ProjectToKeySpec,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.outcomes.errors import ExecutionIssue
from fervis.lookup.outcomes.model import Undefined


@dataclass(frozen=True)
class ScalarInput:
    id: str
    value: RuntimeValue
    value_type: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedOperationInput:
    operation_id: str
    input_id: str
    value: RuntimeValue
    value_type: str = ""
    proof_refs: tuple[str, ...] = ()


ExecutableOperationSpec: TypeAlias = (
    FilterSpec
    | ProjectSpec
    | ProjectToKeySpec
    | JoinSpec
    | UnionSpec
    | RoleExpandSpec
    | CrossJoinSpec
    | AntiJoinSpec
    | UniversalConditionSpec
    | AggregateSpec
    | OrderSpec
    | ComputeSpec
)


@dataclass(frozen=True)
class ExecutableOperation:
    id: str
    spec: ExecutableOperationSpec
    output_relation: str = ""


@dataclass(frozen=True)
class RelationEngineInput:
    relations: tuple[RelationRows, ...] = ()
    operations: tuple[ExecutableOperation, ...] = ()
    scalar_inputs: tuple[ScalarInput, ...] = ()
    environment_values: Mapping[str, RuntimeValue] | None = None
    environment_types: Mapping[str, str] | None = None
    operation_proof_refs: Mapping[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class RelationEngineOutput:
    relations: tuple[RelationRows, ...] = ()
    scalars: Mapping[str, RuntimeValue] | None = None
    scalar_proofs: Mapping[str, tuple[str, ...]] | None = None
    scalar_types: Mapping[str, str] | None = None
    undefined: Undefined | None = None
    issue: ExecutionIssue | None = None

    def relation(self, relation_id: str) -> RelationRows:
        for relation in self.relations:
            if relation.id == relation_id:
                return relation
        raise RelationEngineError(f"unknown relation {relation_id}")
