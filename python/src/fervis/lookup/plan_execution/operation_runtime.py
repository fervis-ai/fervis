"""Runtime data for deterministic relation operation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, TypeAlias

from fervis.lookup.plan_execution.errors import RelationEngineError, VerificationError
from fervis.lookup.answer_program.operations import (
    SqlQuerySpec,
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
    SqlQuerySpec
    | FilterSpec
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
    source_relation_ids: tuple[str, ...] = ()
    relation_loader: Callable[[str, Callable[[str], RelationRows]], RelationRows] | None = None


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


def scalar_inputs_for_operations(inputs: tuple[ResolvedOperationInput, ...]) -> tuple[ScalarInput, ...]:
    by_id: dict[str, ScalarInput] = {}
    for item in inputs:
        candidate = ScalarInput(
            id=item.input_id,
            value=item.value,
            value_type=item.value_type,
            proof_refs=item.proof_refs,
        )
        existing = by_id.get(candidate.id)
        if existing is not None and (
            existing.value != candidate.value
            or existing.value_type != candidate.value_type
        ):
            raise VerificationError(f"conflicting operation input {candidate.id}")
        if existing is None:
            by_id[candidate.id] = candidate
            continue
        by_id[candidate.id] = ScalarInput(
            id=candidate.id,
            value=candidate.value,
            value_type=candidate.value_type,
            proof_refs=tuple(
                dict.fromkeys((*existing.proof_refs, *candidate.proof_refs))
            ),
        )
    return tuple(by_id.values())


def operation_input_proofs(inputs: tuple[ResolvedOperationInput, ...]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item in inputs:
        grouped.setdefault(item.operation_id,[]).extend(item.proof_refs)
    return {key:tuple(dict.fromkeys(refs)) for key,refs in grouped.items()}
