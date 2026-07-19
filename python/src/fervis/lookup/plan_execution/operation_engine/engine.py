"""Operation engine dispatcher."""

from __future__ import annotations

from typing_extensions import assert_never

from fervis.lookup.plan_execution.operation_runtime import (
    RelationEngineError,
    RelationEngineInput,
    RelationEngineOutput,
    ScalarInput,
    ExecutableOperation,
    ResolvedRankSpec,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.outcomes.errors import (
    IncompleteEvidenceError,
    UndefinedOperationError,
)
from fervis.lookup.outcomes.model import Undefined
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AntiJoinSpec,
    CrossJoinSpec,
    ComputeSpec,
    FilterSpec,
    JoinSpec,
    ProjectSpec,
    ProjectToKeySpec,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)

from .aggregate_operations import _aggregate, _rank
from .compute_operations import _compute
from .relation_operations import (
    _anti_join,
    _cross_join,
    _filter,
    _join,
    _project,
    _project_to_key,
    _role_expand,
    _union,
    _universal_condition,
)
from .shared import (
    _input_relations,
    _input_scalar_proof_refs,
    _operation_proof_refs,
    _role_set_kind_refs,
    _with_role_set_kind,
)


def execute_operations(engine_input: RelationEngineInput) -> RelationEngineOutput:
    operation_proof_refs = dict(engine_input.operation_proof_refs or {})
    role_set_kind_refs = _role_set_kind_refs(engine_input.operations)
    relations: dict[str, RelationRows] = {}
    for relation in engine_input.relations:
        if relation.id in relations:
            raise RelationEngineError(f"duplicate relation {relation.id}")
        relation = _with_role_set_kind(relation, role_set_kind_refs.get(relation.id))
        relations[relation.id] = relation
    scalars: dict[str, RuntimeValue] = {}
    scalar_proofs: dict[str, tuple[str, ...]] = {}
    scalar_types: dict[str, str] = {}
    computed_outputs: dict[str, tuple[str, RuntimeValue]] = {}
    for scalar_input in engine_input.scalar_inputs:
        if not isinstance(scalar_input, ScalarInput):
            raise RelationEngineError("scalar input must be ScalarInput")
        if scalar_input.id in scalars:
            raise RelationEngineError(f"duplicate scalar {scalar_input.id}")
        scalars[scalar_input.id] = scalar_input.value
        scalar_proofs[scalar_input.id] = scalar_input.proof_refs
        scalar_types[scalar_input.id] = scalar_input.value_type
    for operation in engine_input.operations:
        if not isinstance(operation, ExecutableOperation):
            raise RelationEngineError("operation must be ExecutableOperation")
        try:
            result = _execute_operation(
                operation,
                relations,
                scalars,
                scalar_proofs,
                scalar_types,
                computed_outputs,
                operation_proof_refs=operation_proof_refs,
            )
        except IncompleteEvidenceError as exc:
            return RelationEngineOutput(
                relations=tuple(relations.values()),
                scalars=scalars,
                scalar_proofs=scalar_proofs,
                scalar_types=scalar_types,
                issue=exc.issue(),
            )
        except UndefinedOperationError as exc:
            proof_refs = _operation_proof_refs(
                operation,
                _input_relations(operation, relations),
                scalar_refs=(
                    *_input_scalar_proof_refs(operation, scalar_proofs),
                    *operation_proof_refs.get(operation.id, ()),
                ),
            )
            return RelationEngineOutput(
                relations=tuple(relations.values()),
                scalars=scalars,
                scalar_proofs=scalar_proofs,
                scalar_types=scalar_types,
                undefined=Undefined(
                    operation=exc.operation_ref(operation.id, proof_refs=proof_refs),
                    proof_refs=proof_refs,
                ),
            )
        if isinstance(result, RelationRows):
            result = _with_role_set_kind(result, role_set_kind_refs.get(result.id))
            if result.id in relations:
                raise RelationEngineError(f"duplicate relation {result.id}")
            relations[result.id] = result
        elif isinstance(operation.spec, ComputeSpec):
            output_scalar = operation.spec.output_scalar
            if output_scalar in scalars:
                raise RelationEngineError(f"duplicate scalar {output_scalar}")
            scalars[output_scalar] = result
            computed_outputs[operation.id] = (output_scalar, result)
            scalar_proofs[output_scalar] = _operation_proof_refs(
                operation,
                _input_relations(operation, relations),
                scalar_refs=(
                    *_input_scalar_proof_refs(operation, scalar_proofs),
                    *operation_proof_refs.get(operation.id, ()),
                ),
            )
            scalar_types[output_scalar] = "decimal"
        else:
            raise RelationEngineError(f"{operation.id} produced invalid result")
    return RelationEngineOutput(
        relations=tuple(relations.values()),
        scalars=scalars,
        scalar_proofs=scalar_proofs,
        scalar_types=scalar_types,
    )


def _execute_operation(
    operation: ExecutableOperation,
    relations: dict[str, RelationRows],
    scalars: dict[str, RuntimeValue],
    scalar_proofs: dict[str, tuple[str, ...]],
    scalar_types: dict[str, str],
    computed_outputs: dict[str, tuple[str, RuntimeValue]],
    *,
    operation_proof_refs: dict[str, tuple[str, ...]],
) -> RelationRows | RuntimeValue:
    spec = operation.spec
    if isinstance(spec, FilterSpec):
        return _filter(
            operation,
            spec,
            relations,
            scalars,
            scalar_proofs,
            scalar_types,
            operation_refs=operation_proof_refs.get(operation.id, ()),
        )
    if isinstance(spec, ProjectSpec):
        return _project(operation, spec, relations)
    if isinstance(spec, ProjectToKeySpec):
        return _project_to_key(operation, spec, relations)
    if isinstance(spec, JoinSpec):
        return _join(operation, spec, relations)
    if isinstance(spec, UnionSpec):
        return _union(operation, spec, relations)
    if isinstance(spec, RoleExpandSpec):
        return _role_expand(operation, spec, relations)
    if isinstance(spec, CrossJoinSpec):
        return _cross_join(operation, spec, relations)
    if isinstance(spec, AntiJoinSpec):
        return _anti_join(operation, spec, relations)
    if isinstance(spec, UniversalConditionSpec):
        return _universal_condition(
            operation,
            spec,
            relations,
            scalars,
            scalar_proofs,
            scalar_types,
            operation_refs=operation_proof_refs.get(operation.id, ()),
        )
    if isinstance(spec, AggregateSpec):
        return _aggregate(
            operation,
            spec,
            relations,
            operation_refs=operation_proof_refs.get(operation.id, ()),
        )
    if isinstance(spec, ResolvedRankSpec):
        return _rank(
            operation,
            spec,
            relations,
            operation_refs=operation_proof_refs.get(operation.id, ()),
        )
    if isinstance(spec, ComputeSpec):
        return _compute(
            spec,
            computed_outputs,
            scalars=scalars,
            scalar_types=scalar_types,
        )
    assert_never(spec)
