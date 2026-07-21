"""Shared operation-engine primitives."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Iterable

from fervis.lookup.plan_execution.operation_runtime import RelationEngineError
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessSourceKind,
    CompletenessStatus,
    PaginationCompleteness,
    RelationEvidence,
    RelationSetKind,
    RelationRows,
    Row,
    relation_snapshot_hash,
)
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    AntiJoinSpec,
    CrossJoinSpec,
    ComputeSpec,
    FilterSpec,
    JoinKey,
    JoinSpec,
    OrderSpec,
    Predicate,
    NamedExpression,
    ProjectSpec,
    ProjectToKeySpec,
    RelationRole,
    RelationRoleRef,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
)
from fervis.lookup.answer_program.expressions import expression_references
from fervis.lookup.answer_program.expressions import expression_input_id
from fervis.lookup.outcomes.errors import UndefinedOperationError
from fervis.lookup.outcomes.operation_semantics import (
    empty_aggregation_undefined_reason,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.plan_execution.declared_values import (
    declared_equal,
    declared_number,
    declared_order_key,
)


def _role_relation(
    relations: dict[str, RelationRows],
    ref: RelationRoleRef,
    expected_role: RelationRole,
) -> RelationRows:
    if ref.role != expected_role:
        raise RelationEngineError(f"{expected_role.value} requires matching role")
    if not ref.required_identity_fields:
        raise RelationEngineError(f"{ref.role.value} requires grain obligation")
    relation = _relation(relations, ref.relation_id)
    _require_completeness(relation, ref)
    _require_set_kind(relation, expected_role)
    _require_grain(relation, ref.required_identity_fields, ref.role.value)
    return relation


def _require_completeness(relation: RelationRows, ref: RelationRoleRef) -> None:
    if relation.completeness.status != CompletenessStatus.COMPLETE:
        raise RelationEngineError(f"{ref.role.value} relation must be complete")


def _require_set_kind(relation: RelationRows, role: RelationRole) -> None:
    expected = _expected_set_kind(role)
    if relation.completeness.set_kind != expected:
        raise RelationEngineError(f"{role.value} requires {expected.value} relation")


def _expected_set_kind(role: RelationRole) -> RelationSetKind:
    if role in {
        RelationRole.ANTI_JOIN_CANDIDATE,
        RelationRole.UNIVERSAL_CANDIDATE_SUBJECT,
        RelationRole.UNIVERSAL_REQUIRED_DIMENSION,
    }:
        return RelationSetKind.UNIVERSE
    if role in {
        RelationRole.ANTI_JOIN_OBSERVED,
        RelationRole.UNIVERSAL_OBSERVATION,
    }:
        return RelationSetKind.OBSERVATION
    raise RelationEngineError(f"unsupported relation role {role.value}")


def _require_grain(
    relation: RelationRows,
    fields: tuple[str, ...],
    label: str,
) -> None:
    if tuple(relation.grain_keys) != tuple(fields):
        raise RelationEngineError(f"{label} grain must exactly match role grain")


def _project_output(
    row: Row,
    fields: tuple[NamedExpression, ...],
    *,
    grain_keys: tuple[str, ...] = (),
) -> dict[str, RuntimeValue]:
    output: dict[str, RuntimeValue] = {}
    for grain_field in grain_keys:
        _assign_or_match(output, grain_field, _field(row, grain_field))
    for field in fields:
        if not isinstance(field.expression, FieldRef):
            raise RelationEngineError(
                "role projection requires direct field expressions"
            )
        _assign_or_match(
            output,
            field.output_field,
            _field(row, field.expression.field_id),
        )
    return output


def _operation_relation(
    operation: ExecutableOperation,
    rows: list[dict[str, RuntimeValue]] | tuple[dict[str, RuntimeValue], ...],
    *,
    grain_keys: tuple[str, ...],
    inputs: tuple[RelationRows, ...],
    scalar_refs: tuple[str, ...] = (),
    field_types: dict[str, str] | None = None,
) -> RelationRows:
    _require_rows_have_fields(rows, grain_keys, "operation grain")
    complete = all(
        relation.completeness.status == CompletenessStatus.COMPLETE
        for relation in inputs
    )
    scope_fingerprint = _combined_scope(inputs)
    return RelationRows(
        id=_output_relation(operation),
        rows=tuple(rows),
        grain_keys=grain_keys,
        field_types=(
            dict(field_types)
            if field_types is not None
            else _projected_field_types(tuple(rows), inputs=inputs)
        ),
        field_answer_output_ids=_projected_field_answer_output_ids(
            tuple(rows),
            inputs=inputs,
        ),
        evidence=RelationEvidence(
            source_refs=tuple(
                dict.fromkeys(
                    ref for relation in inputs for ref in relation.evidence.source_refs
                )
            ),
            read_refs=tuple(
                dict.fromkeys(
                    ref for relation in inputs for ref in relation.evidence.read_refs
                )
            ),
            authority_refs=tuple(
                dict.fromkeys(
                    ref
                    for relation in inputs
                    for ref in relation.evidence.authority_refs
                )
            ),
            snapshot_hash=relation_snapshot_hash(tuple(rows)),
            proof_refs=_operation_proof_refs(
                operation,
                inputs,
                scalar_refs=scalar_refs,
            ),
        ),
        completeness=CompletenessProof(
            status=(
                CompletenessStatus.COMPLETE if complete else CompletenessStatus.UNKNOWN
            ),
            source_kind=CompletenessSourceKind.OPERATION_OUTPUT,
            set_kind=_combined_set_kind(inputs),
            scope_fingerprint=scope_fingerprint,
            pagination=(
                PaginationCompleteness.NOT_PAGINATED
                if complete
                else PaginationCompleteness.UNKNOWN
            ),
            proof_refs=_operation_proof_refs(
                operation,
                inputs,
                scalar_refs=scalar_refs,
            ),
        ),
    )


def _project_grain(
    input_relation: RelationRows,
    fields: tuple[NamedExpression, ...],
) -> tuple[str, ...]:
    if not input_relation.grain_keys:
        return ()
    projections = {
        field.expression.field_id: field.output_field
        for field in fields
        if isinstance(field.expression, FieldRef)
    }
    if not all(field in projections for field in input_relation.grain_keys):
        return ()
    return tuple(projections[field] for field in input_relation.grain_keys)


def _operation_proof_refs(
    operation: ExecutableOperation,
    inputs: tuple[RelationRows, ...],
    *,
    scalar_refs: tuple[str, ...] = (),
) -> tuple[str, ...]:
    refs: list[str] = []
    for relation in inputs:
        for ref in relation.completeness.proof_refs:
            if ref not in refs:
                refs.append(ref)
    for ref in scalar_refs:
        if ref not in refs:
            refs.append(ref)
    if operation.id not in refs:
        refs.append(operation.id)
    return tuple(refs)


def _input_scalar_proof_refs(
    operation: ExecutableOperation,
    scalar_proofs: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    scalar_ids = _operation_scalar_refs(operation)
    refs: list[str] = []
    for scalar_id in scalar_ids:
        for ref in scalar_proofs.get(scalar_id, ()):
            if ref not in refs:
                refs.append(ref)
    return tuple(refs)


def _operation_scalar_refs(operation: ExecutableOperation) -> tuple[str, ...]:
    spec = operation.spec
    if isinstance(spec, ComputeSpec):
        references = expression_references(spec.expression)
        return (
            *(item.output_id for item in references.outputs),
            *(expression_input_id(item) for item in references.parameters),
            *(expression_input_id(item) for item in references.constants),
        )
    if isinstance(spec, FilterSpec):
        return _predicate_scalar_refs(spec.predicate)
    if isinstance(spec, UniversalConditionSpec):
        return _predicate_scalar_refs(spec.predicate)
    return ()


def _predicate_scalar_refs(predicate: Predicate) -> tuple[str, ...]:
    if predicate.right is None:
        return ()
    references = expression_references(predicate.right)
    return (
        *(item.output_id for item in references.outputs),
        *(expression_input_id(item) for item in references.parameters),
        *(expression_input_id(item) for item in references.constants),
    )


def _input_relations(
    operation: ExecutableOperation,
    relations: dict[str, RelationRows],
) -> tuple[RelationRows, ...]:
    spec = operation.spec
    if isinstance(
        spec,
        (
            FilterSpec,
            ProjectSpec,
            OrderSpec,
            ProjectToKeySpec,
            RoleExpandSpec,
            AggregateSpec,
            OrderSpec,
        ),
    ):
        return (_relation(relations, spec.input_relation),)
    if isinstance(spec, UniversalConditionSpec):
        return (
            _relation(relations, spec.candidate_subject.relation_id),
            _relation(relations, spec.required_dimension.relation_id),
            _relation(relations, spec.observation.relation_id),
        )
    if isinstance(spec, AntiJoinSpec):
        return (
            _relation(relations, spec.candidate.relation_id),
            _relation(relations, spec.observed.relation_id),
        )
    if isinstance(spec, (JoinSpec, CrossJoinSpec)):
        return (_relation(relations, spec.left), _relation(relations, spec.right))
    if isinstance(spec, UnionSpec):
        return tuple(_relation(relations, relation_id) for relation_id in spec.inputs)
    return ()


def _combined_set_kind(relations: tuple[RelationRows, ...]) -> RelationSetKind:
    kinds = {relation.completeness.set_kind for relation in relations}
    if len(kinds) == 1:
        return next(iter(kinds))
    return RelationSetKind.UNKNOWN


def _combined_scope(relations: tuple[RelationRows, ...]) -> str:
    scopes: list[str] = []
    for relation in relations:
        scope = relation.completeness.scope_fingerprint
        if scope and scope not in scopes:
            scopes.append(scope)
    return "|".join(scopes)


def _role_set_kind_refs(
    operations: tuple[ExecutableOperation, ...],
) -> dict[str, RelationRoleRef]:
    refs: dict[str, RelationRoleRef] = {}
    for operation in operations:
        spec = operation.spec
        if isinstance(spec, AntiJoinSpec):
            _assign_role_set_kind_ref(refs, spec.candidate)
            _assign_role_set_kind_ref(refs, spec.observed)
        elif isinstance(spec, UniversalConditionSpec):
            _assign_role_set_kind_ref(refs, spec.candidate_subject)
            _assign_role_set_kind_ref(refs, spec.required_dimension)
            _assign_role_set_kind_ref(refs, spec.observation)
    return refs


def _assign_role_set_kind_ref(
    refs: dict[str, RelationRoleRef],
    ref: RelationRoleRef,
) -> None:
    set_kind = _expected_set_kind(ref.role)
    existing = refs.get(ref.relation_id)
    if existing is not None and _expected_set_kind(existing.role) != set_kind:
        raise RelationEngineError(f"relation {ref.relation_id} has conflicting roles")
    refs[ref.relation_id] = ref


def _with_role_set_kind(
    relation: RelationRows,
    ref: RelationRoleRef | None,
) -> RelationRows:
    set_kind = _expected_set_kind_for_ref(ref)
    if set_kind is None:
        return relation
    if ref is None:
        raise AssertionError("set kind requires relation role")
    existing = relation.completeness.set_kind
    if existing not in {RelationSetKind.UNKNOWN, set_kind}:
        raise RelationEngineError(
            f"{ref.role.value} relation {relation.id} requires {set_kind.value} set kind"
        )
    return RelationRows(
        id=relation.id,
        rows=relation.rows,
        grain_keys=relation.grain_keys,
        field_types=relation.field_types,
        field_answer_output_ids=relation.field_answer_output_ids,
        evidence=relation.evidence,
        completeness=replace(relation.completeness, set_kind=set_kind),
    )


def _projected_field_types(
    rows: tuple[dict[str, RuntimeValue], ...],
    *,
    inputs: tuple[RelationRows, ...],
) -> dict[str, str]:
    input_types: dict[str, str] = {}
    for relation in inputs:
        input_types.update(dict(relation.field_types or {}))
    return {
        field_id: input_types[field_id]
        for row in rows
        for field_id in row
        if field_id in input_types
    }


def _projected_field_answer_output_ids(
    rows: tuple[dict[str, RuntimeValue], ...],
    *,
    inputs: tuple[RelationRows, ...],
) -> dict[str, tuple[str, ...]]:
    input_output_ids: dict[str, tuple[str, ...]] = {}
    for relation in inputs:
        input_output_ids.update(dict(relation.field_answer_output_ids or {}))
    return {
        field_id: input_output_ids[field_id]
        for row in rows
        for field_id in row
        if field_id in input_output_ids
    }


def _expected_set_kind_for_ref(ref: RelationRoleRef | None) -> RelationSetKind | None:
    if ref is None:
        return None
    return _expected_set_kind(ref.role)


def _role_expand_grain(
    input_relation: RelationRows,
    spec: RoleExpandSpec,
) -> tuple[str, ...]:
    if not input_relation.grain_keys:
        return ()
    missing_carry = [
        field for field in input_relation.grain_keys if field not in spec.carry_fields
    ]
    if missing_carry:
        raise RelationEngineError(
            f"role_expand output grain requires carried field {missing_carry[0]}"
        )
    output_grain = (*input_relation.grain_keys, spec.role_field)
    _require_fields_declared(spec.output_fields, output_grain, "role_expand output")
    return output_grain


def _concat_grain(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    grain = (*left, *right)
    if len(set(grain)) != len(grain):
        raise RelationEngineError("duplicate grain field")
    return grain


def _join_grain(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    return _concat_grain(left, tuple(field for field in right if field not in left))


def _require_fields_declared(
    fields: tuple[str, ...],
    required: tuple[str, ...],
    label: str,
) -> None:
    for field in required:
        if field not in fields:
            raise RelationEngineError(f"{label} missing field {field}")


def _require_rows_have_fields(
    rows: list[dict[str, RuntimeValue]] | tuple[dict[str, RuntimeValue], ...],
    fields: tuple[str, ...],
    label: str,
) -> None:
    for row in rows:
        for field in fields:
            if field not in row:
                raise RelationEngineError(f"{label} missing field {field}")


def _raise_undefined_empty_aggregation(
    aggregations: tuple[AggregationSpec, ...],
) -> None:
    for aggregation in aggregations:
        reason = empty_aggregation_undefined_reason(aggregation.function)
        if reason is not None:
            raise UndefinedOperationError(
                reason_code=reason,
                input_refs=(aggregation.input_field,),
            )


def _aggregate_value(
    aggregation: AggregationSpec,
    rows: list[Row],
    field_types: dict[str, str],
) -> RuntimeValue:
    function = aggregation.function
    if function == AggregationFunction.COUNT:
        return len(rows)
    values = [_field(row, aggregation.input_field) for row in rows]
    if not values:
        reason = empty_aggregation_undefined_reason(function)
        if reason is not None:
            raise UndefinedOperationError(
                reason_code=reason,
                input_refs=(aggregation.input_field,),
            )
    if function in {AggregationFunction.SUM, AggregationFunction.AVG}:
        numeric = [
            declared_number(value, field_types.get(aggregation.input_field))
            for value in values
        ]
        total = sum(numeric, start=declared_number(0, "decimal"))
        if function == AggregationFunction.SUM:
            return total
        return total / len(numeric)
    if function == AggregationFunction.MIN:
        return (
            min(
                values,
                key=lambda value: declared_order_key(
                    value, field_types.get(aggregation.input_field)
                ),
            )
            if values
            else None
        )
    if function == AggregationFunction.MAX:
        return (
            max(
                values,
                key=lambda value: declared_order_key(
                    value, field_types.get(aggregation.input_field)
                ),
            )
            if values
            else None
        )
    raise RelationEngineError(f"unsupported aggregation {function}")


def _join_match(
    left: Row,
    right: Row,
    join_keys: Iterable[JoinKey],
    left_types: dict[str, str],
    right_types: dict[str, str],
) -> bool:
    return all(
        declared_equal(
            _field(left, key.left),
            left_types.get(key.left),
            _field(right, key.right),
            right_types.get(key.right),
        )
        for key in join_keys
    )


def _merge_rows(
    left: Row,
    right: Row,
    join_keys: Iterable[JoinKey],
    left_types: dict[str, str] | None = None,
    right_types: dict[str, str] | None = None,
) -> dict[str, RuntimeValue]:
    del join_keys
    output = dict(left)
    for field, value in right.items():
        if field in output:
            if declared_equal(
                output[field],
                (left_types or {}).get(field),
                value,
                (right_types or {}).get(field),
            ):
                continue
            raise RelationEngineError(f"field conflict {field}")
        output[field] = value
    return output


def _assign_field(
    row: dict[str, RuntimeValue],
    field: str,
    value: RuntimeValue,
) -> None:
    if field in row:
        raise RelationEngineError(f"field conflict {field}")
    row[field] = value


def _assign_or_match(
    row: dict[str, RuntimeValue],
    field: str,
    value: RuntimeValue,
) -> None:
    existing = row.get(field)
    if field in row and existing != value:
        raise RelationEngineError(f"field conflict {field}")
    row[field] = value


def _relation(relations: dict[str, RelationRows], relation_id: str) -> RelationRows:
    if relation_id not in relations:
        raise RelationEngineError(f"unknown relation {relation_id}")
    return relations[relation_id]


def _field(row: Row, field: str) -> RuntimeValue:
    if field not in row:
        raise RelationEngineError(f"missing field {field}")
    return row[field]


def _output_relation(operation: ExecutableOperation) -> str:
    if not operation.output_relation:
        raise RelationEngineError(f"operation {operation.id} requires output relation")
    return operation.output_relation


class _Descending:
    def __init__(self, value: tuple[int, Decimal | str]) -> None:
        self.value = value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _Descending):
            return NotImplemented
        left_rank, left = self.value
        right_rank, right = other.value
        if left_rank != right_rank:
            return left_rank > right_rank
        if isinstance(left, Decimal) and isinstance(right, Decimal):
            return left > right
        if isinstance(left, str) and isinstance(right, str):
            return left > right
        raise RelationEngineError("ordering keys have incompatible runtime kinds")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Descending) and self.value == other.value


__all__ = tuple(name for name in globals() if not name.startswith("__"))
