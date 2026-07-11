"""Relation-producing operation implementations."""

from __future__ import annotations

from collections import OrderedDict

from fervis.lookup.plan_execution.operation_runtime import RelationEngineError
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.answer_program.operations import (
    AntiJoinSpec,
    CrossJoinSpec,
    FilterSpec,
    JoinSpec,
    Operation,
    ProjectSpec,
    ProjectToIdentitySpec,
    RelationRole,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)

from .predicates import _predicate, _predicate_fact
from .shared import (
    _assign_field,
    _concat_grain,
    _field,
    _input_scalar_proof_refs,
    _identity_type_for_grain,
    _join_grain,
    _join_match,
    _merge_rows,
    _operation_relation,
    _project_grain,
    _project_output,
    _relation,
    _require_grain,
    _role_expand_grain,
    _role_relation,
)


def _filter(
    operation: Operation,
    spec: FilterSpec,
    relations: dict[str, RelationRows],
    scalars: dict[str, object],
    scalar_proofs: dict[str, tuple[str, ...]],
    *,
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    rows = [
        dict(row)
        for row in _relation(relations, spec.input_relation).rows
        if _predicate(row, spec.predicate, scalars)
    ]
    input_relation = _relation(relations, spec.input_relation)
    return _operation_relation(
        operation,
        rows,
        grain_keys=input_relation.grain_keys,
        inputs=(input_relation,),
        scalar_refs=(
            *_input_scalar_proof_refs(operation, scalar_proofs),
            *operation_refs,
        ),
        identity_type=input_relation.identity_type,
    )


def _project(
    operation: Operation,
    spec: ProjectSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    output_rows = []
    for row in _relation(relations, spec.input_relation).rows:
        output: dict[str, object] = {}
        for field in spec.fields:
            output[field.output or field.source] = _field(row, field.source)
        output_rows.append(output)
    input_relation = _relation(relations, spec.input_relation)
    return _operation_relation(
        operation,
        output_rows,
        grain_keys=_project_grain(input_relation, spec.fields),
        inputs=(input_relation,),
        identity_type=_identity_type_for_grain(
            _project_grain(input_relation, spec.fields),
            (input_relation,),
        ),
    )


def _project_to_identity(
    operation: Operation,
    spec: ProjectToIdentitySpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    output_by_key: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
    for row in input_relation.rows:
        key = tuple(_field(row, field) for field in spec.identity_fields)
        projected = _project_output(row, spec.fields, grain_keys=spec.identity_fields)
        existing = output_by_key.get(key)
        if existing is None:
            output_by_key[key] = projected
        elif existing != projected:
            raise RelationEngineError("conflicting project_to_identity row")
    return _operation_relation(
        operation,
        tuple(output_by_key.values()),
        grain_keys=spec.identity_fields,
        inputs=(input_relation,),
        identity_type=_identity_type_for_grain(
            spec.identity_fields,
            (input_relation,),
        ),
    )


def _join(
    operation: Operation,
    spec: JoinSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    left_rows = _relation(relations, spec.left).rows
    right_rows = _relation(relations, spec.right).rows
    output = []
    for left in left_rows:
        for right in right_rows:
            if _join_match(left, right, spec.join_keys):
                output.append(_merge_rows(left, right, spec.join_keys))
    return _operation_relation(
        operation,
        output,
        grain_keys=_join_grain(
            _relation(relations, spec.left).grain_keys,
            _relation(relations, spec.right).grain_keys,
        ),
        inputs=(_relation(relations, spec.left), _relation(relations, spec.right)),
    )


def _union(
    operation: Operation,
    spec: UnionSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    output_by_key: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
    output: list[dict[str, object]] = []
    input_relations: list[RelationRows] = []
    for relation_id in spec.inputs:
        relation = _relation(relations, relation_id)
        input_relations.append(relation)
        for row in relation.rows:
            projected = {field: _field(row, field) for field in spec.output_fields}
            if not spec.identity_fields:
                output.append(projected)
                continue
            key = tuple(_field(projected, field) for field in spec.identity_fields)
            existing = output_by_key.get(key)
            if existing is None:
                output_by_key[key] = projected
            elif existing != projected:
                raise RelationEngineError("conflicting union row")
    rows = tuple(output_by_key.values()) if spec.identity_fields else tuple(output)
    return _operation_relation(
        operation,
        rows,
        grain_keys=spec.identity_fields,
        inputs=tuple(input_relations),
        identity_type=_identity_type_for_grain(
            spec.identity_fields,
            tuple(input_relations),
        ),
    )


def _role_expand(
    operation: Operation,
    spec: RoleExpandSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    output = []
    input_relation = _relation(relations, spec.input_relation)
    output_grain = _role_expand_grain(input_relation, spec)
    for row in input_relation.rows:
        carry = {field: _field(row, field) for field in spec.carry_fields}
        for mapping in spec.mappings:
            expanded = dict(carry)
            _assign_field(expanded, spec.role_field, mapping.role)
            _assign_field(
                expanded,
                mapping.output_field,
                _field(row, mapping.source_field),
            )
            output.append(
                {field: _field(expanded, field) for field in spec.output_fields}
            )
    return _operation_relation(
        operation,
        output,
        grain_keys=output_grain,
        inputs=(input_relation,),
    )


def _cross_join(
    operation: Operation,
    spec: CrossJoinSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    output = []
    left_relation = _relation(relations, spec.left)
    right_relation = _relation(relations, spec.right)
    for left in left_relation.rows:
        for right in right_relation.rows:
            output.append(_merge_rows(left, right, ()))
    return _operation_relation(
        operation,
        output,
        grain_keys=_concat_grain(left_relation.grain_keys, right_relation.grain_keys),
        inputs=(left_relation, right_relation),
    )


def _anti_join(
    operation: Operation,
    spec: AntiJoinSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    candidate = _role_relation(
        relations,
        spec.candidate,
        RelationRole.ANTI_JOIN_CANDIDATE,
    )
    observed = _role_relation(
        relations,
        spec.observed,
        RelationRole.ANTI_JOIN_OBSERVED,
    )
    _require_grain(candidate, tuple(key.left for key in spec.join_keys), "candidate")
    _require_grain(observed, tuple(key.right for key in spec.join_keys), "observed")
    observed_keys = {
        tuple(_field(row, key.right) for key in spec.join_keys) for row in observed.rows
    }
    candidates: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
    for row in candidate.rows:
        key = tuple(_field(row, item.left) for item in spec.join_keys)
        projected = _project_output(
            row,
            spec.output_fields,
            grain_keys=spec.candidate.required_identity_fields,
        )
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = projected
        elif existing != projected:
            raise RelationEngineError("ambiguous candidate output")

    output = [
        projected for key, projected in candidates.items() if key not in observed_keys
    ]
    return _operation_relation(
        operation,
        output,
        grain_keys=spec.candidate.required_identity_fields,
        inputs=(candidate, observed),
    )


def _universal_condition(
    operation: Operation,
    spec: UniversalConditionSpec,
    relations: dict[str, RelationRows],
    scalars: dict[str, object],
    scalar_proofs: dict[str, tuple[str, ...]],
    *,
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    candidates = _role_relation(
        relations,
        spec.candidate_subject,
        RelationRole.UNIVERSAL_CANDIDATE_SUBJECT,
    )
    dimensions = _role_relation(
        relations,
        spec.required_dimension,
        RelationRole.UNIVERSAL_REQUIRED_DIMENSION,
    )
    observations = _role_relation(
        relations,
        spec.observation,
        RelationRole.UNIVERSAL_OBSERVATION,
    )
    candidate_key_fields = tuple(key.left for key in spec.subject_keys)
    observation_subject_fields = tuple(key.right for key in spec.subject_keys)
    dimension_key_fields = tuple(key.left for key in spec.dimension_keys)
    observation_dimension_fields = tuple(key.right for key in spec.dimension_keys)
    _require_grain(candidates, candidate_key_fields, "candidate subject")
    _require_grain(dimensions, dimension_key_fields, "required dimension")
    _require_grain(
        observations,
        (*observation_subject_fields, *observation_dimension_fields),
        "observation",
    )

    observation_index: dict[tuple[object, ...], tuple[object, ...]] = {}
    for row in observations.rows:
        key = (
            *tuple(_field(row, field) for field in observation_subject_fields),
            *tuple(_field(row, field) for field in observation_dimension_fields),
        )
        predicate_fact = _predicate_fact(row, spec.predicate, scalars)
        existing = observation_index.get(key)
        if existing is None:
            observation_index[key] = predicate_fact
        elif existing != predicate_fact:
            raise RelationEngineError("conflicting universal observation")

    output_by_key: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
    for candidate in candidates.rows:
        candidate_key = tuple(
            _field(candidate, field) for field in candidate_key_fields
        )
        passed = True
        for dimension in dimensions.rows:
            dimension_key = tuple(
                _field(dimension, field) for field in dimension_key_fields
            )
            observation_key = (*candidate_key, *dimension_key)
            if observation_key not in observation_index:
                raise RelationEngineError("missing observation for universal condition")
            if not observation_index[observation_key][0]:
                passed = False
        if passed:
            projected = _project_output(
                candidate,
                spec.output_fields,
                grain_keys=spec.candidate_subject.required_identity_fields,
            )
            existing = output_by_key.get(candidate_key)
            if existing is None:
                output_by_key[candidate_key] = projected
            elif existing != projected:
                raise RelationEngineError("ambiguous candidate subject output")
    return _operation_relation(
        operation,
        tuple(output_by_key.values()),
        grain_keys=spec.candidate_subject.required_identity_fields,
        inputs=(candidates, dimensions, observations),
        scalar_refs=(
            *_input_scalar_proof_refs(operation, scalar_proofs),
            *operation_refs,
        ),
    )
