"""Relation-producing operation implementations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineError,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.plan_execution.declared_values import (
    declared_equal,
    declared_comparison_key,
    declared_comparison_types_compatible,
    declared_key,
    declared_types_compatible,
)
from fervis.lookup.answer_program.operations import (
    AntiJoinSpec,
    CrossJoinSpec,
    FilterSpec,
    JoinSpec,
    JoinMode,
    NamedExpression,
    ProjectSpec,
    ProjectToKeySpec,
    RelationRole,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.plan_execution.expression_schema import expression_value_type
from .expression_evaluator import (
    ExpressionEnvironment,
    evaluate_condition,
    evaluate_expression,
    evaluated_condition_fact,
)
from .shared import (
    _assign_field,
    _concat_grain,
    _field,
    _input_scalar_proof_refs,
    _join_grain,
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
    operation: ExecutableOperation,
    spec: FilterSpec,
    relations: dict[str, RelationRows],
    scalars: dict[str, RuntimeValue],
    scalar_proofs: dict[str, tuple[str, ...]],
    scalar_types: dict[str, str],
    environment_values: dict[str, RuntimeValue],
    environment_types: dict[str, str],
    *,
    node_outputs: dict[str, dict[str, RuntimeValue]],
    node_output_types: dict[str, dict[str, str]],
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    rows = [
        dict(row)
        for row in input_relation.rows
        if evaluate_condition(
            spec.condition,
            environment=ExpressionEnvironment(
                node_output_types=node_output_types,
                node_outputs=node_outputs,
                row=row,
                field_types=input_relation.field_types or {},
                scalars=scalars,
                scalar_types=scalar_types,
                environment_values=environment_values,
                environment_types=environment_types,
            ),
        )
    ]
    return _operation_relation(
        operation,
        rows,
        grain_keys=input_relation.grain_keys,
        inputs=(input_relation,),
        field_types=dict(input_relation.field_types or {}),
        scalar_refs=(
            *_input_scalar_proof_refs(operation, scalar_proofs),
            *operation_refs,
        ),
    )


def _project(
    operation: ExecutableOperation,
    spec: ProjectSpec,
    relations: dict[str, RelationRows],
    scalars: dict[str, RuntimeValue],
    scalar_proofs: dict[str, tuple[str, ...]],
    scalar_types: dict[str, str],
    node_outputs: dict[str, dict[str, RuntimeValue]],
    *,
    node_output_types: dict[str, dict[str, str]],
    environment_values: dict[str, RuntimeValue],
    environment_types: dict[str, str],
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    output_rows = []
    output_types = {
        named.output_field: expression_value_type(
            named.expression,
            field_types=input_relation.field_types,
            scalar_types=scalar_types,
            environment_types=environment_types,
            node_output_types=node_output_types,
        )
        for named in spec.outputs
    }
    for row_number, row in enumerate(input_relation.rows, start=1):
        output: dict[str, RuntimeValue] = {}
        for named in spec.outputs:
            evaluated = evaluate_expression(
                named.expression,
                environment=ExpressionEnvironment(
                    node_output_types=node_output_types,
                    row=row,
                    row_number=row_number,
                    field_types=input_relation.field_types,
                    scalars=scalars,
                    scalar_types=scalar_types,
                    node_outputs=node_outputs,
                    environment_values=environment_values,
                    environment_types=environment_types,
                ),
            )
            _assign_field(output, named.output_field, evaluated.value)
            existing = output_types.get(named.output_field)
            if existing is not None and not declared_types_compatible(
                existing, evaluated.value_type
            ):
                raise RelationEngineError("projection output types are inconsistent")
        output_rows.append(output)
    return _operation_relation(
        operation,
        output_rows,
        grain_keys=_project_grain(input_relation, spec.outputs),
        inputs=(input_relation,),
        field_types=output_types,
        scalar_refs=_input_scalar_proof_refs(operation, scalar_proofs),
    )


def _project_to_key(
    operation: ExecutableOperation,
    spec: ProjectToKeySpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    output_by_key: OrderedDict[tuple[RuntimeValue, ...], dict[str, RuntimeValue]] = (
        OrderedDict()
    )
    selected_fields = (*spec.key_fields, *spec.carry_fields)
    types = dict(input_relation.field_types or {})
    for row in input_relation.rows:
        key = tuple(
            declared_key(
                _field(row, field), (input_relation.field_types or {}).get(field)
            )
            for field in spec.key_fields
        )
        projected = {field: _field(row, field) for field in selected_fields}
        previous = output_by_key.get(key)
        if previous is not None and any(
            declared_key(previous[field], types.get(field))
            != declared_key(projected[field], types.get(field))
            for field in selected_fields
        ):
            raise RelationEngineError(
                "key projection has conflicting values for one identity"
            )
        output_by_key[key] = projected
    return _operation_relation(
        operation,
        tuple(output_by_key.values()),
        grain_keys=spec.key_fields,
        inputs=(input_relation,),
        field_types={
            field: (input_relation.field_types or {}).get(field, "")
            for field in selected_fields
        },
    )


def _join(
    operation: ExecutableOperation,
    spec: JoinSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    left_relation = _relation(relations, spec.left)
    right_relation = _relation(relations, spec.right)
    left_types = dict(left_relation.field_types or {})
    right_types = dict(right_relation.field_types or {})
    for key in spec.join_keys:
        if not declared_comparison_types_compatible(
            left_types.get(key.left), right_types.get(key.right)
        ):
            raise RelationEngineError("join keys have incompatible declared types")
    left_rows = left_relation.rows
    right_rows = right_relation.rows
    output = []
    right_fields = set(right_types).union(*(row.keys() for row in right_rows))
    def match_key(row, *, left_side):
        values = tuple(
            _field(row, key.left if left_side else key.right)
            for key in spec.join_keys
        )
        if any(value is None for value in values):
            return None
        return tuple(
            declared_comparison_key(value, left_types.get(key.left), right_types.get(key.right))
            for key, value in zip(spec.join_keys, values)
        )

    right_index: dict[tuple[tuple[str, str], ...], list[Mapping[str, RuntimeValue]]] = {}
    for right in right_rows:
        key = match_key(right, left_side=False)
        if key is not None:
            right_index.setdefault(key, []).append(right)
    for left in left_rows:
        key = match_key(left, left_side=True)
        matches = right_index.get(key, ()) if key is not None else ()
        for right in matches:
            output.append(_merge_rows(left, right, spec.join_keys, left_types, right_types))
        if not matches and spec.mode is JoinMode.LEFT:
            output.append({**{field: None for field in right_fields}, **left})
    return _operation_relation(
        operation,
        output,
        grain_keys=_join_grain(
            left_relation.grain_keys,
            right_relation.grain_keys,
        ),
        inputs=(left_relation, right_relation),
        field_types=_merged_field_types(left_types, right_types),
    )


def _union(
    operation: ExecutableOperation,
    spec: UnionSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    output_by_key: OrderedDict[tuple[RuntimeValue, ...], dict[str, RuntimeValue]] = (
        OrderedDict()
    )
    output: list[dict[str, RuntimeValue]] = []
    input_relations: list[RelationRows] = []
    output_types: dict[str, str] = {}
    for relation_id in spec.inputs:
        relation = _relation(relations, relation_id)
        input_relations.append(relation)
        for field in spec.output_fields:
            candidate_type = (relation.field_types or {}).get(field, "")
            if field in output_types and not declared_types_compatible(
                output_types[field], candidate_type
            ):
                raise RelationEngineError(
                    "union fields have incompatible declared types"
                )
            output_types.setdefault(field, candidate_type)
        for row in relation.rows:
            projected = {field: _field(row, field) for field in spec.output_fields}
            if not spec.identity_fields:
                output.append(projected)
                continue
            key = tuple(
                declared_key(_field(projected, field), output_types.get(field))
                for field in spec.identity_fields
            )
            existing = output_by_key.get(key)
            if existing is None:
                output_by_key[key] = projected
            elif not _rows_equal(existing, output_types, projected, output_types):
                raise RelationEngineError("conflicting union row")
    rows = tuple(output_by_key.values()) if spec.identity_fields else tuple(output)
    return _operation_relation(
        operation,
        rows,
        grain_keys=spec.identity_fields,
        inputs=tuple(input_relations),
        field_types=output_types,
    )


def _role_expand(
    operation: ExecutableOperation,
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
        field_types=_role_expand_field_types(input_relation, spec),
    )


def _cross_join(
    operation: ExecutableOperation,
    spec: CrossJoinSpec,
    relations: dict[str, RelationRows],
) -> RelationRows:
    output = []
    left_relation = _relation(relations, spec.left)
    right_relation = _relation(relations, spec.right)
    for left in left_relation.rows:
        for right in right_relation.rows:
            output.append(
                _merge_rows(
                    left,
                    right,
                    (),
                    dict(left_relation.field_types or {}),
                    dict(right_relation.field_types or {}),
                )
            )
    return _operation_relation(
        operation,
        output,
        grain_keys=_concat_grain(left_relation.grain_keys, right_relation.grain_keys),
        inputs=(left_relation, right_relation),
        field_types=_merged_field_types(
            dict(left_relation.field_types or {}),
            dict(right_relation.field_types or {}),
        ),
    )


def _anti_join(
    operation: ExecutableOperation,
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
    candidate_types = dict(candidate.field_types or {})
    observed_types = dict(observed.field_types or {})
    for join_key in spec.join_keys:
        if not declared_comparison_types_compatible(
            candidate_types.get(join_key.left),
            observed_types.get(join_key.right),
        ):
            raise RelationEngineError("anti-join keys have incompatible declared types")
    observed_keys = {
        tuple(
            declared_comparison_key(_field(row, key.right), candidate_types.get(key.left), observed_types.get(key.right))
            for key in spec.join_keys
        )
        for row in observed.rows
    }
    candidates: OrderedDict[tuple[RuntimeValue, ...], dict[str, RuntimeValue]] = (
        OrderedDict()
    )
    for row in candidate.rows:
        candidate_key = tuple(
            declared_comparison_key(_field(row, item.left), candidate_types.get(item.left), observed_types.get(item.right))
            for item in spec.join_keys
        )
        projected = _project_output(
            row,
            spec.output_fields,
            grain_keys=spec.candidate.required_identity_fields,
        )
        existing = candidates.get(candidate_key)
        if existing is None:
            candidates[candidate_key] = projected
        elif not _rows_equal(existing, candidate_types, projected, candidate_types):
            raise RelationEngineError("ambiguous candidate output")

    output = [
        projected
        for candidate_key, projected in candidates.items()
        if candidate_key not in observed_keys
    ]
    return _operation_relation(
        operation,
        output,
        grain_keys=spec.candidate.required_identity_fields,
        inputs=(candidate, observed),
        field_types=_project_field_types(
            candidate_types, spec.output_fields, spec.candidate.required_identity_fields
        ),
    )


def _universal_condition(
    operation: ExecutableOperation,
    spec: UniversalConditionSpec,
    relations: dict[str, RelationRows],
    scalars: dict[str, RuntimeValue],
    scalar_proofs: dict[str, tuple[str, ...]],
    scalar_types: dict[str, str],
    environment_values: dict[str, RuntimeValue],
    environment_types: dict[str, str],
    *,
    node_outputs: dict[str, dict[str, RuntimeValue]],
    node_output_types: dict[str, dict[str, str]],
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
    candidate_types = dict(candidates.field_types or {})
    dimension_types = dict(dimensions.field_types or {})
    observation_types = dict(observations.field_types or {})
    for subject_key in spec.subject_keys:
        if not declared_comparison_types_compatible(
            candidate_types.get(subject_key.left),
            observation_types.get(subject_key.right),
        ):
            raise RelationEngineError(
                "universal subject keys have incompatible declared types"
            )
    for dimension_join_key in spec.dimension_keys:
        if not declared_comparison_types_compatible(
            dimension_types.get(dimension_join_key.left),
            observation_types.get(dimension_join_key.right),
        ):
            raise RelationEngineError(
                "universal dimension keys have incompatible declared types"
            )

    observation_index: dict[tuple[RuntimeValue, ...], tuple[RuntimeValue, ...]] = {}
    for row in observations.rows:
        observation_key = tuple(
            declared_comparison_key(_field(row, key.right), source_types.get(key.left), observation_types.get(key.right))
            for keys, source_types in ((spec.subject_keys, candidate_types), (spec.dimension_keys, dimension_types))
            for key in keys
        )
        condition_fact = evaluated_condition_fact(
            spec.condition,
            environment=ExpressionEnvironment(
                node_output_types=node_output_types,
                node_outputs=node_outputs,
                row=row,
                field_types=observations.field_types or {},
                scalars=scalars,
                scalar_types=scalar_types,
                environment_values=environment_values,
                environment_types=environment_types,
            ),
        )
        existing = observation_index.get(observation_key)
        if existing is None:
            observation_index[observation_key] = condition_fact
        elif existing != condition_fact:
            raise RelationEngineError("conflicting universal observation")

    output_by_key: OrderedDict[tuple[RuntimeValue, ...], dict[str, RuntimeValue]] = (
        OrderedDict()
    )
    for candidate in candidates.rows:
        candidate_key = tuple(
            declared_comparison_key(_field(candidate, key.left), candidate_types.get(key.left), observation_types.get(key.right))
            for key in spec.subject_keys
        )
        passed = True
        for dimension in dimensions.rows:
            dimension_key = tuple(
                declared_comparison_key(_field(dimension, key.left), dimension_types.get(key.left), observation_types.get(key.right))
                for key in spec.dimension_keys
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
            existing_output = output_by_key.get(candidate_key)
            if existing_output is None:
                output_by_key[candidate_key] = projected
            elif not _rows_equal(
                existing_output, candidate_types, projected, candidate_types
            ):
                raise RelationEngineError("ambiguous candidate subject output")
    return _operation_relation(
        operation,
        tuple(output_by_key.values()),
        grain_keys=spec.candidate_subject.required_identity_fields,
        inputs=(candidates, dimensions, observations),
        field_types=_project_field_types(
            dict(candidates.field_types or {}),
            spec.output_fields,
            spec.candidate_subject.required_identity_fields,
        ),
        scalar_refs=(
            *_input_scalar_proof_refs(operation, scalar_proofs),
            *operation_refs,
        ),
    )


def _merged_field_types(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    output = dict(left)
    for field, field_type in right.items():
        if field in output and not declared_types_compatible(output[field], field_type):
            raise RelationEngineError(f"field {field} has incompatible declared types")
        output.setdefault(field, field_type)
    return output


def _rows_equal(
    left: dict[str, RuntimeValue],
    left_types: dict[str, str],
    right: dict[str, RuntimeValue],
    right_types: dict[str, str],
) -> bool:
    return left.keys() == right.keys() and all(
        declared_equal(
            left[field], left_types.get(field), right[field], right_types.get(field)
        )
        for field in left
    )


def _project_field_types(
    input_types: dict[str, str],
    fields: tuple[NamedExpression, ...],
    grain_fields: tuple[str, ...] = (),
) -> dict[str, str]:
    output = {field: input_types.get(field, "") for field in grain_fields}
    for projection in fields:
        if not isinstance(projection.expression, FieldRef):
            raise RelationEngineError(
                "role projection requires direct field expression"
            )
        source = projection.expression.field_id
        output[projection.output_field] = input_types.get(source, "")
    return output


def _role_expand_field_types(
    input_relation: RelationRows, spec: RoleExpandSpec
) -> dict[str, str]:
    input_types = dict(input_relation.field_types or {})
    output = {field: input_types.get(field, "") for field in spec.carry_fields}
    output[spec.role_field] = "string"
    for mapping in spec.mappings:
        output[mapping.output_field] = input_types.get(mapping.source_field, "")
    return {field: output.get(field, "") for field in spec.output_fields}
