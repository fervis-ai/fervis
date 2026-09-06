"""Operation invariant checks."""

from __future__ import annotations

from fervis.lookup.answer_program.operations import JoinMode

from typing_extensions import assert_never

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AntiJoinSpec,
    ComputeSpec,
    CrossJoinSpec,
    FilterSpec,
    JoinSpec,
    NamedExpression,
    Operation,
    ProjectSpec,
    ProjectToKeySpec,
    KeepAll,
    OrderSpec,
    Take,
    AtPosition,
    RelationRole,
    RelationRoleRef,
    RoleExpandSpec,
    SortDirection,
    SortKey,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.expressions import (
    Expression,
    FieldRef,
    expression_references,
)


def verify_operation(operation: Operation) -> None:
    spec = operation.spec
    if not isinstance(spec, ComputeSpec) and not operation.output_relation:
        raise VerificationError(f"{operation.id} requires output relation")
    if isinstance(spec, FilterSpec):
        _require_input(spec.input_relation, "filter")
        _require_condition(spec.condition, "filter")
    elif isinstance(spec, ProjectSpec):
        _require_input(spec.input_relation, "project")
        if not spec.outputs:
            raise VerificationError("project requires fields")
        _require_unique_fields(
            tuple(output.output_field for output in spec.outputs),
            "project",
        )
    elif isinstance(spec, ProjectToKeySpec):
        _require_input(spec.input_relation, "project_to_key")
        if not spec.key_fields:
            raise VerificationError("project_to_key requires key fields")
        _require_unique_fields((*spec.key_fields, *spec.carry_fields), "project_to_key")
    elif isinstance(spec, JoinSpec):
        if not isinstance(spec.mode, JoinMode):
            raise VerificationError("join requires a declared join mode")
        _require_binary_join(spec.left, spec.right, spec.join_keys, "join")
    elif isinstance(spec, UnionSpec):
        if len(spec.inputs) < 2:
            raise VerificationError("union requires at least two inputs")
        _require_unique_fields(spec.output_fields, "union")
    elif isinstance(spec, RoleExpandSpec):
        _require_input(spec.input_relation, "role_expand")
        if not spec.mappings:
            raise VerificationError("role_expand requires mappings")
        if not spec.output_fields:
            raise VerificationError("role_expand requires output fields")
        _require_unique_fields(spec.output_fields, "role_expand")
    elif isinstance(spec, CrossJoinSpec):
        _require_input(spec.left, "cross_join")
        _require_input(spec.right, "cross_join")
    elif isinstance(spec, AntiJoinSpec):
        _require_relation_role(
            spec.candidate,
            RelationRole.ANTI_JOIN_CANDIDATE,
            "anti_join",
        )
        _require_relation_role(
            spec.observed,
            RelationRole.ANTI_JOIN_OBSERVED,
            "anti_join",
        )
        if spec.candidate.relation_id == spec.observed.relation_id:
            raise VerificationError("anti_join requires distinct relation roles")
        if not spec.join_keys:
            raise VerificationError("anti_join requires join keys")
        if not spec.output_fields:
            raise VerificationError("anti_join requires output fields")
        _require_unique_fields(
            tuple(output.output_field for output in spec.output_fields),
            "anti_join",
        )
        _require_direct_projections(spec.output_fields, "anti_join")
    elif isinstance(spec, UniversalConditionSpec):
        _require_universal_condition(spec)
        _require_direct_projections(spec.output_fields, "universal_condition")
    elif isinstance(spec, AggregateSpec):
        _require_input(spec.input_relation, "aggregate")
        if not spec.aggregations:
            raise VerificationError("aggregate requires aggregations")
        _require_aggregations(spec)
    elif isinstance(spec, OrderSpec):
        _require_order(spec)
    elif isinstance(spec, ComputeSpec):
        _require_compute(spec)
    else:
        assert_never(spec)


def _require_direct_projections(
    outputs: tuple[NamedExpression, ...],
    label: str,
) -> None:
    if any(not isinstance(output.expression, FieldRef) for output in outputs):
        raise VerificationError(f"{label} output requires direct field expression")


def _require_input(input_relation: str, label: str) -> None:
    if not input_relation:
        raise VerificationError(f"{label} requires input relation")


def _require_binary_join(
    left: str,
    right: str,
    join_keys: object,
    label: str,
) -> None:
    if not left or not right:
        raise VerificationError(f"{label} requires left and right inputs")
    if not join_keys:
        raise VerificationError(f"{label} requires join keys")


def _require_universal_condition(spec: UniversalConditionSpec) -> None:
    _require_relation_role(
        spec.candidate_subject,
        RelationRole.UNIVERSAL_CANDIDATE_SUBJECT,
        "universal_condition",
    )
    _require_relation_role(
        spec.required_dimension,
        RelationRole.UNIVERSAL_REQUIRED_DIMENSION,
        "universal_condition",
    )
    _require_relation_role(
        spec.observation,
        RelationRole.UNIVERSAL_OBSERVATION,
        "universal_condition",
    )
    if not spec.subject_keys:
        raise VerificationError("universal_condition requires subject keys")
    if not spec.dimension_keys:
        raise VerificationError("universal_condition requires dimension keys")
    if not spec.output_fields:
        raise VerificationError("universal_condition requires output fields")
    _require_condition(spec.condition, "universal_condition")
    _require_unique_fields(
        tuple(output.output_field for output in spec.output_fields),
        "universal_condition",
    )


def _require_relation_role(
    ref: RelationRoleRef,
    expected_role: RelationRole,
    label: str,
) -> None:
    if not ref.relation_id:
        raise VerificationError(f"{label} requires relation id")
    if ref.role != expected_role:
        raise VerificationError(f"{label} requires {expected_role.value}")
    if not ref.required_identity_fields:
        raise VerificationError(f"{label} requires grain obligation")


def _require_order(spec: OrderSpec) -> None:
    _require_input(spec.input_relation, "order")
    if not spec.order_by:
        raise VerificationError("order requires ordering keys")
    _require_sort_keys(spec.order_by, "order")
    if not isinstance(spec.selection, (KeepAll, Take, AtPosition)):
        raise VerificationError("order requires a selection")
    if (
        isinstance(spec.selection, (Take, AtPosition))
        and not expression_references((spec.selection.limit if isinstance(spec.selection, Take) else spec.selection.position)).leaves
    ):
        raise VerificationError("order take limit requires an expression")


def _require_compute(spec: ComputeSpec) -> None:
    if not expression_references(spec.expression).leaves:
        raise VerificationError("compute requires scalar inputs")
    if not spec.output_scalar:
        raise VerificationError("compute requires output scalar")


def _require_condition(condition: Expression, label: str) -> None:
    if not expression_references(condition).leaves:
        raise VerificationError(f"{label} requires condition")


def _require_aggregations(spec: AggregateSpec) -> None:
    output_fields = [*spec.group_by]
    for aggregation in spec.aggregations:
        if aggregation.function not in set(AggregationFunction):
            raise VerificationError("aggregate requires supported function")
        if not aggregation.output_field:
            raise VerificationError("aggregate requires output field")
        if (
            aggregation.function != AggregationFunction.COUNT
            and not aggregation.input_field
        ):
            raise VerificationError("aggregate requires input field")
        if not isinstance(aggregation.distinct_argument, bool):
            raise VerificationError("aggregate distinct flag must be boolean")
        if aggregation.distinct_argument and not aggregation.input_field and not aggregation.grain_fields:
            raise VerificationError(
                "distinct row count requires an explicit argument field"
            )
        _require_unique_fields(aggregation.grain_fields, "aggregate observation grain")
        if aggregation.filter is not None:
            _require_condition(aggregation.filter, "aggregate filter")
        output_fields.append(aggregation.output_field)
    _require_unique_fields(tuple(output_fields), "aggregate")


def _require_sort_keys(sort_keys: tuple[SortKey, ...], label: str) -> None:
    for sort_key in sort_keys:
        if not sort_key.field:
            raise VerificationError(f"{label} requires sort field")
        if sort_key.direction not in set(SortDirection):
            raise VerificationError(f"{label} requires supported sort direction")


def _require_unique_fields(fields: tuple[str, ...], label: str) -> None:
    seen: set[str] = set()
    for field in fields:
        if field in seen:
            raise VerificationError(f"{label} has duplicate output field")
        seen.add(field)
