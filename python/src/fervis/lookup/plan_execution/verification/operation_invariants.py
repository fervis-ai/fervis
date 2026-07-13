"""Operation invariant checks."""

from __future__ import annotations

from typing_extensions import assert_never

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.values import (
    ConstantRef,
    EnvironmentRef,
    NodeOutputRef,
    ParameterRef,
)
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AntiJoinSpec,
    ComputeSpec,
    compute_expression_leaves,
    CrossJoinSpec,
    FilterSpec,
    JoinSpec,
    Operation,
    Predicate,
    PredicateOperator,
    ProjectSpec,
    ProjectToKeySpec,
    RankSpec,
    RelationRole,
    RelationRoleRef,
    RoleExpandSpec,
    SortDirection,
    SortKey,
    TiePolicy,
    UnionSpec,
    UniversalConditionSpec,
)


BINARY_PREDICATE_OPERATORS = frozenset(
    {
        PredicateOperator.EQUALS,
        PredicateOperator.NOT_EQUALS,
        PredicateOperator.LT,
        PredicateOperator.LTE,
        PredicateOperator.GT,
        PredicateOperator.GTE,
        PredicateOperator.CONTAINS,
    }
)
UNARY_PREDICATE_OPERATORS = frozenset(
    {
        PredicateOperator.IS_NULL,
        PredicateOperator.NOT_NULL,
    }
)


def verify_operation(operation: Operation) -> None:
    spec = operation.spec
    if not isinstance(spec, ComputeSpec) and not operation.output_relation:
        raise VerificationError(f"{operation.id} requires output relation")
    if isinstance(spec, FilterSpec):
        _require_input(spec.input_relation, "filter")
        _require_predicate(spec.predicate, "filter")
    elif isinstance(spec, ProjectSpec):
        _require_input(spec.input_relation, "project")
        if not spec.fields:
            raise VerificationError("project requires fields")
        _require_unique_fields(
            tuple(field.output or field.source for field in spec.fields),
            "project",
        )
    elif isinstance(spec, ProjectToKeySpec):
        _require_input(spec.input_relation, "project_to_key")
        if not spec.key_fields:
            raise VerificationError("project_to_key requires key fields")
        _require_unique_fields(spec.key_fields, "project_to_key")
    elif isinstance(spec, JoinSpec):
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
            tuple(field.output or field.source for field in spec.output_fields),
            "anti_join",
        )
    elif isinstance(spec, UniversalConditionSpec):
        _require_universal_condition(spec)
    elif isinstance(spec, AggregateSpec):
        _require_input(spec.input_relation, "aggregate")
        if not spec.aggregations:
            raise VerificationError("aggregate requires aggregations")
        _require_aggregations(spec)
    elif isinstance(spec, RankSpec):
        _require_rank(spec)
    elif isinstance(spec, ComputeSpec):
        _require_compute(spec)
    else:
        assert_never(spec)


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
    _require_predicate(spec.predicate, "universal_condition")
    _require_unique_fields(
        tuple(field.output or field.source for field in spec.output_fields),
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


def _require_rank(spec: RankSpec) -> None:
    _require_input(spec.input_relation, "rank")
    if not spec.order_by:
        raise VerificationError("rank requires ordering and deterministic tie policy")
    if not spec.tie_policy:
        raise VerificationError("rank requires deterministic tie policy")
    _require_sort_keys(spec.order_by, "rank")
    if spec.tie_policy not in set(TiePolicy):
        raise VerificationError("rank requires deterministic tie policy")
    if not isinstance(
        spec.limit,
        (ParameterRef, NodeOutputRef, ConstantRef, EnvironmentRef),
    ):
        raise VerificationError("rank limit has unclassified value origin")
    if not spec.tie_breakers:
        raise VerificationError("field tie policy requires tie breakers")
    _require_sort_keys(spec.tie_breakers, "rank")


def _require_compute(spec: ComputeSpec) -> None:
    if not compute_expression_leaves(spec.expression):
        raise VerificationError("compute requires scalar inputs")
    if not spec.output_scalar:
        raise VerificationError("compute requires output scalar")


def _require_predicate(predicate: Predicate, label: str) -> None:
    if not predicate.left or not predicate.operator:
        raise VerificationError(f"{label} requires predicate")
    if predicate.operator not in set(PredicateOperator):
        raise VerificationError(f"{label} requires supported predicate operator")
    has_field_rhs = bool(predicate.right)
    has_scalar_rhs = bool(predicate.right_scalar)
    if predicate.operator in BINARY_PREDICATE_OPERATORS:
        if has_field_rhs == has_scalar_rhs:
            raise VerificationError(f"{label} requires exactly one right-hand side")
        return
    if predicate.operator in UNARY_PREDICATE_OPERATORS and (
        has_field_rhs or has_scalar_rhs
    ):
        raise VerificationError(f"{label} does not accept a right-hand side")


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
