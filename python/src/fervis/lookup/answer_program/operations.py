"""Generic relational operation model."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.types.enums import StrEnum
from typing import TypeAlias
from typing_extensions import assert_never

from fervis.lookup.answer_program.expressions import Expression, ExpressionReferences, expression_input_id, expression_references
from fervis.lookup.answer_program.values import ConstantRef, ParameterRef, NodeOutputRef
from fervis.lookup.answer_program.result_projection import EntityKeyProjection


class OperationKind(StrEnum):
    SQL_QUERY = "sql_query"
    FILTER = "filter"
    PROJECT = "project"
    PROJECT_TO_KEY = "project_to_key"
    JOIN = "join"
    UNION = "union"
    ROLE_EXPAND = "role_expand"
    CROSS_JOIN = "cross_join"
    ANTI_JOIN = "anti_join"
    UNIVERSAL_CONDITION = "universal_condition"
    AGGREGATE = "aggregate"
    ORDER = "order"
    COMPUTE = "compute"


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class AggregationFunction(StrEnum):
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    BOOL_ANY = "bool_any"
    BOOL_ALL = "bool_all"


class RelationRole(StrEnum):
    ANTI_JOIN_CANDIDATE = "anti_join.candidate"
    ANTI_JOIN_OBSERVED = "anti_join.observed"
    UNIVERSAL_CANDIDATE_SUBJECT = "universal_condition.candidate_subject"
    UNIVERSAL_REQUIRED_DIMENSION = "universal_condition.required_dimension"
    UNIVERSAL_OBSERVATION = "universal_condition.observation"


@dataclass(frozen=True)
class JoinKey:
    left: str
    right: str


@dataclass(frozen=True)
class RelationRoleRef:
    relation_id: str
    role: RelationRole
    required_identity_fields: tuple[str, ...]


@dataclass(frozen=True)
class SortKey:
    field: str
    direction: SortDirection


@dataclass(frozen=True)
class NamedExpression:
    output_field: str
    expression: Expression


@dataclass(frozen=True)
class RoleMapping:
    role: str
    source_field: str
    output_field: str


@dataclass(frozen=True)
class AggregationSpec:
    function: AggregationFunction
    output_field: str
    input_field: str = ""
    filter: Expression | None = None
    distinct_argument: bool = False
    grain_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilterSpec:
    input_relation: str
    condition: Expression
    proof_refs: tuple[str, ...] = ()
    kind: OperationKind = field(default=OperationKind.FILTER, init=False)


@dataclass(frozen=True)
class ProjectSpec:
    input_relation: str
    outputs: tuple[NamedExpression, ...]
    kind: OperationKind = field(default=OperationKind.PROJECT, init=False)


@dataclass(frozen=True)
class ProjectToKeySpec:
    input_relation: str
    key_fields: tuple[str, ...]
    carry_fields: tuple[str, ...] = ()
    kind: OperationKind = field(default=OperationKind.PROJECT_TO_KEY, init=False)


class JoinMode(StrEnum):
    INNER = "inner"
    LEFT = "left"


@dataclass(frozen=True)
class JoinSpec:
    left: str
    right: str
    join_keys: tuple[JoinKey, ...]
    mode: JoinMode = JoinMode.INNER
    kind: OperationKind = field(default=OperationKind.JOIN, init=False)


@dataclass(frozen=True)
class UnionSpec:
    inputs: tuple[str, ...]
    output_fields: tuple[str, ...]
    identity_fields: tuple[str, ...] = ()
    kind: OperationKind = field(default=OperationKind.UNION, init=False)


@dataclass(frozen=True)
class RoleExpandSpec:
    input_relation: str
    mappings: tuple[RoleMapping, ...]
    output_fields: tuple[str, ...]
    carry_fields: tuple[str, ...] = ()
    role_field: str = "role"
    kind: OperationKind = field(default=OperationKind.ROLE_EXPAND, init=False)


@dataclass(frozen=True)
class CrossJoinSpec:
    left: str
    right: str
    kind: OperationKind = field(default=OperationKind.CROSS_JOIN, init=False)


@dataclass(frozen=True)
class AntiJoinSpec:
    candidate: RelationRoleRef
    observed: RelationRoleRef
    join_keys: tuple[JoinKey, ...]
    output_fields: tuple[NamedExpression, ...]
    kind: OperationKind = field(default=OperationKind.ANTI_JOIN, init=False)


@dataclass(frozen=True)
class UniversalConditionSpec:
    candidate_subject: RelationRoleRef
    required_dimension: RelationRoleRef
    observation: RelationRoleRef
    subject_keys: tuple[JoinKey, ...]
    dimension_keys: tuple[JoinKey, ...]
    condition: Expression
    output_fields: tuple[NamedExpression, ...]
    kind: OperationKind = field(
        default=OperationKind.UNIVERSAL_CONDITION,
        init=False,
    )


@dataclass(frozen=True)
class AggregateSpec:
    input_relation: str
    group_by: tuple[str, ...]
    aggregations: tuple[AggregationSpec, ...]
    kind: OperationKind = field(default=OperationKind.AGGREGATE, init=False)


@dataclass(frozen=True)
class KeepAll:
    pass


@dataclass(frozen=True)
class Take:
    limit: Expression


@dataclass(frozen=True)
class AtPosition:
    position: Expression


OrderSelection: TypeAlias = KeepAll | Take | AtPosition


def order_selection_expression(selection: OrderSelection) -> Expression | None:
    if isinstance(selection, Take):
        return selection.limit
    if isinstance(selection, AtPosition):
        return selection.position
    return None


@dataclass(frozen=True)
class OrderSpec:
    input_relation: str
    order_by: tuple[SortKey, ...]
    selection: OrderSelection
    kind: OperationKind = field(default=OperationKind.ORDER, init=False)


@dataclass(frozen=True)
class ComputeSpec:
    expression: Expression
    output_scalar: str = ""
    kind: OperationKind = field(default=OperationKind.COMPUTE, init=False)


def compute_value_input_id(expression: Expression) -> str:
    if isinstance(expression, (ParameterRef, ConstantRef)):
        return expression_input_id(expression)
    raise ValueError("compute input coverage requires a parameter or constant")


@dataclass(frozen=True)
class SqlColumnBinding:
    name: str
    field_id: str


@dataclass(frozen=True)
class SqlRelationInput:
    name: str
    relation_id: str
    columns: tuple[SqlColumnBinding, ...]

    def __post_init__(self):
        names = tuple(item.name.casefold() for item in self.columns)
        if not self.name or not self.relation_id or len(set(names)) != len(names):
            raise ValueError('SQL relation input requires unique named columns')
        if any(not item.name or not item.field_id for item in self.columns):
            raise ValueError('SQL column binding is incomplete')


@dataclass(frozen=True)
class SqlNamedInput:
    name: str
    expression: Expression


SQL_VALUE_TYPES = ("integer", "number", "string", "boolean", "date", "datetime", "uuid")


@dataclass(frozen=True)
class SqlOutputField:
    id: str
    value_type: str

    def __post_init__(self):
        if self.value_type not in SQL_VALUE_TYPES:
            raise ValueError("SQL output has an unsupported scalar type")


@dataclass(frozen=True)
class SqlQuerySpec:
    query: str
    inputs: tuple[SqlRelationInput, ...]
    outputs: tuple[SqlOutputField, ...]
    parameters: tuple[SqlNamedInput, ...] = ()
    scalar: bool = False
    # Fixed lexical inputs interpreted against catalog semantics when authoring
    # this query. They carry evidence and cannot be rebound as runtime values.
    meaning_inputs: tuple[ParameterRef, ...] = ()
    entity_keys: tuple[EntityKeyProjection, ...] = ()
    reference_input_ref: str = ""
    reference_operand: str = ""
    timezone: str = "UTC"
    kind: OperationKind = field(default=OperationKind.SQL_QUERY, init=False)

    def __post_init__(self):
        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError("SQL timezone must be a nonempty name")
        if not isinstance(self.reference_input_ref,str) or (self.reference_input_ref and not self.reference_input_ref.strip()):
            raise ValueError('Reference query input ref must be a nonempty string when present')
        if not isinstance(self.reference_operand,str):
            raise ValueError("Reference operand must be text")
        if self.reference_operand and not self.reference_input_ref:
            raise ValueError("Reference operand requires an owning input")
        if self.reference_input_ref:
            if not self.scalar or len(self.entity_keys)!=1 or {item.id for item in self.outputs}!={item.field_id for item in self.entity_keys[0].components}:
                raise ValueError('Reference query must return exactly one complete identity key')

        if not self.query.strip() or not self.inputs or not self.outputs:
            raise ValueError('SQL operation requires a query, input views and output fields')
        for names in (tuple(item.name for item in self.inputs),
                      tuple(item.name for item in self.parameters),
                      tuple(item.id for item in self.outputs)):
            if len(names) != len(set(names)) or any(not name for name in names):
                raise ValueError('SQL operation names must be nonempty and unique')


OperationSpec: TypeAlias = (
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
class Operation:
    id: str
    spec: OperationSpec
    output_relation: str = ""

    @property
    def kind(self) -> OperationKind:
        return self.spec.kind

    @property
    def input_relation_ids(self) -> tuple[str, ...]:
        return operation_input_relation_ids(self.spec)

    @property
    def output_scalar(self) -> str:
        return self.spec.output_scalar if isinstance(self.spec, ComputeSpec) else ""


def operation_input_relation_ids(spec: OperationSpec) -> tuple[str, ...]:
    """Project relation dependencies from the closed operation union."""

    if isinstance(
        spec,
        (
            FilterSpec,
            ProjectSpec,
            ProjectToKeySpec,
            RoleExpandSpec,
            AggregateSpec,
            OrderSpec,
        ),
    ):
        return (spec.input_relation,)
    if isinstance(spec, (JoinSpec, CrossJoinSpec)):
        return (spec.left, spec.right)
    if isinstance(spec, UnionSpec):
        return spec.inputs
    if isinstance(spec, AntiJoinSpec):
        return (spec.candidate.relation_id, spec.observed.relation_id)
    if isinstance(spec, UniversalConditionSpec):
        return (
            spec.candidate_subject.relation_id,
            spec.required_dimension.relation_id,
            spec.observation.relation_id,
        )
    if isinstance(spec, ComputeSpec):
        return ()
    if isinstance(spec, SqlQuerySpec):
        return tuple(dict.fromkeys(item.relation_id for item in spec.inputs))
    assert_never(spec)


def operation_expression_references(spec: OperationSpec) -> tuple[ExpressionReferences, ...]:
    expressions: tuple[Expression, ...]
    if isinstance(spec, SqlQuerySpec):
        expressions = (*tuple(item.expression for item in spec.parameters), *spec.meaning_inputs)
    elif isinstance(spec, ComputeSpec):
        expressions = (spec.expression,)
    elif isinstance(spec, FilterSpec):
        expressions = (spec.condition,)
    elif isinstance(spec, ProjectSpec):
        expressions = tuple(output.expression for output in spec.outputs)
    elif isinstance(spec, UniversalConditionSpec):
        expressions = (spec.condition,)
    elif isinstance(spec, AggregateSpec):
        expressions = tuple(item.filter for item in spec.aggregations if item.filter is not None)
    elif isinstance(spec, OrderSpec):
        expression = order_selection_expression(spec.selection)
        expressions = () if expression is None else (expression,)
    else:
        expressions = ()
    return tuple(expression_references(expression) for expression in expressions)


def operation_node_output_refs(spec: OperationSpec) -> tuple[NodeOutputRef, ...]:
    return tuple(dict.fromkeys(ref for references in operation_expression_references(spec) for ref in references.outputs))


def operation_scalar_output_ids(spec: OperationSpec) -> tuple[str, ...]:
    """Return outputs that one operation proves are scalar-addressable."""

    if isinstance(spec, ComputeSpec):
        return (spec.output_scalar,) if spec.output_scalar else ()
    if isinstance(spec, SqlQuerySpec) and spec.scalar:
        return tuple(item.id for item in spec.outputs)
    if isinstance(spec, AggregateSpec) and not spec.group_by:
        return tuple(aggregation.output_field for aggregation in spec.aggregations)
    return ()
