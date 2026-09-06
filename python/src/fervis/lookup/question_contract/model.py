"""Canonical catalog-blind relational meaning of one factual question."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from fervis.lookup.expression_operators import (
    ExpressionBinaryOperator,
    ExpressionUnaryOperator,
)
from fervis.lookup.semantic_types import (
    CollectionType,
    ScalarType,
    SourceOrigin,
    ValueType,
)
from fervis.types.enums import StrEnum


class FactLocalKind(StrEnum):
    SET = "set"
    ASSOCIATION = "association"
    FACT = "fact"
    EXPRESSION = "expression"
    OUTPUT = "output"


@dataclass(frozen=True, order=True)
class FactLocalRef:
    requested_fact_id: str
    kind: FactLocalKind
    local_id: str

    def __post_init__(self) -> None:
        if not self.requested_fact_id or not self.local_id:
            raise ValueError("fact-local reference is incomplete")

    @property
    def token(self) -> str:
        return f"{self.requested_fact_id}:{self.kind.value}:{self.local_id}"

    @classmethod
    def from_token(cls, token: str) -> "FactLocalRef":
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError("fact-local reference token is malformed")
        requested_fact_id, kind, local_id = parts
        return cls(requested_fact_id, FactLocalKind(kind), local_id)


@dataclass(frozen=True)
class SetTerm:
    id: str
    origin: SourceOrigin


@dataclass(frozen=True)
class AssociationTerm:
    id: str
    from_set_ref: str
    to_set_ref: str
    origin: SourceOrigin


@dataclass(frozen=True)
class FactTerm:
    id: str
    owner_ref: str
    value_type: ScalarType
    origin: SourceOrigin


@dataclass(frozen=True)
class InputTerm:
    id: str
    origin: SourceOrigin
    operand: str | tuple[str, ...]
    value_type: ValueType

    def __post_init__(self) -> None:
        if isinstance(self.value_type, CollectionType):
            if (
                not isinstance(self.operand, tuple)
                or not self.operand
                or any(not item for item in self.operand)
                or len(set(self.operand)) != len(self.operand)
            ):
                raise ValueError("collection input requires unique operand values")
        elif not isinstance(self.operand, str) or not self.operand:
            raise ValueError("scalar input requires one operand value")


class InputDenotationKind(StrEnum):
    IDENTITY_REFERENCE = "IDENTITY_REFERENCE"
    NON_IDENTITY_SCALAR = "NON_IDENTITY_SCALAR"


@dataclass(frozen=True)
class InputDenotation:
    id: str
    input_ref: str
    operand_meaning: str
    denotation_basis: str
    denoted_instance_kind: str | None
    kind: InputDenotationKind

    def __post_init__(self) -> None:
        if (
            not self.id
            or not self.input_ref
            or not self.operand_meaning.strip()
            or not self.denotation_basis.strip()
        ):
            raise ValueError("input denotation is incomplete")
        if (self.kind is InputDenotationKind.IDENTITY_REFERENCE) != (
            self.denoted_instance_kind is not None
        ):
            raise ValueError("denoted instance kind must match input denotation kind")
        if (
            self.denoted_instance_kind is not None
            and not self.denoted_instance_kind.strip()
        ):
            raise ValueError("denoted instance kind must be non-empty")


class BooleanCompositionOperator(StrEnum):
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass(frozen=True)
class BooleanComposition:
    id: str
    operator: BooleanCompositionOperator
    argument_refs: tuple[str, ...]
    origin: SourceOrigin


@dataclass(frozen=True)
class Comparison:
    id: str
    operator: ExpressionBinaryOperator
    left_ref: str
    right_ref: str
    origin: SourceOrigin


@dataclass(frozen=True)
class NullCheck:
    id: str
    operator: ExpressionUnaryOperator
    argument_ref: str
    origin: SourceOrigin


@dataclass(frozen=True)
class Arithmetic:
    id: str
    operator: ExpressionUnaryOperator | ExpressionBinaryOperator
    argument_refs: tuple[str, ...]
    origin: SourceOrigin


class TemporalGrain(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass(frozen=True)
class TemporalBucket:
    id: str
    value_ref: str
    grain: TemporalGrain
    origin: SourceOrigin


class AggregateFunction(StrEnum):
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


@dataclass(frozen=True)
class Aggregate:
    id: str
    function: AggregateFunction
    argument_ref: str
    filter_ref: str | None
    distinct_argument: bool
    origin: SourceOrigin


class Quantifier(StrEnum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    FORALL = "forall"


@dataclass(frozen=True)
class Quantify:
    id: str
    quantifier: Quantifier
    over_set_ref: str
    association_refs: tuple[str, ...]
    condition_ref: str
    origin: SourceOrigin


@dataclass(frozen=True)
class RelatedRow:
    id: str
    set_ref: str
    association_refs: tuple[str, ...]
    condition_ref: str | None
    origin: SourceOrigin


@dataclass(frozen=True)
class Coverage:
    id: str
    candidate_set_ref: str
    required_dimension_set_ref: str
    observation_set_ref: str
    candidate_observation_association_refs: tuple[str, ...]
    dimension_observation_association_refs: tuple[str, ...]
    required_member_condition_ref: str | None
    condition_ref: str
    origin: SourceOrigin


ExpressionNode: TypeAlias = (
    BooleanComposition
    | Comparison
    | NullCheck
    | Arithmetic
    | TemporalBucket
    | Aggregate
    | Quantify
    | RelatedRow
    | Coverage
)


class InstanceInterpretation(StrEnum):
    NORMAL_BUSINESS_INSTANCE = "normal_business_instance"
    RAW_DATA_RECORD = "raw_data_record"


@dataclass(frozen=True)
class Subject:
    set_ref: str
    instance_interpretation: InstanceInterpretation


@dataclass(frozen=True)
class RequestedOutput:
    id: str
    expression_ref: str
    origin: SourceOrigin


class OrderingDirection(StrEnum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True)
class Ordering:
    expression_ref: str
    direction: OrderingDirection
    origin: SourceOrigin


@dataclass(frozen=True)
class AllResults:
    pass


@dataclass(frozen=True)
class FirstRankWithTies:
    pass


@dataclass(frozen=True)
class TakeWithBoundaryTies:
    limit_input_ref: str


@dataclass(frozen=True)
class PositionWithTies:
    limit_input_ref: str


ResultSelection: TypeAlias = AllResults | FirstRankWithTies | TakeWithBoundaryTies | PositionWithTies


@dataclass(frozen=True)
class RequestedFact:
    id: str
    origin: SourceOrigin
    sets: tuple[SetTerm, ...]
    associations: tuple[AssociationTerm, ...]
    facts: tuple[FactTerm, ...]
    expressions: tuple[ExpressionNode, ...]
    subject: Subject
    qualification_ref: str | None
    grouping_refs: tuple[str, ...]
    outputs: tuple[RequestedOutput, ...]
    ordering: tuple[Ordering, ...]
    selection: ResultSelection
    distinct_by: tuple[str, ...]


@dataclass(frozen=True)
class QuestionContract:
    inputs: tuple[InputTerm, ...]
    requested_facts: tuple[RequestedFact, ...]
    input_denotations: tuple[InputDenotation, ...] = ()

    def __post_init__(self) -> None:
        input_refs = {item.id for item in self.inputs}
        denotation_refs = [item.input_ref for item in self.input_denotations]
        if len(denotation_refs) != len(set(denotation_refs)):
            raise ValueError("one supplied input has multiple denotations")
        if input_refs != set(denotation_refs):
            raise ValueError("input denotations must cover every supplied input")


__all__ = tuple(name for name in globals() if not name.startswith("_"))
