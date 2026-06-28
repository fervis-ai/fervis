"""Typed fact-plan outcomes and runtime result outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Mapping

from fervis.lookup.outcomes.clarifications import Clarification

if TYPE_CHECKING:
    from fervis.lookup.plan_execution.relations import RelationRows
    from fervis.lookup.fact_plan.render_spec import RenderSpec


class OutcomeKind(StrEnum):
    ANSWER = "answer"
    NEEDS_CLARIFICATION = "needs_clarification"
    IMPOSSIBLE = "impossible"
    NO_DATA = "no_data"
    UNDEFINED = "undefined"


class EmptyRelationKind(StrEnum):
    BINDING_CANDIDATES = "binding_candidates"
    ANSWER_ROWS = "answer_rows"
    OPERATION_ROWS = "operation_rows"


class BlockedRequirementKind(StrEnum):
    FIELD = "field"
    RELATION = "relation"
    PERMISSION = "permission"
    POLICY = "policy"
    COMPLETE_EVIDENCE_PATH = "complete_evidence_path"
    OPERATION_NOT_SUPPORTED_BY_CATALOG = "operation_not_supported_by_catalog"


class UndefinedReasonCode(StrEnum):
    DIVISION_BY_ZERO = "division_by_zero"
    EMPTY_AVERAGE = "empty_average"
    EMPTY_MIN = "empty_min"
    EMPTY_MAX = "empty_max"
    EMPTY_REQUIRED_SCALAR = "empty_required_scalar"


@dataclass(frozen=True)
class BlockedRequirementField:
    read_id: str
    field_id: str


@dataclass(frozen=True)
class BlockedRequirement:
    id: str
    kind: BlockedRequirementKind
    requested_fact_id: str
    fact_ref: str
    required_for: str = ""
    reviewed_read_ids: tuple[str, ...] = ()
    nearest_fields: tuple[BlockedRequirementField, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmptyRelation:
    kind: EmptyRelationKind
    relation_id: str
    grain_keys: tuple[str, ...] = ()
    requested_fact_ids: tuple[str, ...] = ()
    scope_ref: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class UndefinedOperationRef:
    operation_id: str
    reason_code: UndefinedReasonCode
    input_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class NeedsClarification:
    clarifications: tuple[Clarification, ...] = ()
    proof_refs: tuple[str, ...] = ()
    kind: OutcomeKind = OutcomeKind.NEEDS_CLARIFICATION

    def __post_init__(self) -> None:
        if not self.clarifications:
            raise ValueError("needs clarification requires clarifications")


@dataclass(frozen=True)
class Impossible:
    blocked_requirements: tuple[BlockedRequirement, ...] = ()
    proof_refs: tuple[str, ...] = ()
    kind: OutcomeKind = OutcomeKind.IMPOSSIBLE

    def __post_init__(self) -> None:
        if not self.blocked_requirements:
            raise ValueError("impossible outcome requires blocked requirements")


@dataclass(frozen=True)
class NoData:
    empty_relation: EmptyRelation
    proof_refs: tuple[str, ...] = ()
    kind: OutcomeKind = OutcomeKind.NO_DATA


@dataclass(frozen=True)
class Undefined:
    operation: UndefinedOperationRef
    proof_refs: tuple[str, ...] = ()
    kind: OutcomeKind = OutcomeKind.UNDEFINED


@dataclass(frozen=True)
class AnswerResult:
    render_spec: RenderSpec | None = None
    relations: tuple["RelationRows", ...] = ()
    scalars: Mapping[str, object] | None = None
    proof_refs: tuple[str, ...] = ()
    kind: OutcomeKind = OutcomeKind.ANSWER


ResultOutcome = AnswerResult | NeedsClarification | Impossible | NoData | Undefined


@dataclass(frozen=True)
class FactResult:
    outcome: ResultOutcome
