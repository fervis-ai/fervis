"""Typed values and value-use sinks for lookup fact plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

ANCHOR_DATE_REF = "ANCHOR_DATE"
ANCHOR_TIMEZONE_REF = "ANCHOR_TIMEZONE"


class ValueKind(StrEnum):
    IDENTITY = "identity"
    IDENTITY_SET = "identity_set"
    NAMED = "named"
    TIME = "time"
    LITERAL = "literal"


class LiteralType(StrEnum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class ValueUseKind(StrEnum):
    ROW_FILTER = "row_filter"
    SCALAR_INPUT = "scalar_input"
    RANK_LIMIT = "rank_limit"


class ValueFilterOperator(StrEnum):
    EQUALS = "equals"
    IN = "in"


class ValueComponent(StrEnum):
    VALUE = "value"


class TimeComponent(StrEnum):
    START = "start"
    END = "end"
    INSTANT = "instant"


@dataclass(frozen=True)
class IdentityValuePayload:
    identity_type: str
    identity_field: str
    value: str
    display_value: str = ""
    matched_field_ref: str = ""
    matched_field_path: str = ""


@dataclass(frozen=True)
class IdentitySetValuePayload:
    identity_type: str
    identity_field: str
    values: tuple[str, ...]
    display_value: str = ""
    source_relation_id: str = ""


@dataclass(frozen=True)
class NamedValuePayload:
    text: str
    reference_text: str = ""


@dataclass(frozen=True)
class TimeValuePayload:
    expression: str
    intent: dict[str, object] = field(default_factory=dict)
    anchor_date_ref: str = ANCHOR_DATE_REF
    timezone_ref: str = ANCHOR_TIMEZONE_REF
    resolved_start: str = ""
    resolved_end: str = ""
    granularity: str = ""


@dataclass(frozen=True)
class LiteralValuePayload:
    literal_type: LiteralType
    value: str


@dataclass(frozen=True)
class FactValue:
    id: str
    kind: ValueKind
    label: str = ""
    payload: (
        IdentityValuePayload
        | IdentitySetValuePayload
        | NamedValuePayload
        | TimeValuePayload
        | LiteralValuePayload
        | None
    ) = None
    proof_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    applies_to_requested_fact_ids: tuple[str, ...] = ()

    @classmethod
    def identity(
        cls,
        *,
        id: str,
        identity_type: str,
        identity_field: str,
        value: str,
        display_value: str = "",
        matched_field_ref: str = "",
        matched_field_path: str = "",
        proof_refs: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
    ) -> FactValue:
        return cls(
            id=id,
            kind=ValueKind.IDENTITY,
            label=display_value or value,
            payload=IdentityValuePayload(
                identity_type=identity_type,
                identity_field=identity_field,
                value=value,
                display_value=display_value,
                matched_field_ref=matched_field_ref,
                matched_field_path=matched_field_path,
            ),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            applies_to_requested_fact_ids=tuple(applies_to_requested_fact_ids),
        )

    @classmethod
    def identity_set(
        cls,
        *,
        id: str,
        identity_type: str,
        identity_field: str,
        values: tuple[str, ...],
        display_value: str = "",
        source_relation_id: str = "",
        proof_refs: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
    ) -> FactValue:
        return cls(
            id=id,
            kind=ValueKind.IDENTITY_SET,
            label=display_value or f"{len(values)} {identity_type} identities",
            payload=IdentitySetValuePayload(
                identity_type=identity_type,
                identity_field=identity_field,
                values=tuple(values),
                display_value=display_value,
                source_relation_id=source_relation_id,
            ),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            applies_to_requested_fact_ids=tuple(applies_to_requested_fact_ids),
        )

    @classmethod
    def named(
        cls,
        *,
        id: str,
        text: str,
        reference_text: str = "",
        proof_refs: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
    ) -> FactValue:
        return cls(
            id=id,
            kind=ValueKind.NAMED,
            label=text,
            payload=NamedValuePayload(
                text=text,
                reference_text=reference_text or text,
            ),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            applies_to_requested_fact_ids=tuple(applies_to_requested_fact_ids),
        )

    @classmethod
    def time(
        cls,
        *,
        id: str,
        expression: str,
        intent: dict[str, object] | None = None,
        anchor_date_ref: str = ANCHOR_DATE_REF,
        timezone_ref: str = ANCHOR_TIMEZONE_REF,
        resolved_start: str = "",
        resolved_end: str = "",
        granularity: str = "",
        proof_refs: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
    ) -> FactValue:
        return cls(
            id=id,
            kind=ValueKind.TIME,
            label=expression,
            payload=TimeValuePayload(
                expression=expression,
                intent=dict(intent or {}),
                anchor_date_ref=anchor_date_ref,
                timezone_ref=timezone_ref,
                resolved_start=resolved_start,
                resolved_end=resolved_end,
                granularity=granularity,
            ),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            applies_to_requested_fact_ids=tuple(applies_to_requested_fact_ids),
        )

    @classmethod
    def literal(
        cls,
        *,
        id: str,
        literal_type: LiteralType,
        value: str,
        label: str = "",
        proof_refs: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
    ) -> FactValue:
        return cls(
            id=id,
            kind=ValueKind.LITERAL,
            label=label,
            payload=LiteralValuePayload(literal_type=literal_type, value=value),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            applies_to_requested_fact_ids=tuple(applies_to_requested_fact_ids),
        )


def known_input_id_for_value(value: FactValue) -> str:
    for proof_ref in value.proof_refs:
        if proof_ref.startswith("known_input:"):
            return proof_ref.removeprefix("known_input:")
    return ""


@dataclass(frozen=True)
class RowFilterUse:
    relation_id: str
    field_id: str
    operator: ValueFilterOperator
    value_component: ValueComponent | TimeComponent = ValueComponent.VALUE
    kind: ValueUseKind = ValueUseKind.ROW_FILTER


@dataclass(frozen=True)
class ScalarInputUse:
    operation_id: str
    input_id: str
    kind: ValueUseKind = ValueUseKind.SCALAR_INPUT


@dataclass(frozen=True)
class RankLimitUse:
    operation_id: str
    kind: ValueUseKind = ValueUseKind.RANK_LIMIT


ValueUseTarget = RowFilterUse | ScalarInputUse | RankLimitUse


@dataclass(frozen=True)
class ValueUse:
    id: str
    value_id: str
    target: ValueUseTarget
