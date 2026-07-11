"""Typed values for canonical answer programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Callable, TypeVar, assert_never

from fervis.lookup.answer_program.errors import AnswerProgramContractError

ANCHOR_DATE_REF = "ANCHOR_DATE"
ANCHOR_TIMEZONE_REF = "ANCHOR_TIMEZONE"


class ValueKind(StrEnum):
    IDENTITY = "identity"
    IDENTITY_SET = "identity_set"
    NAMED = "named"
    TIME = "time"
    LITERAL = "literal"
    STRING_SET = "string_set"


class LiteralType(StrEnum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class ValueFilterOperator(StrEnum):
    EQUALS = "equals"
    IN = "in"


class ValueDependencyKind(StrEnum):
    CONVERSATION_MEMORY = "conversation_memory"


@dataclass(frozen=True)
class ValueDependency:
    kind: ValueDependencyKind
    ref: str

    def __post_init__(self) -> None:
        if not self.ref:
            raise ValueError("value dependency requires a reference")


class ValueComponent(StrEnum):
    VALUE = "value"


class TimeComponent(StrEnum):
    START = "start"
    END = "end"
    INSTANT = "instant"


class TimeGranularity(StrEnum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass(frozen=True)
class IdentityValuePayload:
    identity_type: str
    identity_field: str
    value: str
    display_value: str = ""
    matched_field_ref: str = ""
    matched_field_path: str = ""

    @property
    def kind(self) -> ValueKind:
        return ValueKind.IDENTITY

    @property
    def parameter_value_type(self) -> str:
        return "identity"

    def canonical_value(self) -> str:
        return self.value

    def component_value(self, component: ValueComponent | TimeComponent) -> object:
        if component is not ValueComponent.VALUE:
            raise ValueError("identity has only a value component")
        return self.value

    @property
    def row_filter_error(self) -> str:
        return ""


@dataclass(frozen=True)
class IdentitySetValuePayload:
    identity_type: str
    identity_field: str
    values: tuple[str, ...]
    display_value: str = ""
    source_relation_id: str = ""

    def __post_init__(self) -> None:
        if len(set(self.values)) != len(self.values):
            raise AnswerProgramContractError(
                "duplicate_set_value",
                "identity-set value cannot contain duplicates",
            )
        object.__setattr__(self, "values", tuple(sorted(self.values)))

    @property
    def kind(self) -> ValueKind:
        return ValueKind.IDENTITY_SET

    @property
    def parameter_value_type(self) -> str:
        return "identity_set"

    def canonical_value(self) -> list[str]:
        return list(self.values)

    def component_value(self, component: ValueComponent | TimeComponent) -> object:
        if component is not ValueComponent.VALUE:
            raise ValueError("identity set has only a value component")
        return self.values

    @property
    def row_filter_error(self) -> str:
        return "identity set cannot be a row filter"


@dataclass(frozen=True)
class NamedValuePayload:
    text: str
    reference_text: str = ""

    @property
    def kind(self) -> ValueKind:
        return ValueKind.NAMED

    @property
    def parameter_value_type(self) -> str:
        return "named"

    def canonical_value(self) -> str:
        return self.text

    def component_value(self, component: ValueComponent | TimeComponent) -> object:
        if component is not ValueComponent.VALUE:
            raise ValueError("named value has only a value component")
        return self.text

    @property
    def row_filter_error(self) -> str:
        return ""


@dataclass(frozen=True)
class TimeValuePayload:
    expression: str
    intent: dict[str, object] = field(default_factory=dict)
    anchor_date_ref: str = ANCHOR_DATE_REF
    timezone_ref: str = ANCHOR_TIMEZONE_REF
    resolved_start: str = ""
    resolved_end: str = ""
    granularity: str = ""

    def __post_init__(self) -> None:
        expression = self.expression.strip()
        if not expression:
            raise AnswerProgramContractError(
                "invalid_time_value",
                "time value requires an expression",
            )
        if not self.resolved_start or not self.resolved_end or not self.granularity:
            raise AnswerProgramContractError(
                "invalid_time_value",
                "time value requires resolved bounds and granularity",
            )
        try:
            granularity = TimeGranularity(self.granularity)
            start_kind, start = _parse_iso_boundary(self.resolved_start)
            end_kind, end = _parse_iso_boundary(self.resolved_end)
        except ValueError as exc:
            raise AnswerProgramContractError(
                "invalid_time_value",
                "time value has an invalid resolved interval",
            ) from exc
        if start_kind != end_kind:
            raise AnswerProgramContractError(
                "invalid_time_value",
                "time bounds must use the same ISO representation",
            )
        if granularity is TimeGranularity.HOUR and start_kind != "datetime":
            raise AnswerProgramContractError(
                "invalid_time_value",
                "hour granularity requires datetime bounds",
            )
        if granularity is not TimeGranularity.HOUR and start_kind != "date":
            raise AnswerProgramContractError(
                "invalid_time_value",
                "calendar granularity requires date bounds",
            )
        if (
            start_kind == "datetime"
            and isinstance(start, datetime)
            and isinstance(end, datetime)
            and ((start.tzinfo is None) != (end.tzinfo is None))
        ):
            raise AnswerProgramContractError(
                "invalid_time_value",
                "datetime bounds must use consistent timezone awareness",
            )
        if start > end:
            raise AnswerProgramContractError(
                "invalid_time_value",
                "time value start must not follow its end",
            )
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "resolved_start", start.isoformat())
        object.__setattr__(self, "resolved_end", end.isoformat())
        object.__setattr__(self, "granularity", granularity.value)

    @property
    def kind(self) -> ValueKind:
        return ValueKind.TIME

    @property
    def parameter_value_type(self) -> str:
        return "time"

    def canonical_value(self) -> dict[str, str]:
        return {
            "expression": self.expression,
            "resolved_start": self.resolved_start,
            "resolved_end": self.resolved_end,
            "granularity": self.granularity,
        }

    def component_value(self, component: ValueComponent | TimeComponent) -> object:
        if component is ValueComponent.VALUE or component is TimeComponent.INSTANT:
            if self.resolved_start != self.resolved_end:
                raise ValueError("time value does not have an instant")
            return self.resolved_start
        if component is TimeComponent.START:
            return self.resolved_start
        if component is TimeComponent.END:
            return self.resolved_end
        raise ValueError("time value requires an explicit time component")

    @property
    def row_filter_error(self) -> str:
        return ""


@dataclass(frozen=True)
class LiteralValuePayload:
    literal_type: LiteralType
    value: str

    def __post_init__(self) -> None:
        if self.literal_type is LiteralType.NUMBER:
            object.__setattr__(self, "value", _normalized_number(self.value))
            return
        if self.literal_type is LiteralType.BOOLEAN:
            value = self.value.strip().casefold()
            if value not in {"true", "false"}:
                raise AnswerProgramContractError(
                    "binding_type_mismatch",
                    "boolean binding must be true or false",
                )
            object.__setattr__(self, "value", value)

    @property
    def kind(self) -> ValueKind:
        return ValueKind.LITERAL

    @property
    def parameter_value_type(self) -> str:
        return self.literal_type.value

    def canonical_value(self) -> object:
        if self.literal_type is LiteralType.NUMBER:
            return _decimal_number(self.value)
        if self.literal_type is LiteralType.BOOLEAN:
            return self.value.strip().lower() == "true"
        return self.value

    def component_value(self, component: ValueComponent | TimeComponent) -> object:
        if component is not ValueComponent.VALUE:
            raise ValueError("literal has only a value component")
        if self.literal_type is LiteralType.NUMBER:
            return _decimal_number(self.value)
        if self.literal_type is LiteralType.BOOLEAN:
            return self.value == "true"
        return self.value

    @property
    def row_filter_error(self) -> str:
        if self.literal_type is LiteralType.STRING:
            return "literal value cannot be a row filter"
        return "literal value requires a scalar sink"


@dataclass(frozen=True)
class StringSetValuePayload:
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("string-set value requires at least one value")
        if len(set(self.values)) != len(self.values):
            raise AnswerProgramContractError(
                "duplicate_set_value",
                "string-set value cannot contain duplicates",
            )
        object.__setattr__(self, "values", tuple(sorted(self.values)))

    @property
    def kind(self) -> ValueKind:
        return ValueKind.STRING_SET

    @property
    def parameter_value_type(self) -> str:
        return "string_set"

    def canonical_value(self) -> list[str]:
        return list(self.values)

    def component_value(self, component: ValueComponent | TimeComponent) -> object:
        if component is not ValueComponent.VALUE:
            raise ValueError("string set has only a value component")
        return self.values

    @property
    def row_filter_error(self) -> str:
        return ""


def _normalized_number(value: str) -> str:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        raise AnswerProgramContractError(
            "binding_type_mismatch",
            "number binding contains a non-numeric value",
        ) from exc
    if not parsed.is_finite():
        raise AnswerProgramContractError(
            "binding_type_mismatch",
            "number binding must be finite",
        )
    if parsed == 0:
        return "0"
    return format(parsed.normalize(), "f")


def _decimal_number(value: str) -> Decimal:
    return Decimal(value)


def _parse_iso_boundary(value: str) -> tuple[str, date | datetime]:
    if value != value.strip():
        raise ValueError("ISO boundary cannot contain surrounding whitespace")
    if "T" not in value:
        return "date", date.fromisoformat(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return "datetime", parsed


@dataclass(frozen=True)
class FactValue:
    id: str
    payload: (
        IdentityValuePayload
        | IdentitySetValuePayload
        | NamedValuePayload
        | TimeValuePayload
        | LiteralValuePayload
        | StringSetValuePayload
    )
    known_input_id: str = ""
    label: str = ""
    proof_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    dependencies: tuple[ValueDependency, ...] = ()
    applies_to_requested_fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dependencies, tuple):
            raise TypeError("value dependencies must be a tuple")
        canonical = tuple(
            sorted(
                set(self.dependencies),
                key=lambda dependency: (dependency.kind.value, dependency.ref),
            )
        )
        object.__setattr__(self, "dependencies", canonical)

    @property
    def kind(self) -> ValueKind:
        return self.payload.kind

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
        dependencies: tuple[ValueDependency, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
        known_input_id: str = "",
    ) -> FactValue:
        return cls(
            id=id,
            known_input_id=known_input_id,
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
            dependencies=tuple(dependencies),
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
        dependencies: tuple[ValueDependency, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
        known_input_id: str = "",
    ) -> FactValue:
        return cls(
            id=id,
            known_input_id=known_input_id,
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
            dependencies=tuple(dependencies),
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
        dependencies: tuple[ValueDependency, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
        known_input_id: str = "",
    ) -> FactValue:
        return cls(
            id=id,
            known_input_id=known_input_id,
            label=text,
            payload=NamedValuePayload(
                text=text,
                reference_text=reference_text or text,
            ),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            dependencies=tuple(dependencies),
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
        dependencies: tuple[ValueDependency, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
        known_input_id: str = "",
    ) -> FactValue:
        return cls(
            id=id,
            known_input_id=known_input_id,
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
            dependencies=tuple(dependencies),
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
        dependencies: tuple[ValueDependency, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
        known_input_id: str = "",
    ) -> FactValue:
        return cls(
            id=id,
            known_input_id=known_input_id,
            label=label,
            payload=LiteralValuePayload(literal_type=literal_type, value=value),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            dependencies=tuple(dependencies),
            applies_to_requested_fact_ids=tuple(applies_to_requested_fact_ids),
        )

    @classmethod
    def string_set(
        cls,
        *,
        id: str,
        values: tuple[str, ...],
        label: str = "",
        proof_refs: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        dependencies: tuple[ValueDependency, ...] = (),
        applies_to_requested_fact_ids: tuple[str, ...] = (),
        known_input_id: str = "",
    ) -> FactValue:
        return cls(
            id=id,
            known_input_id=known_input_id,
            label=label,
            payload=StringSetValuePayload(values=tuple(values)),
            proof_refs=tuple(proof_refs),
            source_refs=tuple(source_refs),
            dependencies=tuple(dependencies),
            applies_to_requested_fact_ids=tuple(applies_to_requested_fact_ids),
        )


def known_input_id_for_value(value: FactValue) -> str:
    return value.known_input_id


class ValueExpressionOrigin(StrEnum):
    PARAMETER = "parameter"
    NODE_OUTPUT = "node_output"
    CONSTANT = "constant"
    ENVIRONMENT = "environment"


@dataclass(frozen=True)
class ParameterRef:
    parameter_id: str
    component: str = "value"
    item_index: int | None = None
    origin: ValueExpressionOrigin = field(
        default=ValueExpressionOrigin.PARAMETER,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.parameter_id:
            raise ValueError("parameter reference requires parameter id")
        if not self.component:
            raise ValueError("parameter reference requires component")
        if self.item_index is not None and self.item_index < 0:
            raise ValueError("parameter item index cannot be negative")


@dataclass(frozen=True)
class NodeOutputRef:
    node_id: str
    output_id: str
    origin: ValueExpressionOrigin = field(
        default=ValueExpressionOrigin.NODE_OUTPUT,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.node_id or not self.output_id:
            raise ValueError("node-output reference requires node and output ids")


@dataclass(frozen=True)
class ConstantRef:
    constant_id: str
    version_ref: str
    value: FactValue
    component: str = "value"
    item_index: int | None = None
    origin: ValueExpressionOrigin = field(
        default=ValueExpressionOrigin.CONSTANT,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.constant_id or not self.version_ref:
            raise ValueError("constant reference requires id and version")
        if not self.component:
            raise ValueError("constant reference requires component")
        if self.item_index is not None and self.item_index < 0:
            raise ValueError("constant item index cannot be negative")


@dataclass(frozen=True)
class EnvironmentRef:
    key: str
    source_ref: str = ""
    origin: ValueExpressionOrigin = field(
        default=ValueExpressionOrigin.ENVIRONMENT,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("environment reference requires key")


ValueExpression = ParameterRef | NodeOutputRef | ConstantRef | EnvironmentRef

_ValueExpressionFoldResult = TypeVar("_ValueExpressionFoldResult")


def fold_value_expression(
    expression: ValueExpression,
    *,
    parameter: Callable[[ParameterRef], _ValueExpressionFoldResult],
    output: Callable[[NodeOutputRef], _ValueExpressionFoldResult],
    constant: Callable[[ConstantRef], _ValueExpressionFoldResult],
    environment: Callable[[EnvironmentRef], _ValueExpressionFoldResult],
) -> _ValueExpressionFoldResult:
    """Exhaustively interpret one closed value-expression variant."""

    if isinstance(expression, ParameterRef):
        return parameter(expression)
    if isinstance(expression, NodeOutputRef):
        return output(expression)
    if isinstance(expression, ConstantRef):
        return constant(expression)
    if isinstance(expression, EnvironmentRef):
        return environment(expression)
    assert_never(expression)


def value_expression_constant(
    expression: ValueExpression,
) -> ConstantRef | None:
    return fold_value_expression(
        expression,
        parameter=lambda _expression: None,
        output=lambda _expression: None,
        constant=lambda constant: constant,
        environment=lambda _expression: None,
    )


class ParameterRole(StrEnum):
    QUESTION_INPUT = "question_input"
    SEMANTIC_CONTROL = "semantic_control"
    PLAN_CONTROL = "plan_control"


class ParameterValueType(StrEnum):
    IDENTITY = "identity"
    IDENTITY_SET = "identity_set"
    NAMED = "named"
    TIME = "time"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    STRING_SET = "string_set"


class BindingProvenanceKind(StrEnum):
    QUESTION_INPUT = "question_input"
    SEMANTIC_CHOICE = "semantic_choice"
    PLAN_CHOICE = "plan_choice"
    RERUN_PATCH = "rerun_patch"


@dataclass(frozen=True)
class BindingProvenance:
    kind: BindingProvenanceKind
    refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BindingProvenanceKind):
            raise TypeError("binding provenance kind must be BindingProvenanceKind")
        if not isinstance(self.refs, tuple):
            raise TypeError("binding provenance refs must be a tuple")
        if any(not isinstance(ref, str) or not ref for ref in self.refs):
            raise ValueError("binding provenance refs must be non-empty strings")


@dataclass(frozen=True)
class ParameterDeclaration:
    id: str
    role: ParameterRole
    value_type: ParameterValueType
    required: bool = True
    allowed_values: tuple[str, ...] = ()
    semantic_control_ref: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("parameter declaration requires id")
        if not isinstance(self.role, ParameterRole):
            raise TypeError("parameter role must be ParameterRole")
        if not isinstance(self.value_type, ParameterValueType):
            raise TypeError("parameter value_type must be ParameterValueType")
        if not isinstance(self.required, bool):
            raise TypeError("parameter required must be bool")
        if not isinstance(self.allowed_values, tuple):
            raise TypeError("parameter allowed_values must be a tuple")
        if any(
            not isinstance(value, str) or not value
            for value in self.allowed_values
        ):
            raise ValueError("parameter allowed values must be non-empty strings")
        if not isinstance(self.semantic_control_ref, str):
            raise TypeError("parameter semantic_control_ref must be a string")
        if len(set(self.allowed_values)) != len(self.allowed_values):
            raise ValueError("parameter allowed values cannot contain duplicates")
        object.__setattr__(self, "allowed_values", tuple(sorted(self.allowed_values)))
        if self.allowed_values and self.value_type not in {
            ParameterValueType.STRING,
            ParameterValueType.STRING_SET,
        }:
            raise ValueError("allowed values require a string parameter type")


@dataclass(frozen=True)
class ParameterBinding:
    parameter_id: str
    value: FactValue
    provenance: BindingProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_id, str) or not self.parameter_id:
            raise ValueError("parameter binding requires parameter id")
        if not isinstance(self.value, FactValue):
            raise TypeError("parameter binding value must be FactValue")
        if not isinstance(self.provenance, BindingProvenance):
            raise TypeError("parameter binding provenance must be BindingProvenance")


@dataclass(frozen=True)
class BindingSet:
    bindings: tuple[ParameterBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, tuple):
            raise TypeError("binding set members must be a tuple")
        if any(not isinstance(binding, ParameterBinding) for binding in self.bindings):
            raise TypeError("binding set members must be ParameterBinding")
        ids = tuple(binding.parameter_id for binding in self.bindings)
        if len(set(ids)) != len(ids):
            raise ValueError("binding set cannot contain duplicate parameters")

    @classmethod
    def from_bindings(cls, bindings: tuple[ParameterBinding, ...]) -> "BindingSet":
        return cls(bindings=tuple(sorted(bindings, key=lambda item: item.parameter_id)))

    def get(self, parameter_id: str) -> ParameterBinding | None:
        return next(
            (
                binding
                for binding in self.bindings
                if binding.parameter_id == parameter_id
            ),
            None,
        )

    @property
    def parameter_ids(self) -> tuple[str, ...]:
        return tuple(binding.parameter_id for binding in self.bindings)


@dataclass(frozen=True)
class NamedValueExpression:
    sink: str
    expression: ValueExpression

    def __post_init__(self) -> None:
        if not self.sink:
            raise ValueError("named value expression requires sink")


@dataclass(frozen=True)
class ProgramInputs:
    parameters: tuple[ParameterDeclaration, ...]
    bindings: BindingSet
    expressions: tuple[NamedValueExpression, ...] = ()


class BindingPatchOperationKind(StrEnum):
    SET = "set"
    UNSET = "unset"


@dataclass(frozen=True)
class SetParameter:
    parameter_id: str
    value: FactValue
    kind: BindingPatchOperationKind = field(
        default=BindingPatchOperationKind.SET,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_id, str) or not self.parameter_id:
            raise ValueError("set operation requires parameter id")
        if not isinstance(self.value, FactValue):
            raise TypeError("set operation value must be FactValue")


@dataclass(frozen=True)
class UnsetParameter:
    parameter_id: str
    kind: BindingPatchOperationKind = field(
        default=BindingPatchOperationKind.UNSET,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_id, str) or not self.parameter_id:
            raise ValueError("unset operation requires parameter id")


BindingPatchOperation = SetParameter | UnsetParameter


@dataclass(frozen=True)
class BindingPatch:
    operations: tuple[BindingPatchOperation, ...]
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operations, tuple):
            raise TypeError("binding patch operations must be a tuple")
        if not isinstance(self.provenance_refs, tuple):
            raise TypeError("binding patch provenance refs must be a tuple")
        if not self.operations:
            raise ValueError("binding patch requires at least one operation")
        if any(
            not isinstance(operation, (SetParameter, UnsetParameter))
            for operation in self.operations
        ):
            raise TypeError("binding patch operations must be typed")
        if any(
            not isinstance(ref, str) or not ref for ref in self.provenance_refs
        ):
            raise ValueError("binding patch provenance refs must be non-empty strings")
        parameter_ids = tuple(operation.parameter_id for operation in self.operations)
        if len(set(parameter_ids)) != len(parameter_ids):
            raise ValueError("binding patch cannot edit one parameter more than once")

