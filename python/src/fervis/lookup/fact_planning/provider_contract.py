"""Typed provider-output contracts for fact planning."""

from dataclasses import dataclass
from typing import Optional

from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class FieldSelectionOutput(ProviderOutput):
    field_id: str


@dataclass(frozen=True)
class OrderFieldSelectionOutput(ProviderOutput):
    selection_basis: str
    field_id: str


@dataclass(frozen=True)
class MetricSelectionOutput(ProviderOutput):
    selection_basis: str
    id: str
    kind: str
    field_id: Optional[str] = None


@dataclass(frozen=True)
class FunctionSelectionOutput(ProviderOutput):
    selection_basis: str
    id: str
    value: str


@dataclass(frozen=True)
class ListRowsAnswerOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    pattern: str
    source_binding_id: str
    output_fields: tuple[FieldSelectionOutput, ...]
    ordering_field: Optional[OrderFieldSelectionOutput] = None


@dataclass(frozen=True)
class GroupedRowsAnswerOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    pattern: str
    source_binding_id: str
    group_fields: tuple[FieldSelectionOutput, ...]
    output_fields: tuple[FieldSelectionOutput, ...]


@dataclass(frozen=True)
class DirectFieldValueAnswerOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    pattern: str
    source_binding_id: str
    output_field: FieldSelectionOutput


@dataclass(frozen=True)
class AggregateScalarAnswerOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    pattern: str
    source_binding_id: str
    metric: MetricSelectionOutput
    function: FunctionSelectionOutput


@dataclass(frozen=True)
class GroupedAggregateAnswerOutput(ProviderOutput):
    requested_fact_id: str
    pattern: str
    source_binding_id: str
    metric: MetricSelectionOutput
    function: FunctionSelectionOutput
    ordering_field: Optional[OrderFieldSelectionOutput] = None


@dataclass(frozen=True)
class SetOperandOutput(ProviderOutput):
    source_binding_id: str
    identity_fields: tuple[str, ...]


@dataclass(frozen=True)
class SetDifferenceAnswerOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    pattern: str
    candidate: SetOperandOutput
    observed: SetOperandOutput


@dataclass(frozen=True)
class JoinOperandOutput(ProviderOutput):
    source_binding_id: str
    fields: tuple[FieldSelectionOutput, ...]


@dataclass(frozen=True)
class JoinKeyOutput(ProviderOutput):
    left_field_id: str
    right_field_id: str


@dataclass(frozen=True)
class JoinedFieldOutput(ProviderOutput):
    side: str
    field_id: str


@dataclass(frozen=True)
class JoinedRowsAnswerOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    pattern: str
    left: JoinOperandOutput
    right: JoinOperandOutput
    join_keys: tuple[JoinKeyOutput, ...]
    output_fields: tuple[JoinedFieldOutput, ...]


@dataclass(frozen=True)
class SourceScalarInputOutput(ProviderOutput):
    input_id: str
    source_binding_id: str


@dataclass(frozen=True)
class ComputeInputTokenOutput(ProviderOutput):
    input_id: str


@dataclass(frozen=True)
class ComputeOperatorTokenOutput(ProviderOutput):
    operator: str


ComputeExpressionTokenOutput = ComputeInputTokenOutput | ComputeOperatorTokenOutput


def parse_compute_expression_token(
    value: ProviderObject,
) -> ComputeExpressionTokenOutput:
    if value.has_field("input_id"):
        return value.parse_as(ComputeInputTokenOutput)
    return value.parse_as(ComputeOperatorTokenOutput)


@dataclass(frozen=True)
class ScalarOutputOutput(ProviderOutput):
    scalar_id: str
    label: Optional[str] = None


@dataclass(frozen=True)
class ComputedScalarAnswerOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    pattern: str
    scalar_inputs: tuple[SourceScalarInputOutput, ...]
    expression: tuple[ProviderObject, ...]
    output: ScalarOutputOutput


PatternAnswerOutput = (
    ListRowsAnswerOutput
    | GroupedRowsAnswerOutput
    | DirectFieldValueAnswerOutput
    | AggregateScalarAnswerOutput
    | GroupedAggregateAnswerOutput
    | SetDifferenceAnswerOutput
    | JoinedRowsAnswerOutput
    | ComputedScalarAnswerOutput
)


def parse_pattern_answer(value: ProviderObject) -> PatternAnswerOutput:
    pattern = value.discriminator("pattern")
    if pattern == "list_rows":
        return value.parse_as(ListRowsAnswerOutput)
    if pattern == "grouped_rows":
        return value.parse_as(GroupedRowsAnswerOutput)
    if pattern == "direct_field_value":
        return value.parse_as(DirectFieldValueAnswerOutput)
    if pattern == "aggregate_scalar":
        return value.parse_as(AggregateScalarAnswerOutput)
    if pattern == "aggregate_by_group":
        return value.parse_as(GroupedAggregateAnswerOutput)
    if pattern == "set_difference":
        return value.parse_as(SetDifferenceAnswerOutput)
    if pattern == "joined_rows":
        return value.parse_as(JoinedRowsAnswerOutput)
    if pattern == "computed_scalar":
        return value.parse_as(ComputedScalarAnswerOutput)
    raise ValueError(f"unsupported fact plan pattern: {pattern}")


@dataclass(frozen=True)
class FactPlanOutput(ProviderOutput):
    outcome: ProviderObject


@dataclass(frozen=True)
class FactPlanAnswerOutput(ProviderOutput):
    kind: str
    answers: tuple[ProviderObject, ...]


@dataclass(frozen=True)
class BlockedFactFieldOutput(ProviderOutput):
    read_id: str
    field_id: str


@dataclass(frozen=True)
class BlockedFactOutput(ProviderOutput):
    requested_fact_id: str
    basis: str
    evidence_refs: tuple[str, ...]
    reviewed_read_ids: Optional[tuple[str, ...]] = None
    nearest_fields: Optional[tuple[BlockedFactFieldOutput, ...]] = None
    explanation: Optional[str] = None


@dataclass(frozen=True)
class PlanImpossibleOutput(ProviderOutput):
    kind: str
    blocked_facts: tuple[BlockedFactOutput, ...]


@dataclass(frozen=True)
class MissingCatalogRequiredInputOutput(ProviderOutput):
    kind: str
    id: str
    requested_fact_id: str
    required_catalog_input_id: str


@dataclass(frozen=True)
class MissingCatalogChoiceInputOutput(ProviderOutput):
    kind: str
    id: str
    requested_fact_id: str
    required_catalog_choice_input_id: str


@dataclass(frozen=True)
class PlanClarificationOutput(ProviderOutput):
    kind: str
    missing_catalog_inputs: tuple[ProviderObject, ...]
