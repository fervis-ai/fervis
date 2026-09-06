"""Provider-authored DTOs for the semantic relational Question Contract."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class SemanticQuestionContractDecisionOutput(ProviderOutput):
    decision_basis: str
    outcome: ProviderObject


@dataclass(frozen=True)
class SemanticQuestionFrameDecisionOutput(ProviderOutput):
    decision_basis: str
    outcome: ProviderObject


@dataclass(frozen=True)
class FrameOriginOutput(ProviderOutput):
    kind: str
    resolved_input_ref: str | None = None


@dataclass(frozen=True)
class MeaningOriginOutput(ProviderOutput):
    meaning: str
    origin: FrameOriginOutput


@dataclass(frozen=True)
class SuppliedInputValueOutput(ProviderOutput):
    operands: tuple[str, ...]
    value_type: ProviderObject
    origin: FrameOriginOutput


@dataclass(frozen=True)
class NonEntityValueOutput(ProviderOutput):
    value: SuppliedInputValueOutput


@dataclass(frozen=True)
class SingleIdentityValueOutput(ProviderOutput):
    kind: str
    identity_value: str
    origin: FrameOriginOutput


@dataclass(frozen=True)
class IdentityAlternativesValueOutput(ProviderOutput):
    kind: str
    identity_values: tuple[str, ...]
    origin: FrameOriginOutput


@dataclass(frozen=True)
class EntityReferenceOutput(ProviderOutput):
    instance_kind: str
    value: ProviderObject


@dataclass(frozen=True)
class SuppliedEntityReferenceOutput(ProviderOutput):
    meaning: str
    denotation_basis: str
    entity_reference: EntityReferenceOutput


@dataclass(frozen=True)
class NonEntityLedgerValueOutput(ProviderOutput):
    operands: tuple[str, ...]
    origin: FrameOriginOutput


@dataclass(frozen=True)
class NonEntityOperandOutput(ProviderOutput):
    kind: str
    value: NonEntityLedgerValueOutput


@dataclass(frozen=True)
class DurationOperandOutput(ProviderOutput):
    kind: str
    unit: str
    value: NonEntityLedgerValueOutput


@dataclass(frozen=True)
class SuppliedNonEntityValueOutput(ProviderOutput):
    meaning: str
    denotation_basis: str
    non_entity_value: ProviderObject


@dataclass(frozen=True)
class SelectionLimitOutput(ProviderOutput):
    answer_request_number: int
    meaning: str
    denotation_basis: str
    non_entity_value: NonEntityValueOutput


@dataclass(frozen=True)
class SuppliedValuesLedgerOutput(ProviderOutput):
    operands: tuple[ProviderObject, ...]
    selection_limits: tuple[SelectionLimitOutput, ...]


@dataclass(frozen=True)
class QuestionInputInventoryCheckOutput(ProviderOutput):
    all_input_like_phrases_declared: bool


@dataclass(frozen=True)
class SourceOriginOutput(ProviderOutput):
    source: str
    meaning: str
    resolved_input_ref: str | None


@dataclass(frozen=True)
class SetTermOutput(ProviderOutput):
    id: str
    instance_kind: str
    origin: SourceOriginOutput


@dataclass(frozen=True)
class AssociationTermOutput(ProviderOutput):
    id: str
    from_set_ref: str
    to_set_ref: str
    origin: SourceOriginOutput


@dataclass(frozen=True)
class FactTermOutput(ProviderOutput):
    kind: str
    observed_for_ref: str
    origin: SourceOriginOutput
    identified_set_ref: str | None = None


@dataclass(frozen=True)
class GroupingMeaningOutput(ProviderOutput):
    group_ref: str
    grouping_basis: str
    meaning: str
    origin: FrameOriginOutput
    grouping_kind: str
    grouping_value: ProviderObject | None = None


@dataclass(frozen=True)
class FrameRowSourceOutput(ProviderOutput):
    instance_kind: str
    origin: FrameOriginOutput


@dataclass(frozen=True)
class AnswerRequestFrameOutput(ProviderOutput):
    return_request_basis: str
    relational_shape_basis: str
    request: ProviderObject


@dataclass(frozen=True)
class QuestionFrameRequestOutput(ProviderOutput):
    relational_shape: str
    result_grain_basis: str
    result: ProviderObject


@dataclass(frozen=True)
class FrameResultOrderOutput(ProviderOutput):
    ordering_request_basis: str
    ordering: ProviderObject
    selection: ProviderObject


@dataclass(frozen=True)
class OrderedByOutput(ProviderOutput):
    kind: str
    values: tuple[ProviderObject, ...]


@dataclass(frozen=True)
class ReturnedMeaningOutput(ProviderOutput):
    meaning: str
    origin: FrameOriginOutput
    meaning_ref: str


@dataclass(frozen=True)
class RequestedValueOrderingOutput(ProviderOutput):
    ownership_basis: str
    kind: str
    value_ref: str


@dataclass(frozen=True)
class GroupOrderingReferenceOutput(ProviderOutput):
    ownership_basis: str
    kind: str
    group_ref: str


@dataclass(frozen=True)
class NonTemporalGroupingValueOutput(ProviderOutput):
    kind: str


@dataclass(frozen=True)
class TemporalBucketGroupingValueOutput(ProviderOutput):
    kind: str
    grain: str


@dataclass(frozen=True)
class UnreturnedOrderingMeaningOutput(ProviderOutput):
    ownership_basis: str
    kind: str
    meaning: str
    origin: FrameOriginOutput


@dataclass(frozen=True)
class ReturnedProjectionValueOutput(ProviderOutput):
    value_ref: str
    value_kind_basis: str
    value_kind: str
    meaning: str
    origin: FrameOriginOutput


@dataclass(frozen=True)
class CandidateProjectionOutput(ProviderOutput):
    projection_basis: str
    candidate_identity: str
    explicitly_requested_values: tuple[ReturnedProjectionValueOutput, ...]


@dataclass(frozen=True)
class GroupProjectionOutput(ProviderOutput):
    projection_basis: str
    returned_grouping_keys: str
    explicitly_requested_values: tuple[ReturnedProjectionValueOutput, ...]


@dataclass(frozen=True)
class CompleteSemanticQuestionFrameOutput(ProviderOutput):
    kind: str
    answer_requests: tuple[AnswerRequestFrameOutput, ...]
    supplied_values: SuppliedValuesLedgerOutput
    question_input_inventory_check: QuestionInputInventoryCheckOutput


@dataclass(frozen=True)
class CandidateSetOutput(ProviderOutput):
    instance_kind: str
    instance_interpretation: str


@dataclass(frozen=True)
class RequestedOutputOutput(ProviderOutput):
    expression: ProviderObject


@dataclass(frozen=True)
class RequestedValueOutputOutput(ProviderOutput):
    output_ref: str
    expression: ProviderObject


@dataclass(frozen=True)
class RequestedRelatedEntityOutputOutput(ProviderOutput):
    output_ref: str
    instance_kind: str


@dataclass(frozen=True)
class RoleRelationOutput(ProviderOutput):
    association: ProviderObject
    set: SetTermOutput
    related_sets: tuple[ProviderObject, ...]


@dataclass(frozen=True)
class SetGraphOutput(ProviderOutput):
    identity_input_relations: dict[str, RoleRelationOutput | None]
    requested_output_relations: dict[str, RoleRelationOutput]
    other_related_sets: tuple[ProviderObject, ...]


@dataclass(frozen=True)
class RequestedOutputsOutput(ProviderOutput):
    result_key_outputs: tuple[RequestedOutputOutput, ...]
    requested_value_outputs: tuple[ProviderObject, ...]


@dataclass(frozen=True)
class CandidateIdentityGroupingOutput(ProviderOutput):
    id: str
    grouping_basis: str
    kind: str


@dataclass(frozen=True)
class GroupingAssociationOutput(ProviderOutput):
    id: str
    origin: SourceOriginOutput


@dataclass(frozen=True)
class RelatedIdentityGroupingOutput(ProviderOutput):
    id: str
    grouping_basis: str
    kind: str
    set_ref: str


@dataclass(frozen=True)
class ValueGroupingOutput(ProviderOutput):
    id: str
    grouping_basis: str
    kind: str
    expression: ProviderObject


@dataclass(frozen=True)
class OrderingOutput(ProviderOutput):
    ordering_basis: str
    expression: ProviderObject
    direction: str


@dataclass(frozen=True)
class SelectionOutput(ProviderOutput):
    kind: str
    limit: ProviderObject | None = None


@dataclass(frozen=True)
class AnswerRequestOutput(ProviderOutput):
    requested_fact_ref: str
    origin: SourceOriginOutput
    candidate_set: CandidateSetOutput
    grouping: tuple[ProviderObject, ...]
    set_graph: ProviderObject
    qualification: ProviderObject | None
    ordering: tuple[OrderingOutput, ...]
    selection: ProviderObject | None
    distinct_by: tuple[ProviderObject, ...]
    outputs: RequestedOutputsOutput


@dataclass(frozen=True)
class CompleteSemanticQuestionContractOutput(ProviderOutput):
    kind: str
    answer_requests: tuple[AnswerRequestOutput, ...]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
