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
class SuppliedValueDenotationOutput(ProviderOutput):
    basis: str
    kind: str
    instance_kind: str | None = None


@dataclass(frozen=True)
class SuppliedValueOutput(ProviderOutput):
    meaning: str
    denotation: SuppliedValueDenotationOutput
    value: SuppliedInputValueOutput


@dataclass(frozen=True)
class SourceOriginOutput(ProviderOutput):
    source: str
    meaning: str
    resolved_input_ref: str | None


@dataclass(frozen=True)
class SetTermOutput(ProviderOutput):
    id: str
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
    value_type: ProviderObject
    origin: SourceOriginOutput


@dataclass(frozen=True)
class AnswerRequestFrameOutput(ProviderOutput):
    result_kind: str
    qualifying_row_kind: MeaningOriginOutput
    grouping_meanings: tuple[MeaningOriginOutput, ...]
    return_request_basis: str
    returned_result: ProviderObject
    answer_values: tuple[ProviderObject, ...]
    returned_value_refs: tuple[str, ...]
    ordering_value_refs: tuple[str, ...]
    selection: ProviderObject
    universal_shape: str
    returned_candidate_identity: MeaningOriginOutput | None = None


@dataclass(frozen=True)
class NonKeyFrameValueOutput(ProviderOutput):
    value_ref: str
    meaning: str
    origin: FrameOriginOutput


@dataclass(frozen=True)
class ReturnedResultOutput(ProviderOutput):
    kind: str


@dataclass(frozen=True)
class CompleteSemanticQuestionFrameOutput(ProviderOutput):
    kind: str
    answer_requests: tuple[AnswerRequestFrameOutput, ...]
    supplied_values: tuple[SuppliedValueOutput, ...]


@dataclass(frozen=True)
class CandidateSetOutput(ProviderOutput):
    instance_interpretation: str


@dataclass(frozen=True)
class RequestedOutputOutput(ProviderOutput):
    expression: ProviderObject


@dataclass(frozen=True)
class CandidateIdentityGroupingOutput(ProviderOutput):
    id: str
    grouping_basis: str
    kind: str


@dataclass(frozen=True)
class GroupingSetOutput(ProviderOutput):
    id: str


@dataclass(frozen=True)
class GroupingAssociationOutput(ProviderOutput):
    id: str
    origin: SourceOriginOutput


@dataclass(frozen=True)
class RelatedIdentityGroupingOutput(ProviderOutput):
    id: str
    grouping_basis: str
    kind: str
    identified_set: GroupingSetOutput
    association: GroupingAssociationOutput


@dataclass(frozen=True)
class ValueGroupingOutput(ProviderOutput):
    id: str
    grouping_basis: str
    kind: str
    expression: ProviderObject


@dataclass(frozen=True)
class OrderingOutput(ProviderOutput):
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
    other_sets: tuple[SetTermOutput, ...]
    other_associations: tuple[AssociationTermOutput, ...]
    qualification: ProviderObject | None
    ordering: tuple[OrderingOutput, ...]
    selection: ProviderObject | None
    distinct_by: tuple[ProviderObject, ...]
    outputs: tuple[RequestedOutputOutput, ...]


@dataclass(frozen=True)
class CompleteSemanticQuestionContractOutput(ProviderOutput):
    kind: str
    answer_requests: tuple[AnswerRequestOutput, ...]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
