"""Typed provider-output contracts for question interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fervis.lookup.provider_contract import ProviderObject, ProviderOutput
from fervis.types.enums import StrEnum


class QuestionInputOwnerKind(StrEnum):
    GROUP_KEY = "GROUP_KEY"
    POPULATION_TESTS = "POPULATION_TESTS"
    RESULT_LIMIT = "RESULT_LIMIT"


@dataclass(frozen=True)
class QuestionContractDecisionOutput(ProviderOutput):
    decision_basis: str
    outcome: ProviderObject


@dataclass(frozen=True)
class QuestionInputItemInventoryCheckOutput(ProviderOutput):
    why_this_is_an_input: str


@dataclass(frozen=True)
class QuestionInputInventoryCheckOutput(ProviderOutput):
    all_input_like_phrases_declared: bool


@dataclass(frozen=True)
class LiteralTextInputOutput(ProviderOutput):
    input_ref: str
    source: str
    value_source_text: str
    operand_text: str
    role: str
    inventory_check: QuestionInputItemInventoryCheckOutput
    kind: str
    field_label_text: Optional[str] = None
    value_meaning_hint: Optional[str] = None
    occurrence: Optional[int] = None
    resolved_input_ref: Optional[str] = None


@dataclass(frozen=True)
class RowSetReferenceInputOutput(ProviderOutput):
    input_ref: str
    source: str
    reference_text: str
    occurrence: int
    resolved_input_ref: str
    inventory_check: QuestionInputItemInventoryCheckOutput
    kind: str


@dataclass(frozen=True)
class AnswerOutputOutput(ProviderOutput):
    description: str
    role: str


@dataclass(frozen=True)
class AnswerPopulationMembershipTestOutput(ProviderOutput):
    test_id: str
    kind: str
    polarity: str
    test_question: str


@dataclass(frozen=True)
class AnswerPopulationOutput(ProviderOutput):
    population_label: str
    counted_unit: str
    membership_tests: tuple[AnswerPopulationMembershipTestOutput, ...]


@dataclass(frozen=True)
class AnswerSubjectInstanceInterpretationOutput(ProviderOutput):
    kind: str


@dataclass(frozen=True)
class AnswerSubjectOutput(ProviderOutput):
    subject_text: str
    instance_interpretation: AnswerSubjectInstanceInterpretationOutput


@dataclass(frozen=True)
class GroupKeyOutput(ProviderOutput):
    description: str
    domain: str


@dataclass(frozen=True)
class AnswerExpressionOutput(ProviderOutput):
    family: str
    group_key: Optional[GroupKeyOutput] = None


@dataclass(frozen=True)
class GroupKeyQuestionInputUseOutput(ProviderOutput):
    input_ref: str
    owner_kind: str


@dataclass(frozen=True)
class PopulationTestsQuestionInputUseOutput(ProviderOutput):
    input_ref: str
    owner_kind: str
    membership_test_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResultLimitQuestionInputUseOutput(ProviderOutput):
    input_ref: str
    owner_kind: str


@dataclass(frozen=True)
class AnswerRequestOutput(ProviderOutput):
    answer_fact: str
    answer_expression: AnswerExpressionOutput
    answer_subject: AnswerSubjectOutput
    answer_population: AnswerPopulationOutput
    answer_outputs: tuple[AnswerOutputOutput, ...]
    question_input_uses: tuple[ProviderObject, ...]


@dataclass(frozen=True)
class QuestionContractOutput(ProviderOutput):
    kind: str
    answer_requests_count: int
    question_inputs: tuple[ProviderObject, ...]
    answer_requests: tuple[AnswerRequestOutput, ...]
    question_input_inventory_check: QuestionInputInventoryCheckOutput


@dataclass(frozen=True)
class UnresolvedPriorTurnReferenceOutput(ProviderOutput):
    source_text: str
    target_label: str
    why_question_is_incomplete: str


@dataclass(frozen=True)
class UnresolvedPriorTurnReferencesOutput(ProviderOutput):
    kind: str
    references: tuple[UnresolvedPriorTurnReferenceOutput, ...]


@dataclass(frozen=True)
class MissingRequestedFactOutput(ProviderOutput):
    kind: str
    source_text: str
    why_question_is_incomplete: str
