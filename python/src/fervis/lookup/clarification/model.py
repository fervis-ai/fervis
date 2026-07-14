"""Canonical lookup clarification model."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.canonical_data import EntityKeyValue
from fervis.types.enums import StrEnum


class ClarificationNeed(StrEnum):
    TARGET_REFERENCE = "target_reference"
    ANSWER_METRIC = "answer_metric"
    COMPARISON_BASELINE = "comparison_baseline"
    CATALOG_INPUT = "catalog_input"
    QUESTION_INTERPRETATION = "question_interpretation"


class ClarificationReason(StrEnum):
    UNRESOLVED_REFERENCE = "unresolved_reference"
    MULTIPLE_MATCHING_ENTITIES = "multiple_matching_entities"
    UNSUPPORTED_REFERENCE = "unsupported_reference"
    MISSING_ANSWER_METRIC = "missing_answer_metric"
    MISSING_COMPARISON_BASELINE = "missing_comparison_baseline"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    CATALOG_REQUIRES_CHOICE = "catalog_requires_choice"
    AMBIGUOUS_INTERPRETATION = "ambiguous_interpretation"


class ClarificationSubjectKind(StrEnum):
    QUESTION_INPUT = "question_input"
    CATALOG_INPUT = "catalog_input"
    CATALOG_CHOICE = "catalog_choice"
    METRIC_PHRASE = "metric_phrase"
    COMPARISON_PHRASE = "comparison_phrase"
    INTERPRETATION = "interpretation"


class ClarificationEvidenceKind(StrEnum):
    KNOWN_INPUT = "known_input"
    RESOLVER_READ = "resolver_read"
    GROUNDING = "grounding"
    QUESTION_CONTRACT = "question_contract"
    CATALOG_INPUT = "catalog_input"
    CANDIDATE = "candidate"
    PROOF_REF = "proof_ref"


class ClarificationOwner(StrEnum):
    CONVERSATION_RESOLUTION = "conversation_resolution"
    QUESTION_CONTRACT = "question_contract"
    GROUNDING = "grounding"
    SOURCE_BINDING = "source_binding"
    FACT_PLANNING = "fact_planning"


@dataclass(frozen=True)
class ClarificationOption:
    id: str
    label: str = ""
    value: str = ""
    key: EntityKeyValue | None = None
    matched_label: str = ""
    matched_field: str = ""
    matched_value: str = ""
    resolver_read_id: str = ""
    resolver_label: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("clarification option requires id")


@dataclass(frozen=True)
class ClarificationSubject:
    kind: ClarificationSubjectKind
    id: str
    label: str = ""
    source_text: str = ""
    options: tuple[ClarificationOption, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ClarificationSubjectKind):
            raise ValueError("clarification subject requires structured kind")
        if not self.id:
            raise ValueError("clarification subject requires id")


@dataclass(frozen=True)
class ClarificationEvidence:
    kind: ClarificationEvidenceKind
    id: str
    read_id: str = ""
    endpoint_name: str = ""
    field_id: str = ""
    identity_field: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ClarificationEvidenceKind):
            raise ValueError("clarification evidence requires structured kind")
        if not self.id:
            raise ValueError("clarification evidence requires id")


@dataclass(frozen=True)
class ConversationInterpretationEvidence:
    source_id: str
    exact_source_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id or not self.exact_source_texts:
            raise ValueError("conversation interpretation evidence requires source")
        if any(not text.strip() for text in self.exact_source_texts):
            raise ValueError("conversation interpretation evidence requires exact text")


@dataclass(frozen=True)
class ConversationInterpretationCandidate:
    id: str
    contextualized_question: str
    source_evidence: tuple[ConversationInterpretationEvidence, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.contextualized_question.strip():
            raise ValueError("conversation interpretation candidate requires identity")
        if not self.source_evidence:
            raise ValueError(
                "conversation interpretation candidate requires source evidence"
            )


@dataclass(frozen=True)
class ConversationResolutionContinuation:
    candidates: tuple[ConversationInterpretationCandidate, ...] = ()
    accepts_free_text: bool = False

    def __post_init__(self) -> None:
        if bool(self.candidates) == self.accepts_free_text:
            raise ValueError(
                "conversation continuation requires candidates or a free-text slot"
            )


@dataclass(frozen=True)
class QuestionContractContinuation:
    missing_item_id: str
    expected_value_kind: str

    def __post_init__(self) -> None:
        if not self.missing_item_id or not self.expected_value_kind:
            raise ValueError("question-contract continuation requires a typed slot")


@dataclass(frozen=True)
class GroundingContinuation:
    known_input_id: str
    accepts_free_text: bool = False

    def __post_init__(self) -> None:
        if not self.known_input_id:
            raise ValueError("grounding continuation requires known_input_id")


@dataclass(frozen=True)
class CatalogInputTarget:
    row_source_id: str
    param_id: str
    param_ref: str
    value_type: str
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            (self.row_source_id, self.param_id, self.param_ref, self.value_type)
        ):
            raise ValueError("catalog continuation requires a complete input target")


@dataclass(frozen=True)
class SourceBindingCatalogInputContinuation:
    requested_fact_id: str
    target: CatalogInputTarget

    def __post_init__(self) -> None:
        if not self.requested_fact_id:
            raise ValueError("source-binding continuation requires requested fact")


@dataclass(frozen=True)
class FactPlanningCatalogInputContinuation:
    requested_fact_id: str
    planning_requirement_id: str
    target: CatalogInputTarget

    def __post_init__(self) -> None:
        if not self.requested_fact_id or not self.planning_requirement_id:
            raise ValueError("fact-planning continuation requires planning requirement")


ClarificationContinuationSpec = (
    ConversationResolutionContinuation
    | QuestionContractContinuation
    | GroundingContinuation
    | SourceBindingCatalogInputContinuation
    | FactPlanningCatalogInputContinuation
)


@dataclass(frozen=True)
class Clarification:
    id: str
    requested_fact_id: str
    need: ClarificationNeed
    reason: ClarificationReason
    subjects: tuple[ClarificationSubject, ...]
    owner: ClarificationOwner
    continuation: ClarificationContinuationSpec
    evidence: tuple[ClarificationEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("clarification requires id")
        if not self.requested_fact_id:
            raise ValueError("clarification requires requested fact")
        if not isinstance(self.need, ClarificationNeed):
            raise ValueError("clarification requires structured need")
        if not isinstance(self.reason, ClarificationReason):
            raise ValueError("clarification requires structured reason")
        if not self.subjects:
            raise ValueError("clarification requires at least one subject")
        if not isinstance(self.owner, ClarificationOwner):
            raise ValueError("clarification requires structured owner")
        expected_owner = _continuation_owner(self.continuation)
        if self.owner is not expected_owner:
            raise ValueError("clarification owner and continuation must match")
        _validate_continuation_authority(self)


def _continuation_owner(
    continuation: ClarificationContinuationSpec,
) -> ClarificationOwner:
    if isinstance(continuation, ConversationResolutionContinuation):
        return ClarificationOwner.CONVERSATION_RESOLUTION
    if isinstance(continuation, QuestionContractContinuation):
        return ClarificationOwner.QUESTION_CONTRACT
    if isinstance(continuation, GroundingContinuation):
        return ClarificationOwner.GROUNDING
    if isinstance(continuation, SourceBindingCatalogInputContinuation):
        return ClarificationOwner.SOURCE_BINDING
    if isinstance(continuation, FactPlanningCatalogInputContinuation):
        return ClarificationOwner.FACT_PLANNING
    raise TypeError("unsupported clarification continuation")


def _validate_continuation_authority(clarification: Clarification) -> None:
    continuation = clarification.continuation
    if len(clarification.subjects) != 1:
        raise ValueError("clarification continuation requires exactly one subject")
    subject = clarification.subjects[0]
    option_ids = tuple(option.id for option in subject.options)

    if isinstance(continuation, ConversationResolutionContinuation):
        candidate_ids = tuple(candidate.id for candidate in continuation.candidates)
        if option_ids != candidate_ids:
            raise ValueError("conversation options must match continuation candidates")
        return
    if isinstance(continuation, QuestionContractContinuation):
        if subject.id != continuation.missing_item_id:
            raise ValueError("question-contract subject must match its typed slot")
        return
    if isinstance(continuation, GroundingContinuation):
        if subject.id != continuation.known_input_id:
            raise ValueError("grounding subject must match its known input")
        if bool(option_ids) == continuation.accepts_free_text:
            raise ValueError(
                "grounding continuation requires canonical options or a free-text slot"
            )
        if any(option.key is None for option in subject.options):
            raise ValueError("grounding options require complete canonical identity")
        return

    target = continuation.target
    if continuation.requested_fact_id != clarification.requested_fact_id:
        raise ValueError("catalog continuation must match its requested fact")
    if subject.id != f"{target.row_source_id}.{target.param_id}":
        raise ValueError("catalog subject must match its continuation target")
    if option_ids != target.choices:
        raise ValueError("catalog options must match continuation choices")


@dataclass(frozen=True)
class ClarificationResponseSource:
    response_id: str
    clarification_id: str
    exact_user_text: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.response_id.strip(),
                self.clarification_id.strip(),
                self.exact_user_text.strip(),
            )
        ):
            raise ValueError("clarification response source requires exact attribution")


@dataclass(frozen=True)
class ConversationResolutionResponse:
    source: ClarificationResponseSource
    candidate: ConversationInterpretationCandidate | None = None
    annotation: ClarificationAnnotation | None = None


@dataclass(frozen=True)
class ClarificationAnnotation:
    suspended_question_text: str
    clarification_question_text: str

    def __post_init__(self) -> None:
        if not self.suspended_question_text.strip():
            raise ValueError("clarification annotation requires suspended question")
        if not self.clarification_question_text.strip():
            raise ValueError("clarification annotation requires clarification question")


@dataclass(frozen=True)
class QuestionContractResponse:
    source: ClarificationResponseSource
    missing_item_id: str
    expected_value_kind: str


@dataclass(frozen=True)
class GroundingIdentityResponse:
    response_id: str
    clarification_id: str
    requested_fact_id: str
    known_input_id: str
    option: ClarificationOption


@dataclass(frozen=True)
class SourceBindingCatalogInputResponse:
    response_id: str
    clarification_id: str
    requested_fact_id: str
    target: CatalogInputTarget
    value: str


@dataclass(frozen=True)
class FactPlanningCatalogInputResponse:
    response_id: str
    clarification_id: str
    requested_fact_id: str
    planning_requirement_id: str
    target: CatalogInputTarget
    value: str


ClarificationOwnerResponse = (
    ConversationResolutionResponse
    | QuestionContractResponse
    | GroundingIdentityResponse
    | SourceBindingCatalogInputResponse
    | FactPlanningCatalogInputResponse
)
