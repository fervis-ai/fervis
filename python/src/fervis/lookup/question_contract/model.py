"""Catalog-blind requested-fact contract for Lookup questions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fervis.types.enums import StrEnum
import re
from typing import Any, TypeAlias

from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
)
from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)
from fervis.lookup.question_contract._text_spans import contains_copied_span
from fervis.lookup.question_inputs import (
    KnownInputKind,
    LiteralInputRole,
)
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lookup.clarification.model import ClarificationResponseSource


class KnownInputSource(StrEnum):
    QUESTION_CONTEXT = "question_context"
    CONVERSATION_RESOLUTION = "conversation_resolution"


class AnswerSubjectInstanceInterpretationKind(StrEnum):
    NORMAL_BUSINESS_INSTANCE = "NORMAL_BUSINESS_INSTANCE"
    RAW_DATA_RECORD = "RAW_DATA_RECORD"


class AnswerPopulationMembershipTestKind(StrEnum):
    SUBJECT_IDENTITY = "SUBJECT_IDENTITY"
    EXPLICIT_USER_CONSTRAINT = "EXPLICIT_USER_CONSTRAINT"
    NORMAL_INSTANCE_GUARD = "NORMAL_INSTANCE_GUARD"
    RAW_RECORD_GUARD = "RAW_RECORD_GUARD"


class AnswerPopulationMembershipTestPolarity(StrEnum):
    MUST_PASS = "MUST_PASS"
    MUST_FAIL = "MUST_FAIL"


class NormalInstanceProfileId(StrEnum):
    ORDINARY_BUSINESS_INSTANCE_V1 = "ORDINARY_BUSINESS_INSTANCE_V1"


class NormalInstanceExcludedStateRole(StrEnum):
    NOT_REALIZED = "NOT_REALIZED"
    CANCELED_OR_VOIDED = "CANCELED_OR_VOIDED"
    FAILED_OR_REJECTED_BEFORE_EFFECT = "FAILED_OR_REJECTED_BEFORE_EFFECT"
    REVERSED_OR_CORRECTION_ARTIFACT = "REVERSED_OR_CORRECTION_ARTIFACT"
    TEST_PLACEHOLDER_OR_DEMO = "TEST_PLACEHOLDER_OR_DEMO"
    SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT = (
        "SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT"
    )


class NormalInstanceExplicitOverrideReason(StrEnum):
    USER_EXPLICITLY_REQUESTED_STATE = "USER_EXPLICITLY_REQUESTED_STATE"
    USER_EXPLICITLY_REQUESTED_RAW_RECORDS = "USER_EXPLICITLY_REQUESTED_RAW_RECORDS"
    USER_EXPLICITLY_REQUESTED_ALL_RECORDS = "USER_EXPLICITLY_REQUESTED_ALL_RECORDS"
    USER_EXPLICITLY_REQUESTED_NON_NORMAL_POPULATION = (
        "USER_EXPLICITLY_REQUESTED_NON_NORMAL_POPULATION"
    )


class RequestedFactAnswerExpressionFamily(StrEnum):
    LIST_ROWS = "list_rows"
    SCALAR_VALUE = "scalar_value"
    SCALAR_AGGREGATE = "scalar_aggregate"
    GROUPED_AGGREGATE = "grouped_aggregate"
    RANKED_SELECTION = "ranked_selection"
    COMPUTED_SCALAR = "computed_scalar"
    SET_DIFFERENCE = "set_difference"
    COVERAGE_CHECK = "coverage_check"
    EXISTENCE_CHECK = "existence_check"
    COMPARISON_CHECK = "comparison_check"


class ResultSelectionKind(StrEnum):
    ALL_RESULTS = "all_results"
    LIMITED_RESULTS = "limited_results"


class IncompleteFactualRequestKind(StrEnum):
    UNRESOLVED_PRIOR_TURN_REFERENCE = "unresolved_prior_turn_reference"
    MISSING_REQUESTED_FACT = "missing_requested_fact"


class GroupKeyDomainKind(StrEnum):
    SPECIFIED_QUESTION_INPUTS = "SPECIFIED_QUESTION_INPUTS"
    SOURCE_RESULT_VALUES = "SOURCE_RESULT_VALUES"


NORMAL_INSTANCE_EXPLICIT_USER_OVERRIDE_POLICY = (
    "Do not exclude a state role when the user explicitly asks for that state, "
    "raw records, all records, or a non-normal population."
)


@dataclass(frozen=True)
class NormalInstanceExcludedStateRoleDefinition:
    role: NormalInstanceExcludedStateRole
    definition: str

    def to_answer_request_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "definition": self.definition,
        }


NORMAL_INSTANCE_EXCLUDED_STATE_ROLES: tuple[
    NormalInstanceExcludedStateRoleDefinition, ...
] = (
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.NOT_REALIZED,
        definition=(
            "A tentative, draft, planned-only, or provisional representation "
            "that has not become an effective instance of the requested subject."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.CANCELED_OR_VOIDED,
        definition=(
            "The entity was canceled, voided, aborted, or otherwise explicitly "
            "made not count as an ordinary business instance."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.FAILED_OR_REJECTED_BEFORE_EFFECT,
        definition=(
            "The entity failed, was rejected, was declined, or did not become "
            "effective in normal business operations."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.REVERSED_OR_CORRECTION_ARTIFACT,
        definition=(
            "The entity is a reversal, refund, correction, adjustment, or "
            "counter-entry rather than the ordinary business instance itself."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.TEST_PLACEHOLDER_OR_DEMO,
        definition=(
            "The entity is test, demo, sample, placeholder, sandbox, or seed "
            "data rather than an ordinary business instance."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=(
            NormalInstanceExcludedStateRole.SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT
        ),
        definition=(
            "The entity is superseded, deleted, archived as non-current, or "
            "kept only as a historical artifact rather than a current ordinary "
            "business instance."
        ),
    ),
)


@dataclass(frozen=True)
class NormalInstanceProfile:
    subject_text: str
    profile_id: NormalInstanceProfileId = (
        NormalInstanceProfileId.ORDINARY_BUSINESS_INSTANCE_V1
    )
    excluded_state_roles: tuple[NormalInstanceExcludedStateRoleDefinition, ...] = (
        NORMAL_INSTANCE_EXCLUDED_STATE_ROLES
    )
    explicit_user_override_policy: str = NORMAL_INSTANCE_EXPLICIT_USER_OVERRIDE_POLICY

    def __post_init__(self) -> None:
        if not self.subject_text.strip():
            raise ValueError("normal instance profile requires subject text")
        if not self.excluded_state_roles:
            raise ValueError("normal instance profile requires excluded state roles")

    def to_answer_request_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id.value,
            "subject_text": self.subject_text,
            "excluded_state_roles": [
                role.to_answer_request_dict() for role in self.excluded_state_roles
            ],
            "explicit_user_override_policy": self.explicit_user_override_policy,
        }


def normal_instance_profile(subject_text: str) -> NormalInstanceProfile:
    return NormalInstanceProfile(subject_text=subject_text)


def normal_instance_guard_question(subject_text: str) -> str:
    role_values = ", ".join(
        role.role.value for role in NORMAL_INSTANCE_EXCLUDED_STATE_ROLES
    )
    return (
        f"Is this an ordinary business instance of {subject_text} under "
        f"{NormalInstanceProfileId.ORDINARY_BUSINESS_INSTANCE_V1.value}, with none "
        f"of these excluded state roles applying: {role_values}?"
    )


@dataclass(frozen=True)
class RequestedFactGroupKey:
    description: str
    domain: GroupKeyDomainKind
    question_input_refs: tuple[str, ...] = ()
    id: str = "group_key"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("group key requires id")
        if not self.description.strip():
            raise ValueError("group key requires description")
        if not isinstance(self.domain, GroupKeyDomainKind):
            raise ValueError("group key requires structured domain")
        refs = tuple(
            str(item).strip() for item in self.question_input_refs if str(item).strip()
        )
        if refs != self.question_input_refs:
            object.__setattr__(self, "question_input_refs", refs)
        if self.domain == GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS and not refs:
            raise ValueError("specified group key requires question inputs")
        if self.domain == GroupKeyDomainKind.SOURCE_RESULT_VALUES and refs:
            raise ValueError("source-result group key cannot carry input refs")

    def to_model_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "description": self.description,
            "domain": self.domain.value,
        }
        if self.question_input_refs:
            payload["question_input_refs"] = list(self.question_input_refs)
        return payload

    def to_answer_request_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "description": self.description,
            "domain": self.domain.value,
        }
        if self.question_input_refs:
            payload["question_input_refs"] = list(self.question_input_refs)
        return payload


@dataclass(frozen=True)
class RequestedFactAnswerOutput:
    id: str
    role: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("answer output requires id")
        if not self.description.strip():
            object.__setattr__(self, "description", self.id)
        role = self.role.strip()
        if role not in ANSWER_OUTPUT_SUPPORT_ROLE_VALUES:
            raise ValueError("answer output role is invalid")
        if role != self.role:
            object.__setattr__(self, "role", role)

    def to_model_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "description": self.description,
        }
        payload["role"] = self.role
        return payload

    def to_answer_request_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "description": self.description,
        }
        payload["role"] = self.role
        return payload


@dataclass(frozen=True)
class RequestedFactAnswerExpression:
    family: RequestedFactAnswerExpressionFamily
    group_key: RequestedFactGroupKey | None = None
    selection_kind: ResultSelectionKind | None = None
    limit_input_ref: str = ""

    def __post_init__(self) -> None:
        if (
            self.family == RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE
            and self.group_key is None
        ):
            raise ValueError("grouped_aggregate requires group key")
        if (
            self.family != RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE
            and self.group_key is not None
        ):
            raise ValueError("group key requires grouped_aggregate answer expression")
        selects_rows = self.family in {
            RequestedFactAnswerExpressionFamily.LIST_ROWS,
            RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        }
        if selects_rows and self.selection_kind is None:
            raise ValueError("row answer expression requires result selection")
        if not selects_rows and self.selection_kind is not None:
            raise ValueError("result selection requires a row answer expression")
        if (
            self.family is RequestedFactAnswerExpressionFamily.LIST_ROWS
            and self.selection_kind is not ResultSelectionKind.ALL_RESULTS
        ):
            raise ValueError("list rows requires all results")
        if (
            self.family is RequestedFactAnswerExpressionFamily.RANKED_SELECTION
            and self.selection_kind is not ResultSelectionKind.LIMITED_RESULTS
        ):
            raise ValueError("ranked selection requires a limited result selection")
        if self.selection_kind is ResultSelectionKind.LIMITED_RESULTS:
            if (
                not self.limit_input_ref
                and self.family
                is not RequestedFactAnswerExpressionFamily.RANKED_SELECTION
            ):
                raise ValueError("limited result selection requires a limit input")
        elif self.limit_input_ref:
            raise ValueError("limit input requires limited result selection")

    def to_answer_request_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"family": self.family.value}
        if self.group_key is not None:
            payload["group_key"] = self.group_key.to_answer_request_dict()
        if self.selection_kind is not None:
            payload["selection_kind"] = self.selection_kind.value
        if self.limit_input_ref:
            payload["limit_input_ref"] = self.limit_input_ref
        return payload


@dataclass(frozen=True)
class RequestedFactLiteralInput:
    id: str
    source: KnownInputSource
    role: LiteralInputRole
    text: str = ""
    resolved_value_text: str = ""
    field_label_text: str = ""
    value_meaning_hint: str = ""
    resolved_input_ref: str = ""
    occurrence: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("known input requires id")
        if self.occurrence < 1:
            raise ValueError("known input occurrence must be positive")
        if not self.text.strip():
            raise ValueError("known input requires text")
        if not self.resolved_value_text.strip():
            raise ValueError("literal known input requires resolved value text")
        if self.role == LiteralInputRole.RESULT_LIMIT:
            if (
                not self.resolved_value_text.isdigit()
                or int(self.resolved_value_text) < 1
            ):
                raise ValueError(
                    "result_limit literal requires canonical positive integer digits"
                )
            object.__setattr__(
                self,
                "resolved_value_text",
                str(int(self.resolved_value_text)),
            )
        if self.source == KnownInputSource.CONVERSATION_RESOLUTION:
            if not self.resolved_input_ref.strip():
                raise ValueError(
                    "conversation-resolution literal requires resolved input ref"
                )
        elif self.resolved_input_ref:
            raise ValueError(
                "question-context literal must not include resolved input ref"
            )

    @property
    def kind(self) -> KnownInputKind:
        return KnownInputKind.LITERAL

    @property
    def is_reference_value(self) -> bool:
        return self.role == LiteralInputRole.REFERENCE_VALUE

    @property
    def is_time_value(self) -> bool:
        return self.role == LiteralInputRole.TIME_VALUE

    @property
    def is_result_limit(self) -> bool:
        return self.role == LiteralInputRole.RESULT_LIMIT

    def to_model_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "kind": KnownInputKind.LITERAL.value,
            "source": self.source.value,
            "text": self.text,
        }
        if self.resolved_input_ref:
            payload["resolved_input_ref"] = self.resolved_input_ref
        if self.occurrence != 1:
            payload["occurrence"] = self.occurrence
        payload["resolved_value_text"] = self.resolved_value_text
        if self.field_label_text:
            payload["field_label_text"] = self.field_label_text
        if self.value_meaning_hint:
            payload["value_meaning_hint"] = self.value_meaning_hint
        payload["role"] = self.role.value
        return payload


@dataclass(frozen=True)
class RequestedFactRowSetReferenceInput:
    id: str
    text: str
    resolved_input_ref: str
    occurrence: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("known input requires id")
        if not self.text.strip():
            raise ValueError("known input requires text")
        if self.occurrence < 1:
            raise ValueError("known input occurrence must be positive")
        if not self.resolved_input_ref.strip():
            raise ValueError("row set reference requires resolved input ref")

    @property
    def kind(self) -> KnownInputKind:
        return KnownInputKind.ROW_SET_REFERENCE

    @property
    def source(self) -> KnownInputSource:
        return KnownInputSource.CONVERSATION_RESOLUTION

    @property
    def is_reference_value(self) -> bool:
        return False

    @property
    def is_time_value(self) -> bool:
        return False

    @property
    def is_result_limit(self) -> bool:
        return False

    def to_model_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": KnownInputKind.ROW_SET_REFERENCE.value,
            "source": KnownInputSource.CONVERSATION_RESOLUTION.value,
            "text": self.text,
            "occurrence": self.occurrence,
            "resolved_input_ref": self.resolved_input_ref,
        }


RequestedFactKnownInput: TypeAlias = (
    RequestedFactLiteralInput | RequestedFactRowSetReferenceInput
)


@dataclass(frozen=True)
class RequestedFactAnswerSubjectInstanceInterpretation:
    kind: AnswerSubjectInstanceInterpretationKind

    def short_label(self, *, subject_text: str) -> str:
        if self.kind == AnswerSubjectInstanceInterpretationKind.RAW_DATA_RECORD:
            return f"raw data record meaning of {subject_text}"
        return f"ordinary business meaning of {subject_text}"

    @property
    def meaning_template(self) -> str:
        if self.kind == AnswerSubjectInstanceInterpretationKind.RAW_DATA_RECORD:
            return (
                "Answer over persisted data records for '{subject_text}', because the "
                "user explicitly asked for records, rows, logs, audit entries, raw "
                "data, database entries, or another data artifact."
            )
        return (
            "Answer over ordinary business instances of '{subject_text}' as they are "
            "normally understood in business operations and reporting. Do not assume "
            "this includes every persisted representation the host API may expose for "
            "'{subject_text}'. Explicit user wording may narrow or override this "
            "normal instance interpretation."
        )

    def to_answer_request_dict(self, *, subject_text: str) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "short_label": self.short_label(subject_text=subject_text),
            "meaning": self.meaning_template.format(subject_text=subject_text),
        }


@dataclass(frozen=True)
class RequestedFactAnswerSubject:
    subject_text: str
    instance_interpretation: RequestedFactAnswerSubjectInstanceInterpretation = (
        RequestedFactAnswerSubjectInstanceInterpretation(
            AnswerSubjectInstanceInterpretationKind.NORMAL_BUSINESS_INSTANCE
        )
    )

    def __post_init__(self) -> None:
        if not self.subject_text.strip():
            raise ValueError("answer subject requires subject text")

    def to_answer_request_dict(self) -> dict[str, object]:
        return {
            "subject_text": self.subject_text,
            "instance_interpretation": (
                self.instance_interpretation.to_answer_request_dict(
                    subject_text=self.subject_text
                )
            ),
        }


@dataclass(frozen=True)
class RequestedFactAnswerPopulationMembershipTest:
    id: str
    kind: AnswerPopulationMembershipTestKind
    polarity: AnswerPopulationMembershipTestPolarity
    test_question: str
    owned_question_input_refs: tuple[str, ...] = ()
    normal_instance_profile: NormalInstanceProfile | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("answer population test requires id")
        if not self.test_question.strip():
            raise ValueError("answer population test requires question")
        owned_refs = tuple(
            str(item).strip()
            for item in self.owned_question_input_refs
            if str(item).strip()
        )
        if len(owned_refs) != len(set(owned_refs)):
            raise ValueError("answer population test owner refs must be unique")
        if owned_refs != self.owned_question_input_refs:
            object.__setattr__(self, "owned_question_input_refs", owned_refs)
        if (
            self.kind is AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT
            and not owned_refs
        ):
            raise ValueError("explicit user constraint requires an owned input")
        if (
            self.kind is not AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT
            and owned_refs
        ):
            raise ValueError("only explicit user constraints own question inputs")
        if (
            self.kind != AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
            and self.normal_instance_profile is not None
        ):
            raise ValueError("normal instance profile requires normal instance guard")

    def to_answer_request_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "test_id": self.id,
            "kind": self.kind.value,
            "polarity": self.polarity.value,
            "test_question": self.test_question,
            "owned_question_input_refs": list(self.owned_question_input_refs),
        }
        if self.normal_instance_profile is not None:
            payload["normal_instance_profile"] = (
                self.normal_instance_profile.to_answer_request_dict()
            )
        return payload

    def to_question_contract_dict(self) -> dict[str, object]:
        return {
            "test_id": self.id,
            "kind": self.kind.value,
            "polarity": self.polarity.value,
            "test_question": self.test_question,
            "owned_question_input_refs": list(self.owned_question_input_refs),
        }


@dataclass(frozen=True)
class RequestedFactAnswerPopulation:
    population_label: str
    counted_unit: str
    membership_tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...]

    def __post_init__(self) -> None:
        if not self.population_label.strip():
            raise ValueError("answer population requires population label")
        if not self.counted_unit.strip():
            raise ValueError("answer population requires counted unit")
        if not self.membership_tests:
            raise ValueError("answer population requires membership tests")
        object.__setattr__(
            self,
            "membership_tests",
            tuple(
                _normalized_population_membership_test(
                    test,
                    counted_unit=self.counted_unit,
                )
                for test in self.membership_tests
            ),
        )
        seen: set[str] = set()
        has_subject_identity = False
        for test in self.membership_tests:
            if test.id in seen:
                raise ValueError("duplicate answer population test")
            seen.add(test.id)
            if test.kind == AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY:
                has_subject_identity = True
        if not has_subject_identity:
            raise ValueError("answer population requires subject identity test")

    def to_answer_request_dict(self) -> dict[str, object]:
        return {
            "population_label": self.population_label,
            "counted_unit": self.counted_unit,
            "membership_tests": [
                test.to_answer_request_dict() for test in self.membership_tests
            ],
        }

    def to_question_contract_dict(self) -> dict[str, object]:
        return {
            "population_label": self.population_label,
            "counted_unit": self.counted_unit,
            "membership_tests": [
                test.to_question_contract_dict() for test in self.membership_tests
            ],
        }


def _normalized_population_membership_test(
    test: RequestedFactAnswerPopulationMembershipTest,
    *,
    counted_unit: str,
) -> RequestedFactAnswerPopulationMembershipTest:
    if test.kind != AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD:
        return test
    profile = normal_instance_profile(counted_unit)
    return RequestedFactAnswerPopulationMembershipTest(
        id=test.id,
        kind=test.kind,
        polarity=test.polarity,
        test_question=normal_instance_guard_question(counted_unit),
        owned_question_input_refs=test.owned_question_input_refs,
        normal_instance_profile=profile,
    )


def default_answer_population(
    *,
    description: str,
    subject_text: str,
    instance_interpretation: RequestedFactAnswerSubjectInstanceInterpretation,
) -> RequestedFactAnswerPopulation:
    guard_kind = (
        AnswerPopulationMembershipTestKind.RAW_RECORD_GUARD
        if instance_interpretation.kind
        == AnswerSubjectInstanceInterpretationKind.RAW_DATA_RECORD
        else AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
    )
    guard_question = (
        f"Is this a persisted data record for {subject_text}?"
        if guard_kind == AnswerPopulationMembershipTestKind.RAW_RECORD_GUARD
        else (
            f"Is this an ordinary business instance of {subject_text} as normally "
            "understood in business operations and reporting?"
        )
    )
    return RequestedFactAnswerPopulation(
        population_label=description or subject_text,
        counted_unit=subject_text,
        membership_tests=(
            RequestedFactAnswerPopulationMembershipTest(
                id="pop_test_1",
                kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                test_question=f"Does the row/value represent {subject_text}?",
            ),
            RequestedFactAnswerPopulationMembershipTest(
                id="pop_test_2",
                kind=guard_kind,
                polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                test_question=(
                    normal_instance_guard_question(subject_text)
                    if guard_kind
                    == AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
                    else guard_question
                ),
                normal_instance_profile=(
                    normal_instance_profile(subject_text)
                    if guard_kind
                    == AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
                    else None
                ),
            ),
        ),
    )


@dataclass(frozen=True)
class RequestedFactPopulationConstraint:
    id: str
    included_values: tuple[str, ...]
    excluded_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("requested fact population constraint requires identity")
        if not self.included_values:
            raise ValueError(
                "requested fact population constraint requires included values"
            )
        if len(set(self.included_values)) != len(self.included_values):
            raise ValueError(
                "requested fact population constraint included values must be unique"
            )
        if len(set(self.excluded_values)) != len(self.excluded_values):
            raise ValueError(
                "requested fact population constraint excluded values must be unique"
            )
        if set(self.included_values) & set(self.excluded_values):
            raise ValueError(
                "requested fact population constraint values cannot overlap"
            )

    def to_model_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "included_values": list(self.included_values),
            "excluded_values": list(self.excluded_values),
        }


@dataclass(frozen=True)
class RequestedFact:
    id: str
    description: str
    answer_expression: RequestedFactAnswerExpression | None = None
    answer_subject: RequestedFactAnswerSubject | None = None
    answer_population: RequestedFactAnswerPopulation | None = None
    answer_outputs: tuple[RequestedFactAnswerOutput, ...] = ()
    known_inputs: tuple[RequestedFactKnownInput, ...] = ()
    input_refs: tuple[str, ...] = ()
    required_for: str = ""
    population_constraints: tuple[RequestedFactPopulationConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.description:
            raise ValueError("requested fact requires id and description")
        if not self.answer_outputs:
            raise ValueError("requested fact requires answer outputs")
        row_count_outputs = tuple(
            output for output in self.answer_outputs if output.role == "ROW_COUNT"
        )
        if len(row_count_outputs) > 1:
            raise ValueError(
                "requested fact can have at most one row population answer output"
            )
        if self.answer_population is None and self.answer_subject is not None:
            object.__setattr__(
                self,
                "answer_population",
                default_answer_population(
                    description=self.description,
                    subject_text=self.answer_subject.subject_text,
                    instance_interpretation=(
                        self.answer_subject.instance_interpretation
                    ),
                ),
            )
        input_refs = tuple(
            str(item).strip() for item in self.input_refs if str(item).strip()
        )
        if input_refs != self.input_refs:
            object.__setattr__(self, "input_refs", input_refs)
        if self.answer_population is not None:
            input_ref_set = set(input_refs)
            for test in self.answer_population.membership_tests:
                unknown_refs = tuple(
                    ref
                    for ref in test.owned_question_input_refs
                    if ref not in input_ref_set
                )
                if unknown_refs:
                    raise ValueError(
                        "answer population test owner refs must be used by requested fact"
                    )
        if (
            self.answer_expression is not None
            and self.answer_expression.family
            == RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE
            and self.answer_expression.group_key is None
        ):
            raise ValueError("grouped_aggregate requires group key")
        if (
            self.answer_expression is not None
            and self.answer_expression.group_key is not None
            and self.answer_expression.group_key.domain
            == GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS
        ):
            input_ref_set = set(input_refs)
            unknown_refs = tuple(
                ref
                for ref in self.answer_expression.group_key.question_input_refs
                if ref not in input_ref_set
            )
            if unknown_refs:
                raise ValueError(
                    "answer expression group key refs must be used by requested fact"
                )
        for output in self.answer_outputs:
            if output.role == "GROUP_KEY":
                raise ValueError("GROUP_KEY belongs to answer_expression.group_key")
        constraint_ids = tuple(item.id for item in self.population_constraints)
        if len(set(constraint_ids)) != len(constraint_ids):
            raise ValueError("duplicate requested fact population constraint")

    @property
    def support_answer_outputs(self) -> tuple[RequestedFactAnswerOutput, ...]:
        answer_outputs = self.answer_outputs
        if (
            self.answer_expression is not None
            and self.answer_expression.family
            is RequestedFactAnswerExpressionFamily.EXISTENCE_CHECK
        ):
            answer_outputs = tuple(
                replace(output, role="ROW_COUNT") for output in answer_outputs
            )
        group_key = (
            self.answer_expression.group_key
            if self.answer_expression is not None
            else None
        )
        if group_key is None:
            return answer_outputs
        return (
            RequestedFactAnswerOutput(
                id=group_key.id,
                description=group_key.description,
                role="GROUP_KEY",
            ),
            *answer_outputs,
        )

    def answer_request_model_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "answer_fact": self.description,
        }
        if self.answer_expression is not None:
            payload["answer_expression"] = (
                self.answer_expression.to_answer_request_dict()
            )
        if self.answer_subject is not None:
            payload["answer_subject"] = self.answer_subject.to_answer_request_dict()
        if self.answer_population is not None:
            payload["answer_population"] = (
                self.answer_population.to_answer_request_dict()
            )
        payload["answer_outputs"] = [
            output.to_answer_request_dict() for output in self.answer_outputs
        ]
        if self.population_constraints:
            payload["population_constraints"] = [
                constraint.to_model_dict() for constraint in self.population_constraints
            ]
        return payload


@dataclass(frozen=True)
class QuestionContract:
    question_inputs: tuple[RequestedFactKnownInput, ...] = ()
    requested_facts: tuple[RequestedFact, ...] = ()

    def __post_init__(self) -> None:
        if not self.requested_facts:
            raise ValueError("question contract requires requested facts")
        if not self.question_inputs:
            question_inputs: list[RequestedFactKnownInput] = []
            inputs_by_id: dict[str, RequestedFactKnownInput] = {}
            normalized_facts: list[RequestedFact] = []
            for fact in self.requested_facts:
                input_refs: list[str] = []
                for known in fact.known_inputs:
                    existing = inputs_by_id.get(known.id)
                    if existing is not None and existing != known:
                        raise ValueError("duplicate known input")
                    if existing is None:
                        inputs_by_id[known.id] = known
                        question_inputs.append(known)
                    input_refs.append(known.id)
                if tuple(input_refs) != fact.input_refs:
                    normalized_facts.append(replace(fact, input_refs=tuple(input_refs)))
                else:
                    normalized_facts.append(fact)
            if question_inputs:
                object.__setattr__(self, "question_inputs", tuple(question_inputs))
                object.__setattr__(self, "requested_facts", tuple(normalized_facts))
        seen: set[str] = set()
        known_input_ids: set[str] = set()
        question_input_ids: set[str] = set()
        for known in self.question_inputs:
            if known.id in question_input_ids:
                raise ValueError("duplicate question input")
            question_input_ids.add(known.id)
        inputs_by_id = {known.id: known for known in self.question_inputs}
        requested_facts: list[RequestedFact] = []
        for fact in self.requested_facts:
            if fact.id in seen:
                raise ValueError("duplicate requested fact")
            seen.add(fact.id)
            materialized_fact = fact
            if self.question_inputs:
                expected_known_inputs = tuple(
                    inputs_by_id[input_ref]
                    for input_ref in fact.input_refs
                    if input_ref in inputs_by_id
                )
                if fact.known_inputs != expected_known_inputs:
                    materialized_fact = replace(
                        fact,
                        known_inputs=expected_known_inputs,
                    )
            requested_facts.append(materialized_fact)
            output_ids: set[str] = set()
            for output in materialized_fact.support_answer_outputs:
                if output.id in output_ids:
                    raise ValueError("duplicate requested fact answer output")
                output_ids.add(output.id)
            for known in materialized_fact.known_inputs:
                if known.id in output_ids:
                    raise ValueError(
                        "answer output and known input ids must be disjoint"
                    )
                if not self.question_inputs:
                    if known.id in known_input_ids:
                        raise ValueError("duplicate known input")
                    known_input_ids.add(known.id)
            for input_ref in materialized_fact.input_refs:
                if self.question_inputs and input_ref not in question_input_ids:
                    raise ValueError("answer request references unknown question input")
        if tuple(requested_facts) != self.requested_facts:
            object.__setattr__(self, "requested_facts", tuple(requested_facts))

    def inputs_for_fact(
        self, requested_fact_id: str
    ) -> tuple[RequestedFactKnownInput, ...]:
        fact = next(
            (item for item in self.requested_facts if item.id == requested_fact_id),
            None,
        )
        if fact is None:
            return ()
        if not self.question_inputs:
            return fact.known_inputs
        inputs_by_id = {item.id: item for item in self.question_inputs}
        return tuple(
            inputs_by_id[input_ref]
            for input_ref in fact.input_refs
            if input_ref in inputs_by_id
        )

    def requested_fact_ids_for_input(
        self,
        input_ref: str,
    ) -> tuple[str, ...]:
        return tuple(
            fact.id for fact in self.requested_facts if input_ref in fact.input_refs
        )

    def to_model_dict(self) -> dict[str, object]:
        question_input_ids = tuple(known.id for known in self.question_inputs)
        return {
            "kind": "question_contract",
            "answer_requests_count": len(self.requested_facts),
            "question_inputs": [
                known.to_model_dict() for known in self.question_inputs
            ],
            "answer_requests": [
                _answer_request_contract_dict(
                    fact,
                    question_input_ids=question_input_ids,
                )
                for fact in self.requested_facts
            ],
        }


@dataclass(frozen=True)
class IncompleteFactualRequestItem:
    missing_kind: IncompleteFactualRequestKind
    source_text: str
    why_question_is_incomplete: str
    target_label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.missing_kind, IncompleteFactualRequestKind):
            raise ValueError("incomplete factual request requires structured kind")
        if not self.source_text.strip():
            raise ValueError("incomplete factual request requires source text")
        if not self.why_question_is_incomplete.strip():
            raise ValueError("incomplete factual request requires a reason")
        if (
            self.missing_kind
            is IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE
            and not self.target_label.strip()
        ):
            raise ValueError("missing target reference requires target label")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "missing_kind": self.missing_kind.value,
            "source_text": self.source_text,
            "target_label": self.target_label or None,
            "why_question_is_incomplete": self.why_question_is_incomplete,
        }


@dataclass(frozen=True)
class QuestionContractNeedsClarification:
    missing: tuple[IncompleteFactualRequestItem, ...]

    def __post_init__(self) -> None:
        if not self.missing:
            raise ValueError("question-contract clarification requires missing inputs")
        kinds = {item.missing_kind for item in self.missing}
        if len(kinds) != 1:
            raise ValueError("question-contract clarification cannot mix missing kinds")
        if (
            IncompleteFactualRequestKind.MISSING_REQUESTED_FACT in kinds
            and len(self.missing) != 1
        ):
            raise ValueError("missing requested fact must be singular")

    def to_model_dict(self) -> dict[str, object]:
        if all(
            item.missing_kind
            is IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE
            for item in self.missing
        ):
            references = [
                {
                    "source_text": item.source_text,
                    "target_label": item.target_label,
                    "why_question_is_incomplete": item.why_question_is_incomplete,
                }
                for item in self.missing
            ]
            return {
                "kind": "unresolved_prior_turn_references",
                "references": references,
            }
        item = self.missing[0]
        return {
            "kind": "missing_requested_fact",
            "source_text": item.source_text,
            "why_question_is_incomplete": item.why_question_is_incomplete,
        }


QuestionContractOutcome = QuestionContract | QuestionContractNeedsClarification


@dataclass(frozen=True)
class QuestionContractResult:
    decision_basis: str
    outcome: QuestionContractOutcome

    def __post_init__(self) -> None:
        if not self.decision_basis.strip():
            raise ValueError("question contract result requires a decision basis")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "decision_basis": self.decision_basis,
            "outcome": self.outcome.to_model_dict(),
        }


@dataclass(frozen=True)
class QuestionContractRequest:
    current_question: str
    conversation_context: dict[str, Any]
    conversation_resolution: CompiledConversationResolution | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)
    clarification_source: ClarificationResponseSource | None = None
    clarification_missing_item_id: str = ""
    clarification_expected_value_kind: str = ""


def _answer_request_contract_dict(
    fact: RequestedFact,
    *,
    question_input_ids: tuple[str, ...],
) -> dict[str, object]:
    payload = {
        "id": fact.id,
        **fact.answer_request_model_dict(),
        "used_question_inputs": [
            input_id for input_id in question_input_ids if input_id in fact.input_refs
        ],
    }
    payload["answer_outputs"] = [
        output.to_model_dict() for output in fact.answer_outputs
    ]
    return payload


def requested_fact_evidence_ref(requested_fact_id: str) -> str:
    return f"requested_fact:{requested_fact_id}"


def validate_question_contract_against_question(
    contract: QuestionContract,
    *,
    question: str,
    context_texts: tuple[str, ...],
) -> None:
    question_text = _normalized_text(question)
    known_input_texts = (question_text,) + tuple(
        _normalized_text(text) for text in context_texts if str(text or "").strip()
    )
    known_inputs = (
        contract.question_inputs
        if contract.question_inputs
        else tuple(
            known for fact in contract.requested_facts for known in fact.known_inputs
        )
    )
    for known in known_inputs:
        if not _text_in_any_context(known.text, known_input_texts):
            raise ValueError("known input text must come from question context")


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _text_in_any_context(text: object, contexts: tuple[str, ...]) -> bool:
    normalized = _normalized_text(text)
    return bool(
        normalized
        and any(contains_copied_span(context, normalized) for context in contexts)
    )
