"""Canonical pre-plan grounding models for question-contract known inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fervis.lookup.relation_catalog import IdentityMetadata
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lookup.answer_program.values import (
    FactValue,
    TimeComponent,
    ValueComponent,
)


class LookupTextResolutionDecision(StrEnum):
    CAN_RESOLVE_LOOKUP_TEXT = "CAN_RESOLVE_LOOKUP_TEXT"
    CANNOT_RESOLVE_LOOKUP_TEXT = "CANNOT_RESOLVE_LOOKUP_TEXT"


class GroundingTerminalKind(StrEnum):
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    AMBIGUOUS_BINDING = "ambiguous_binding"
    UNSUPPORTED_REFERENCE = "unsupported_reference"
    TIME_RESOLUTION_FAILED = "time_resolution_failed"


class GroundedValueCertificationMethod(StrEnum):
    RESOLVER_SOURCE_READ = "resolver_source_read"
    IMPORTED_PRIOR_IDENTITY = "imported_prior_identity"


@dataclass(frozen=True)
class GroundedValueCertification:
    value_id: str
    method: GroundedValueCertificationMethod
    authority_refs: tuple[str, ...] = ()
    lineage_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.value_id.strip():
            raise ValueError("grounded value certification requires value id")

    def to_payload(self) -> dict[str, object]:
        return {
            "value_id": self.value_id,
            "method": self.method.value,
            "authority_refs": list(self.authority_refs),
            "lineage_refs": list(self.lineage_refs),
        }


@dataclass(frozen=True)
class ResolverQueryParamCard:
    param_ref: str
    name: str
    type: str
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolverOutputFieldCard:
    field_ref: str
    field_path: str
    type: str
    choices: tuple[str, ...] = ()
    identity: IdentityMetadata | None = None


@dataclass(frozen=True)
class InputBindingRoute:
    known_input_id: str
    resolver_row_source_id: str
    resolver_read_id: str
    resolver_endpoint_name: str
    lookup_param_id: str
    lookup_param_ref: str
    lookup_field_ids: tuple[str, ...]
    lookup_field_refs: tuple[str, ...]
    return_field_id: str
    return_field_ref: str
    identity_type: str
    identity_field: str
    display: str
    resolver_resource_names: tuple[str, ...] = ()
    query_params: tuple[ResolverQueryParamCard, ...] = ()
    selected_output_fields: tuple[ResolverOutputFieldCard, ...] = ()


@dataclass(frozen=True)
class GroundingRequestedFactCard:
    requested_fact_id: str
    answer_fact: str
    answer_population_label: str = ""
    answer_population_counted_unit: str = ""
    answer_outputs: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class InputBindingOption:
    id: str
    known_input_id: str
    path: str
    route: InputBindingRoute | None = None


@dataclass(frozen=True)
class KnownInputBindingTask:
    known_input_id: str
    known_input_text: str
    known_input_kind: str
    requested_fact_id: str
    options: tuple[InputBindingOption, ...]
    lookup_text: str
    known_input_description: str = ""
    applies_to_requested_fact_ids: tuple[str, ...] = ()
    requested_facts: tuple[GroundingRequestedFactCard, ...] = ()

    def __post_init__(self) -> None:
        if not self.lookup_text.strip():
            raise ValueError("known input binding task requires lookup text")


@dataclass(frozen=True)
class KnownTimeResolutionTask:
    known_input_id: str
    known_input_text: str
    requested_fact_id: str
    time_expression: str
    applies_to_requested_fact_ids: tuple[str, ...] = ()
    requested_facts: tuple[GroundingRequestedFactCard, ...] = ()

    def __post_init__(self) -> None:
        if not self.time_expression.strip():
            raise ValueError("known time resolution task requires time expression")


@dataclass(frozen=True)
class InputBindingCompatibility:
    known_input_id: str
    binding_option_ids: tuple[str, ...]


@dataclass(frozen=True)
class TimeResolutionIntent:
    known_input_id: str
    date_intent: dict[str, object]


@dataclass(frozen=True)
class GroundingRequest:
    question: str
    tasks: tuple[KnownInputBindingTask, ...]
    time_tasks: tuple[KnownTimeResolutionTask, ...] = ()
    conversation_context: dict[str, Any] = field(default_factory=dict)
    host: HostPromptContext = field(default_factory=HostPromptContext)


@dataclass(frozen=True)
class GroundingCompatibilityResult:
    compatibilities: tuple[InputBindingCompatibility, ...]
    time_resolutions: tuple[TimeResolutionIntent, ...] = ()


def resolver_fit_question_for_option(
    *,
    task: KnownInputBindingTask,
    option: InputBindingOption,
) -> str:
    return (
        f"Can this resolver search lookup text "
        f"'{task.lookup_text}' and return canonical "
        f"API identity '{_option_identity_type(option)}' for target meaning "
        f"'{task.known_input_description}'?"
    )


def option_has_targetable_identity(option: InputBindingOption) -> bool:
    return bool(_targetable_identity_type(option))


def _targetable_identity_type(option: InputBindingOption) -> str:
    if option.route is not None:
        return option.route.identity_type
    return ""


def _option_identity_type(option: InputBindingOption) -> str:
    targetable = _targetable_identity_type(option)
    if targetable:
        return targetable
    return "no_returned_identity"


@dataclass(frozen=True)
class GroundedInputUse:
    id: str
    value_id: str
    row_source_id: str
    param_id: str
    field_id: str = ""
    value_component: ValueComponent | TimeComponent = ValueComponent.VALUE


@dataclass(frozen=True)
class GroundingCandidate:
    id: str
    label: str = ""
    entity_kind: str = ""
    matched_label: str = ""
    matched_field: str = ""
    matched_value: str = ""
    resolver_read_id: str = ""
    resolver_label: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("grounding candidate requires id")


@dataclass(frozen=True)
class GroundingIssue:
    kind: GroundingTerminalKind
    known_input_id: str
    requested_fact_id: str
    message: str = ""
    known_input_text: str = ""
    known_input_description: str = ""
    candidates: tuple[str, ...] = ()
    candidate_options: tuple[GroundingCandidate, ...] = ()
    proof_refs: tuple[str, ...] = ()
    resolver_read_id: str = ""
    resolver_endpoint_name: str = ""
    resolver_field_id: str = ""
    identity_field: str = ""


@dataclass(frozen=True)
class CanonicalInputLedger:
    values: tuple[FactValue, ...] = ()
    uses: tuple[GroundedInputUse, ...] = ()
    issues: tuple[GroundingIssue, ...] = ()
    certifications: tuple[GroundedValueCertification, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues
