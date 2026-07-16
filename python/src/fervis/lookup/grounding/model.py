"""Canonical pre-plan grounding models for question-contract known inputs."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

from fervis.lookup.answer_program.values import (
    FactValue,
    TimeComponent,
    ValueComponent,
)
from fervis.lookup.canonical_data import EntityKeyValue
from fervis.lookup.fact_plan.row_sources import RowSource
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogScalarParameterValue,
)
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.types.enums import StrEnum


class InputBindingPurpose(StrEnum):
    IDENTITY_VALIDATION = "identity_validation"
    REFERENCE_GROUNDING = "reference_grounding"


class LookupTextResolutionDecision(StrEnum):
    CAN_RESOLVE_LOOKUP_TEXT = "CAN_RESOLVE_LOOKUP_TEXT"
    CANNOT_RESOLVE_LOOKUP_TEXT = "CANNOT_RESOLVE_LOOKUP_TEXT"


@dataclass(frozen=True)
class ExpectedInputIdentity:
    entity_kind: str
    key_id: str
    key_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.entity_kind or not self.key_id or not self.key_component_ids:
            raise ValueError(
                "expected input identity must name its complete candidate key"
            )


class GroundingTerminalKind(StrEnum):
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    AMBIGUOUS_BINDING = "ambiguous_binding"
    UNSUPPORTED_REFERENCE = "unsupported_reference"
    INCOMPLETE_REFERENCE = "incomplete_reference"
    TIME_RESOLUTION_FAILED = "time_resolution_failed"


class GroundedValueCertificationMethod(StrEnum):
    IDENTITY_VALIDATION_READ = "identity_validation_read"
    RESOLVER_SOURCE_READ = "resolver_source_read"
    IMPORTED_PRIOR_IDENTITY = "imported_prior_identity"
    CLARIFICATION_SELECTION = "clarification_selection"


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
class InputBindingKeyComponent:
    component_id: str
    field_id: str
    field_ref: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.field_id or not self.field_ref:
            raise ValueError("input binding key component is incomplete")


@dataclass(frozen=True)
class ResolverCandidate:
    known_input_id: str
    resolver_source: RowSource
    entity_kind: str
    key_id: str
    key_components: tuple[InputBindingKeyComponent, ...]
    resolver_resource_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.known_input_id
            or not self.resolver_source.id
            or not self.resolver_source.row_path_id
            or not self.resolver_source.read_id
            or not self.resolver_source.endpoint_name
            or not self.entity_kind
            or not self.key_id
        ):
            raise ValueError("resolver candidate is incomplete")
        if not self.key_components:
            raise ValueError("resolver candidate requires a complete candidate key")

    @property
    def resolver_row_source_id(self) -> str:
        return self.resolver_source.id

    @property
    def resolver_row_path_id(self) -> str:
        return self.resolver_source.row_path_id

    @property
    def resolver_read_id(self) -> str:
        return self.resolver_source.read_id

    @property
    def resolver_endpoint_name(self) -> str:
        return self.resolver_source.endpoint_name

    @property
    def result_surface(self) -> str:
        return f"entity {self.entity_kind}:{self.key_id}"


@dataclass(frozen=True)
class InputBindingOption:
    id: str
    known_input_id: str
    candidate: ResolverCandidate

    def __post_init__(self) -> None:
        if not self.id or not self.known_input_id:
            raise ValueError("input binding option is incomplete")
        if self.candidate.known_input_id != self.known_input_id:
            raise ValueError("input binding option and candidate have different owners")


@dataclass(frozen=True)
class KnownInputBindingTask:
    known_input_id: str
    known_input_text: str
    known_input_kind: str
    requested_fact_id: str
    options: tuple[InputBindingOption, ...]
    lookup_text: str
    field_label_text: str = ""
    known_input_description: str = ""
    applies_to_requested_fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.lookup_text.strip():
            raise ValueError("known input binding task requires lookup text")
        option_ids = tuple(option.id for option in self.options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("known input binding task repeats a resolver option")


@dataclass(frozen=True)
class GroundingRequestedFactCard:
    requested_fact_id: str
    answer_fact: str
    answer_population_label: str = ""
    answer_population_counted_unit: str = ""
    answer_outputs: tuple[dict[str, str], ...] = ()


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
class ResolverRequestValue:
    param_ref: str
    value: str | int | float | bool

    def __post_init__(self) -> None:
        if not self.param_ref:
            raise ValueError("resolver request value requires parameter reference")


@dataclass(frozen=True)
class CompatibleInputBinding:
    option_id: str
    lookup_value: CatalogScalarParameterValue
    request_values: tuple[ResolverRequestValue, ...]
    response_match_field_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.option_id or not self.response_match_field_paths:
            raise ValueError("compatible input binding is incomplete")
        param_refs = tuple(item.param_ref for item in self.request_values)
        if len(param_refs) != len(set(param_refs)):
            raise ValueError("compatible input binding repeats a parameter")
        if len(self.response_match_field_paths) != len(
            set(self.response_match_field_paths)
        ):
            raise ValueError("compatible input binding repeats a match field")


@dataclass(frozen=True)
class InputBindingCompatibility:
    known_input_id: str
    bindings: tuple[CompatibleInputBinding, ...]


@dataclass(frozen=True)
class TimeResolutionIntent:
    known_input_id: str
    date_intent: dict[str, object]


@dataclass(frozen=True)
class GroundingRequest:
    question: str
    tasks: tuple[KnownInputBindingTask, ...]
    resolver_catalog: RelationCatalog
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
    candidate = option.candidate
    meaning = (
        task.known_input_description or task.field_label_text or "the supplied value"
    )
    return (
        f"Can read {candidate.resolver_read_id} resolve {task.lookup_text} as the "
        f"returned resource and produce {candidate.result_surface} for target "
        f"meaning {meaning}?"
    )


@dataclass(frozen=True)
class GroundedInputUse:
    id: str
    value_id: str
    row_source_id: str
    param_id: str
    requested_fact_id: str
    field_id: str = ""
    entity_kind: str = ""
    key_component_id: str = ""
    value_component: ValueComponent | TimeComponent = ValueComponent.VALUE

    def __post_init__(self) -> None:
        if not self.requested_fact_id.strip():
            raise ValueError("grounded input use requires requested fact id")


@dataclass(frozen=True)
class GroundingCandidate:
    id: str
    label: str = ""
    key: EntityKeyValue | None = None
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
