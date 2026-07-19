"""Typed provider-output contracts for grounding."""

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


@dataclass(frozen=True)
class FlatTimeIntentOutput(ProviderOutput):
    year: int
    month: int
    day: int
    year_policy: str
    relative_offset: int
    named_value: int
    end_year: int
    end_month: int
    end_day: int
    end_year_policy: str
    count: int
    direction: str
    time_shape: str
    unit: str
    mode: str


@dataclass(frozen=True)
class DateIntentOutput(ProviderOutput):
    expression: str
    intent: FlatTimeIntentOutput


@dataclass(frozen=True)
class KnownTimeResolutionOutput(ProviderOutput):
    date_intent: DateIntentOutput


@dataclass(frozen=True)
class LookupRequestParamOutput(ProviderOutput):
    param_ref: str
    value: str | int | float | bool


@dataclass(frozen=True)
class ResolverResolutionOutput(ProviderOutput):
    decision: str
    lookup_request_params: tuple[LookupRequestParamOutput, ...]
    returned_identity_verification_fields: tuple[str, ...]


@dataclass(frozen=True)
class OptionReviewOutput(ProviderOutput):
    resource_type: str
    resource_type_match: str
    resolver_fit_question: str
    because: str
    resolution: ResolverResolutionOutput


@dataclass(frozen=True)
class KnownInputBindingReviewOutput(ProviderOutput):
    resource_type_basis: str
    resource_type_x: str
    identifier_kind_basis: str
    identifier_kind: str
    option_reviews: dict[str, OptionReviewOutput]


@dataclass(frozen=True)
class GroundingOutput(ProviderOutput):
    known_time_resolutions: dict[str, KnownTimeResolutionOutput]
    known_input_binding_reviews: dict[str, KnownInputBindingReviewOutput]
