"""Provider DTOs for semantic temporal grounding."""

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


__all__ = ["DateIntentOutput", "FlatTimeIntentOutput", "KnownTimeResolutionOutput"]
