"""Identity resolution outcomes shared by planning and canonical execution."""

from dataclasses import dataclass
from collections.abc import Mapping
from fervis.types.enums import StrEnum
from fervis.lookup.canonical_data import (
    EntityKeyValue, RuntimeValue, runtime_value_to_payload,
    runtime_value_from_payload,
)


class IdentityExecutionFailureReason(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS_RESULT = "AMBIGUOUS_RESULT"
    INVALID_RESOLVER_RESULT = "INVALID_RESOLVER_RESULT"


@dataclass(frozen=True)
class ObservedReferenceValue:
    field_ref: str
    type_name: str
    label: str
    value: RuntimeValue

    def __post_init__(self) -> None:
        if not all((self.field_ref, self.type_name, self.label)):
            raise ValueError("Observed reference choice requires a typed property")

    def to_payload(self) -> dict[str, RuntimeValue]:
        return {
            "fieldRef": self.field_ref, "type": self.type_name,
            "label": self.label, "value": runtime_value_to_payload(self.value),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ObservedReferenceValue":
        if not isinstance(payload, Mapping):
            raise ValueError("observed reference property must be an object")
        values = (payload.get("fieldRef"), payload.get("type"), payload.get("label"))
        if "value" not in payload or any(
            not isinstance(item, str) or not item for item in values
        ):
            raise ValueError("observed reference property requires field, type and label")
        return cls(
            field_ref=str(values[0]), type_name=str(values[1]), label=str(values[2]),
            value=runtime_value_from_payload(payload.get("value")),
        )


@dataclass(frozen=True)
class ObservedReferenceCandidate:
    source_ref: str
    properties: tuple[ObservedReferenceValue, ...]
    display_properties: tuple[ObservedReferenceValue, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_ref or not self.properties:
            raise ValueError("Observed reference candidate requires a source and properties")
        if self.display_properties and not all(
            item in self.properties for item in self.display_properties
        ):
            raise ValueError("Displayed properties must belong to the observed candidate")


def observed_reference_values_from_payload(raw: object) -> tuple[ObservedReferenceValue, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("observed reference properties must be an array")
    return tuple(ObservedReferenceValue.from_payload(item) for item in raw)


@dataclass(frozen=True)
class ReferenceResolutionFailure:
    input_ref: str
    reason: IdentityExecutionFailureReason
    candidates: tuple[EntityKeyValue, ...] = ()
    operand: str = ""
    observed_candidates: tuple[ObservedReferenceCandidate, ...] = ()
