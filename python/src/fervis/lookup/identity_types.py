"""Identity resolution outcomes shared by planning and canonical execution."""

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from fervis.lookup.canonical_data import EntityKeyValue


class IdentityExecutionFailureReason(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS_RESULT = "AMBIGUOUS_RESULT"
    INVALID_RESOLVER_RESULT = "INVALID_RESOLVER_RESULT"


@dataclass(frozen=True)
class ReferenceResolutionFailure:
    input_ref: str
    reason: IdentityExecutionFailureReason
    candidates: tuple[EntityKeyValue, ...] = ()
    operand: str = ""
