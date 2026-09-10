"""Provider output for incomplete factual questions."""

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


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


__all__ = tuple(name for name in globals() if not name.startswith("_"))
