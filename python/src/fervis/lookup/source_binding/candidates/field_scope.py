"""Read-eligibility field scope for source-binding prompt surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


class ReadAssessmentLike(Protocol):
    source_candidate_id: str
    is_retained: bool
    relevant_field_refs: tuple[str, ...]


class ReadEligibilityResultLike(Protocol):
    read_assessments: tuple[ReadAssessmentLike, ...]


@dataclass(frozen=True)
class SourceBindingFieldScope:
    scoped_candidate_ids: frozenset[str]
    field_refs_by_candidate_id: Mapping[str, frozenset[str]]

    @classmethod
    def unscoped(cls) -> SourceBindingFieldScope:
        return cls(scoped_candidate_ids=frozenset(), field_refs_by_candidate_id={})

    @classmethod
    def from_read_eligibility(
        cls,
        read_eligibility: ReadEligibilityResultLike | None,
    ) -> SourceBindingFieldScope:
        if read_eligibility is None:
            return cls.unscoped()
        scoped_candidate_ids = frozenset(
            item.source_candidate_id
            for item in read_eligibility.read_assessments
            if item.is_retained
        )
        field_refs_by_candidate_id = {
            item.source_candidate_id: frozenset(item.relevant_field_refs)
            for item in read_eligibility.read_assessments
            if item.is_retained
        }
        return cls(
            scoped_candidate_ids=scoped_candidate_ids,
            field_refs_by_candidate_id=field_refs_by_candidate_id,
        )

    def field_refs_for_candidate(self, candidate_id: str) -> frozenset[str] | None:
        if candidate_id not in self.scoped_candidate_ids:
            return None
        return self.field_refs_by_candidate_id.get(candidate_id, frozenset())

    def includes_field_ref(self, *, candidate_id: str, field_ref: str) -> bool:
        field_refs = self.field_refs_for_candidate(candidate_id)
        if field_refs is None:
            return True
        return field_ref in field_refs
