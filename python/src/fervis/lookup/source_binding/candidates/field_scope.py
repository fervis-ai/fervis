"""Read-eligibility field scope for source-binding prompt surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


class ReadAssessmentLike(Protocol):
    @property
    def source_candidate_id(self) -> str: ...

    @property
    def is_retained(self) -> bool: ...

    @property
    def relevant_field_refs(self) -> tuple[str, ...]: ...


class ReadEligibilityResultLike(Protocol):
    @property
    def read_assessments(self) -> tuple[ReadAssessmentLike, ...]: ...


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
