"""Read retention and identity decisions owned by Read Eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from fervis.lookup.grounding import IdentityResolutionTask
from fervis.lookup.question_contract import RequestedFactSemanticIndex
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceField,
    RowSourceKind,
    api_read_source_groups,
)
from fervis.types.enums import StrEnum


class SemanticReadDecision(StrEnum):
    RETAIN = "RETAIN"
    DROP = "DROP"


@dataclass(frozen=True)
class SemanticReadCandidate:
    candidate_ref: str
    read_id: str
    sources: tuple[RowSource, ...]

    @property
    def source_refs(self) -> tuple[str, ...]:
        return tuple(source.id for source in self.sources)

    @property
    def fields(self) -> tuple[RowSourceField, ...]:
        return tuple(
            {
                field.field_ref: field
                for source in self.sources
                for field in source.fields
            }.values()
        )


@dataclass(frozen=True)
class ReadRequirementAssessment:
    requested_fact_id: str
    candidate_ref: str
    source_refs: tuple[str, ...]
    read_id: str
    relevant_field_refs: tuple[str, ...]
    assessment_basis: str
    decision: SemanticReadDecision


@dataclass(frozen=True)
class CanonicalOptionAssessment:
    canonical_option_id: str
    assessment: str
    decision: str


@dataclass(frozen=True)
class ResolverRouteAssessment:
    resolver_route_id: str
    assessment: str
    decision: str


@dataclass(frozen=True)
class IdentityRouteSelection:
    task_ref: str
    canonical_option_assessments: tuple[CanonicalOptionAssessment, ...]
    canonical_option_basis: str
    canonical_option_id: str
    resolver_route_assessments: tuple[ResolverRouteAssessment, ...]
    resolver_route_basis: str
    resolver_route_id: str


class SemanticFitDecision(StrEnum):
    FITS = "FITS"
    DOES_NOT_FIT = "DOES_NOT_FIT"


@dataclass(frozen=True)
class NoCanonicalInterpretation:
    task_ref: str
    canonical_option_assessments: tuple[CanonicalOptionAssessment, ...]
    canonical_option_basis: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class NoResolverRoute:
    task_ref: str
    canonical_option_assessments: tuple[CanonicalOptionAssessment, ...]
    canonical_option_basis: str
    canonical_option_id: str
    resolver_route_assessments: tuple[ResolverRouteAssessment, ...]
    resolver_route_basis: str
    evidence_refs: tuple[str, ...]


IdentityRouteOutcome: TypeAlias = (
    IdentityRouteSelection | NoCanonicalInterpretation | NoResolverRoute
)


@dataclass(frozen=True)
class SemanticReadEligibilityRequest:
    indexes: tuple[RequestedFactSemanticIndex, ...]
    source_catalog: RowSourceCatalog
    answer_catalog: RelationCatalog
    identity_tasks: tuple[IdentityResolutionTask, ...]
    resolver_catalog: RelationCatalog

    @property
    def read_candidates(self) -> tuple[SemanticReadCandidate, ...]:
        api_sources_by_read: dict[str, list[RowSource]] = {}
        for source in self.source_catalog.sources:
            if source.kind is RowSourceKind.API_READ:
                api_sources_by_read.setdefault(source.read_id, []).append(source)
        groups_by_read = {
            read_id: api_read_source_groups(tuple(sources))
            for read_id, sources in api_sources_by_read.items()
        }
        candidates: list[SemanticReadCandidate] = []
        emitted_reads: set[str] = set()
        for source in self.source_catalog.sources:
            if source.kind is not RowSourceKind.API_READ:
                candidates.append(
                    SemanticReadCandidate(
                        candidate_ref=source.id,
                        read_id=source.read_id,
                        sources=(source,),
                    )
                )
                continue
            if source.read_id in emitted_reads:
                continue
            emitted_reads.add(source.read_id)
            groups = groups_by_read[source.read_id]
            for position, group in enumerate(groups, start=1):
                candidates.append(
                    SemanticReadCandidate(
                        candidate_ref=(
                            source.read_id
                            if len(groups) == 1
                            else f"{source.read_id}:variant:{position}"
                        ),
                        read_id=source.read_id,
                        sources=group,
                    )
                )
        return tuple(candidates)

    def __post_init__(self) -> None:
        source_refs = tuple(item.id for item in self.source_catalog.sources)
        if len(source_refs) != len(set(source_refs)):
            raise ValueError("read eligibility repeats an available source")
        known_read_ids = {read.id for read in self.answer_catalog.reads}
        unknown_read_ids = {
            source.read_id
            for source in self.source_catalog.sources
            if source.read_id and source.read_id not in known_read_ids
        }
        if unknown_read_ids:
            raise ValueError(
                "read eligibility source is absent from the answer catalog"
            )
        candidate_refs = tuple(item.candidate_ref for item in self.read_candidates)
        if len(candidate_refs) != len(set(candidate_refs)):
            raise ValueError("read eligibility repeats a read candidate")
        task_refs = tuple(item.task_ref for item in self.identity_tasks)
        if len(task_refs) != len(set(task_refs)):
            raise ValueError("read eligibility repeats an identity task")


@dataclass(frozen=True)
class SemanticReadEligibilityResult:
    read_assessments: tuple[ReadRequirementAssessment, ...]
    identity_outcomes: tuple[IdentityRouteOutcome, ...]


def combine_semantic_read_eligibility_results(
    results: tuple[SemanticReadEligibilityResult, ...],
) -> SemanticReadEligibilityResult:
    """Combine disjoint assessed batches into one current-run decision ledger."""

    if not results:
        raise ValueError("read eligibility results must not be empty")
    assessments = tuple(
        assessment for result in results for assessment in result.read_assessments
    )
    assessment_keys = tuple(
        (item.requested_fact_id, item.candidate_ref) for item in assessments
    )
    if len(assessment_keys) != len(set(assessment_keys)):
        raise ValueError("read eligibility batches repeat a fact-local candidate")
    outcomes = tuple(
        outcome for result in results for outcome in result.identity_outcomes
    )
    task_refs = tuple(item.task_ref for item in outcomes)
    if len(task_refs) != len(set(task_refs)):
        raise ValueError("read eligibility batches repeat an identity task")
    return SemanticReadEligibilityResult(
        read_assessments=assessments,
        identity_outcomes=outcomes,
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
