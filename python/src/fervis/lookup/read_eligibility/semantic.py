"""Read retention and identity decisions owned by Read Eligibility."""

from __future__ import annotations
from fervis.lookup.source_reads.access_model import ReadAccessCatalog

from dataclasses import dataclass
from typing import TypeAlias

from fervis.lookup.grounding import CanonicalIdentityOption, IdentityResolutionTask
from fervis.lookup.question_contract import InputTerm, RequestedFactSemanticIndex
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
class ReturnedIdentityUse:
    identity_ref: str
    field_refs: tuple[str, ...]


@dataclass(frozen=True)
class AnswerReadIdentityUse:
    read_id: str
    request_param_refs: tuple[str, ...]
    returned_identities: tuple[ReturnedIdentityUse, ...]


@dataclass(frozen=True)
class CanonicalIdentityAnswerUses:
    canonical_option_id: str
    identity_ref: str
    answer_reads: tuple[AnswerReadIdentityUse, ...]


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
    read_access: ReadAccessCatalog = ReadAccessCatalog()

    def input_term(self, input_ref: str) -> InputTerm:
        terms = {
            index.input_by_ref[input_ref]
            for index in self.indexes
            if input_ref in index.input_by_ref
        }
        if len(terms) != 1:
            raise ValueError("identity task input is not uniquely declared")
        return next(iter(terms))

    def canonical_identity_answer_uses(
        self,
        task: IdentityResolutionTask,
    ) -> tuple[CanonicalIdentityAnswerUses, ...]:
        return tuple(
            self._canonical_identity_answer_uses(task, option)
            for option in task.canonical_options
        )

    def _canonical_identity_answer_uses(
        self,
        task: IdentityResolutionTask,
        option: CanonicalIdentityOption,
    ) -> CanonicalIdentityAnswerUses:
        routes = tuple(
            route
            for route in task.resolver_routes
            if route.route_ref in option.resolver_route_refs
        )
        contracts = {
            (
                route.option.candidate.entity_kind,
                route.option.candidate.key_id,
                tuple(
                    component.component_id
                    for component in route.option.candidate.key_components
                ),
            )
            for route in routes
        }
        if len(contracts) != 1:
            raise ValueError("canonical option does not own one identity contract")
        entity_kind, key_id, component_ids = next(iter(contracts))
        answer_reads: list[AnswerReadIdentityUse] = []
        for candidate in self.read_candidates:
            request_param_refs = _complete_identity_request_params(
                candidate,
                entity_kind=entity_kind,
                key_id=key_id,
                component_ids=component_ids,
            )
            returned_identity = tuple(
                evidence
                for source in candidate.sources
                for evidence in source.identity_evidence
                if evidence.entity_kind == entity_kind and evidence.key_id == key_id
            )
            if not request_param_refs and not returned_identity:
                continue
            answer_reads.append(
                AnswerReadIdentityUse(
                    read_id=candidate.read_id,
                    request_param_refs=request_param_refs,
                    returned_identities=tuple(
                        ReturnedIdentityUse(
                            identity_ref=identity_ref,
                            field_refs=tuple(
                                dict.fromkeys(
                                    field_ref
                                    for item in returned_identity
                                    if item.identity_ref == identity_ref
                                    for field_ref in item.field_refs
                                )
                            ),
                        )
                        for identity_ref in dict.fromkeys(
                            item.identity_ref for item in returned_identity
                        )
                    ),
                )
            )
        return CanonicalIdentityAnswerUses(
            canonical_option_id=option.canonical_option_id,
            identity_ref=option.identity_ref,
            answer_reads=tuple(answer_reads),
        )

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


def _complete_identity_request_params(
    candidate: SemanticReadCandidate,
    *,
    entity_kind: str,
    key_id: str,
    component_ids: tuple[str, ...],
) -> tuple[str, ...]:
    params = tuple(
        param
        for source in candidate.sources
        for param in source.params
        if param.entity_target is not None
        and param.entity_target.entity_kind == entity_kind
        and param.entity_target.key_id == key_id
        and param.entity_target.component_id in component_ids
    )
    if {
        param.entity_target.component_id
        for param in params
        if param.entity_target is not None
    } != set(component_ids):
        return ()
    return tuple(dict.fromkeys(param.param_ref for param in params))


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
