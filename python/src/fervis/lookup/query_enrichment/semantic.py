"""Semantic-requirement recall contract for Query Enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from fervis.lookup.question_contract import (
    RequestedFactSemanticIndex,
    AssociationTerm,
    FactLocalRef,
    FactTerm,
    InputTerm,
    SetTerm,
)
from fervis.lookup.semantic_types import IdentifierType, SourceOrigin
from fervis.types.enums import StrEnum


RESOURCE_NAME_GROUP_SIZE = 64


class SemanticRecallRequirementKind(StrEnum):
    SET = "set"
    FACT = "fact"
    ASSOCIATION = "association"


@dataclass(frozen=True)
class SemanticRecallRequirement:
    requirement_ref: FactLocalRef
    kind: SemanticRecallRequirementKind
    origin: SourceOrigin
    owner_refs: tuple[FactLocalRef, ...]
    identity_set_ref: FactLocalRef | None
    use_site_refs: tuple[str, ...]


class SemanticRecallBucketKind(StrEnum):
    POPULATION = "population"
    OUTPUT = "output"


@dataclass(frozen=True)
class SemanticRecallBucket:
    bucket_ref: str
    kind: SemanticRecallBucketKind
    origin: SourceOrigin
    requirement_refs: tuple[FactLocalRef, ...]


@dataclass(frozen=True)
class ReferenceInputRecallTask:
    input_use_ref: str
    input_ref: str
    input_origin: SourceOrigin
    operand_meaning: str
    reference_fact_ref: FactLocalRef
    expected_set_ref: FactLocalRef | None


@dataclass(frozen=True)
class RecallBucketMatch:
    bucket_ref: str
    exhaustive_resource_names: tuple[str, ...]
    matching_resource_names: tuple[str, ...]


@dataclass(frozen=True)
class InputResourceSearchTerms:
    input_use_ref: str
    catalog_search_terms: tuple[str, ...]


@dataclass(frozen=True)
class SemanticQueryEnrichmentResult:
    recall_bucket_matches: tuple[RecallBucketMatch, ...]
    input_resource_search_terms: tuple[InputResourceSearchTerms, ...]


@dataclass(frozen=True)
class SemanticQueryEnrichmentRequest:
    recall_buckets: tuple[SemanticRecallBucket, ...]
    reference_tasks: tuple[ReferenceInputRecallTask, ...]
    resource_names: tuple[str, ...]

    def __post_init__(self) -> None:
        bucket_refs = tuple(item.bucket_ref for item in self.recall_buckets)
        if len(bucket_refs) != len(set(bucket_refs)):
            raise ValueError("semantic query enrichment repeats a recall bucket")
        task_refs = tuple(item.input_use_ref for item in self.reference_tasks)
        if len(task_refs) != len(set(task_refs)):
            raise ValueError("semantic query enrichment repeats a reference use")
        if len(self.resource_names) != len(set(self.resource_names)):
            raise ValueError("semantic query enrichment repeats a resource name")

    @property
    def resource_name_groups(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            self.resource_names[offset : offset + RESOURCE_NAME_GROUP_SIZE]
            for offset in range(
                0, len(self.resource_names), RESOURCE_NAME_GROUP_SIZE
            )
        )


def semantic_recall_requirements(
    index: RequestedFactSemanticIndex,
) -> tuple[SemanticRecallRequirement, ...]:
    output: list[SemanticRecallRequirement] = []
    kind_order = {"set": 0, "association": 1, "fact": 2}
    for requirement_ref in sorted(
        index.source_requirement_refs,
        key=lambda item: (kind_order[item.kind.value], item.local_id),
    ):
        term = index.term_by_ref[requirement_ref]
        kind: SemanticRecallRequirementKind
        owners: tuple[FactLocalRef, ...]
        identity: FactLocalRef | None = None
        if isinstance(term, SetTerm):
            kind = SemanticRecallRequirementKind.SET
            owners = ()
        elif isinstance(term, AssociationTerm):
            kind = SemanticRecallRequirementKind.ASSOCIATION
            owners = (
                index.fact_local_ref_by_local_id[term.from_set_ref],
                index.fact_local_ref_by_local_id[term.to_set_ref],
            )
        elif isinstance(term, FactTerm):
            kind = SemanticRecallRequirementKind.FACT
            owners = (index.fact_local_ref_by_local_id[term.owner_ref],)
            if isinstance(term.value_type, IdentifierType):
                identity = index.fact_local_ref_by_local_id[term.value_type.set_ref]
        else:
            raise TypeError(f"unknown semantic term {type(term).__name__}")
        output.append(
            SemanticRecallRequirement(
                requirement_ref=requirement_ref,
                kind=kind,
                origin=term.origin,
                owner_refs=owners,
                identity_set_ref=identity,
                use_site_refs=_term_use_site_refs(requirement_ref, index=index),
            )
        )
    return tuple(output)


def semantic_recall_buckets(
    index: RequestedFactSemanticIndex,
) -> tuple[SemanticRecallBucket, ...]:
    population_refs = _population_requirement_refs(index)
    buckets = [
        SemanticRecallBucket(
            bucket_ref=f"{index.requested_fact_id}:recall:population",
            kind=SemanticRecallBucketKind.POPULATION,
            origin=index.term_by_ref[index.subject_obligation.subject_set_ref].origin,
            requirement_refs=_sorted_refs(population_refs),
        )
    ]
    ordering_source_refs = frozenset(
        ref
        for ordering_ref in index.ordering_refs
        for ref in (
            {ordering_ref}
            if ordering_ref in index.source_requirement_refs
            else index.transitive_dependencies_by_ref.get(
                ordering_ref, frozenset()
            )
        )
        if isinstance(ref, FactLocalRef) and ref in index.source_requirement_refs
    )
    ordering_refs = index.source_support_closure(ordering_source_refs)
    for output, requested_output in zip(
        index.output_requirements,
        index.requested_fact.outputs,
        strict=True,
    ):
        output_refs = frozenset(
            ref
            for ref in output.dependencies
            if isinstance(ref, FactLocalRef)
            and ref in index.source_requirement_refs
        )
        buckets.append(
            SemanticRecallBucket(
                bucket_ref=f"{index.requested_fact_id}:recall:{output.output_ref.local_id}",
                kind=SemanticRecallBucketKind.OUTPUT,
                origin=requested_output.origin,
                requirement_refs=_sorted_refs(
                    frozenset(
                        {
                            *population_refs,
                            *ordering_refs,
                            *index.source_support_closure(output_refs),
                        }
                    )
                ),
            )
        )
    return tuple(buckets)


def _population_requirement_refs(
    index: RequestedFactSemanticIndex,
) -> frozenset[FactLocalRef]:
    refs = {index.subject_obligation.subject_set_ref}
    qualification_ref = index.requested_fact.qualification_ref
    if qualification_ref is not None:
        expression_ref = index.fact_local_ref_by_local_id[qualification_ref]
        refs.update(
            ref
            for ref in index.transitive_dependencies_by_ref.get(
                expression_ref, frozenset()
            )
            if isinstance(ref, FactLocalRef)
            and ref in index.source_requirement_refs
        )
    return index.source_support_closure(frozenset(refs))


def _sorted_refs(refs: frozenset[FactLocalRef]) -> tuple[FactLocalRef, ...]:
    return tuple(sorted(refs, key=lambda ref: ref.token))


def reference_input_recall_tasks(
    indexes: tuple[RequestedFactSemanticIndex, ...],
    *,
    inputs: Mapping[str, InputTerm],
) -> tuple[ReferenceInputRecallTask, ...]:
    return tuple(
        ReferenceInputRecallTask(
            input_use_ref=use.use_ref,
            input_ref=use.input_ref,
            input_origin=inputs[use.input_ref].origin,
            operand_meaning=use.operand_meaning,
            reference_fact_ref=use.reference_fact_ref,
            expected_set_ref=use.identity_set_ref,
        )
        for index in indexes
        for use in index.input_use_sites
        if use.reference_fact_ref is not None
    )


def validate_semantic_query_enrichment_result(
    result: SemanticQueryEnrichmentResult,
    *,
    recall_buckets: tuple[SemanticRecallBucket, ...],
    reference_tasks: tuple[ReferenceInputRecallTask, ...],
    resource_names: tuple[str, ...],
) -> SemanticQueryEnrichmentResult:
    required_refs = tuple(item.bucket_ref for item in recall_buckets)
    actual_refs = tuple(item.bucket_ref for item in result.recall_bucket_matches)
    if len(set(actual_refs)) != len(actual_refs) or set(actual_refs) != set(required_refs):
        raise ValueError("query enrichment must cover every recall bucket once")
    use_refs = tuple(item.input_use_ref for item in reference_tasks)
    actual_uses = tuple(item.input_use_ref for item in result.input_resource_search_terms)
    if len(set(actual_uses)) != len(actual_uses) or set(actual_uses) != set(use_refs):
        raise ValueError("query enrichment must cover every reference input use once")
    allowed_resources = frozenset(resource_names)
    for item in result.recall_bucket_matches:
        if any(
            name not in allowed_resources
            for name in (*item.exhaustive_resource_names, *item.matching_resource_names)
        ):
            raise ValueError("query enrichment invented a resource name")
        if not set(item.matching_resource_names) <= set(
            item.exhaustive_resource_names
        ):
            raise ValueError(
                "matching resource names must come from exhaustive resource names"
            )
    return result


def requirements_by_ref(
    requirements: tuple[SemanticRecallRequirement, ...],
) -> Mapping[FactLocalRef, SemanticRecallRequirement]:
    return MappingProxyType({item.requirement_ref: item for item in requirements})


def _term_use_site_refs(
    requirement_ref: FactLocalRef,
    *,
    index: RequestedFactSemanticIndex,
) -> tuple[str, ...]:
    expression_refs = {
        expression_ref
        for expression_ref, dependencies in index.transitive_dependencies_by_ref.items()
        if requirement_ref in dependencies or expression_ref == requirement_ref
    }
    return tuple(
        use.use_ref
        for use in index.input_use_sites
        if use.expression_ref in expression_refs
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
