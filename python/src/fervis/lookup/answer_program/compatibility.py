"""Exact compatibility manifests for immutable answer programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import assert_never

from fervis.lookup.answer_program.codec import canonical_contract_fingerprint
from fervis.lookup.answer_program.model import (
    ANSWER_PROGRAM_SCHEMA_REVISION,
    AnswerProgram,
    FunctionSemanticVersion,
    ProgramCompatibility,
    SourceContractKind,
    SourceContractPin,
)
from fervis.lookup.answer_program.relations import (
    PopulationChoiceControllerKind,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources.builder import (
    build_row_source_catalog,
    memory_row_source_id,
)
from fervis.lookup.fact_plan.row_sources.model import (
    CALENDAR_ROW_SOURCE_ID,
    RowSourceCatalog,
)
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.relation_catalog import RelationCatalog


ANSWER_PROGRAM_COMPILER_VERSION = "fervis.answer_program.compiler@1"
FUNCTION_SEMANTIC_VERSION = "1"


@dataclass(frozen=True)
class SourceContractKey:
    kind: SourceContractKind
    source_id: str


@dataclass(frozen=True)
class CompatibilityRequirements:
    function_keys: tuple[str, ...]
    source_keys: tuple[SourceContractKey, ...]


def compatibility_requirements(
    program: AnswerProgram,
) -> CompatibilityRequirements:
    """Derive the exact compatibility surface from the parsed program graph."""

    function_keys = {
        "answer.classify",
        "answer.materialize",
        "fact.materialize",
        "proof.construct",
    }
    function_keys.update(
        f"relation.{operation.spec.kind.value}" for operation in program.operations
    )
    if any(
        relation.source.kind in {SourceKind.API_READ, SourceKind.GENERATED_CALENDAR}
        for relation in program.relations
    ):
        function_keys.update(("source.instantiate_read", "source.read_relation"))
    if program.capabilities or _uses_row_filtering(program):
        function_keys.add("relation.row_filter")
    if program.capabilities:
        function_keys.add("program.capability.narrow_population")

    return CompatibilityRequirements(
        function_keys=tuple(sorted(function_keys)),
        source_keys=tuple(
            sorted(
                {
                    _source_contract_key(relation.source)
                    for relation in program.relations
                },
                key=lambda item: (item.kind.value, item.source_id),
            )
        ),
    )


def build_program_compatibility(
    program: AnswerProgram,
    *,
    catalog: RelationCatalog,
    row_sources: RowSourceCatalog,
    memory_relations: tuple[RelationRows, ...],
) -> ProgramCompatibility:
    """Pin the exact current contracts required by a program graph."""

    requirements = compatibility_requirements(program)
    return ProgramCompatibility(
        schema_revision=ANSWER_PROGRAM_SCHEMA_REVISION,
        compiler_version=ANSWER_PROGRAM_COMPILER_VERSION,
        function_semantics=tuple(
            FunctionSemanticVersion(
                function_key=key,
                version=FUNCTION_SEMANTIC_VERSION,
            )
            for key in requirements.function_keys
        ),
        source_contracts=tuple(
            SourceContractPin(
                kind=key.kind,
                source_id=key.source_id,
                fingerprint=_current_source_fingerprint(
                    key,
                    catalog=catalog,
                    row_sources=row_sources,
                    memory_relations=memory_relations,
                ),
            )
            for key in requirements.source_keys
        ),
    )


def verify_program_compatibility(
    program: AnswerProgram,
    *,
    catalog: RelationCatalog,
    memory_relations: tuple[RelationRows, ...],
) -> None:
    """Reject any incomplete, stale, or contradictory compatibility manifest."""

    compatibility = program.compatibility
    if compatibility.schema_revision != ANSWER_PROGRAM_SCHEMA_REVISION:
        raise VerificationError("incompatible_program_schema")
    if compatibility.compiler_version != ANSWER_PROGRAM_COMPILER_VERSION:
        raise VerificationError("incompatible_compiler_version")

    requirements = compatibility_requirements(program)
    if not _is_exact_manifest(compatibility, requirements=requirements):
        raise VerificationError("incomplete_compatibility_manifest")
    if any(
        item.version != FUNCTION_SEMANTIC_VERSION
        for item in compatibility.function_semantics
    ):
        raise VerificationError("incompatible_function_semantics")

    row_sources = build_row_source_catalog(
        catalog,
        memory_relations=memory_relations,
    )
    for pin in compatibility.source_contracts:
        try:
            current_fingerprint = _current_source_fingerprint(
                SourceContractKey(kind=pin.kind, source_id=pin.source_id),
                catalog=catalog,
                row_sources=row_sources,
                memory_relations=memory_relations,
            )
        except (KeyError, StopIteration) as exc:
            raise VerificationError("incompatible_source_contract") from exc
        if current_fingerprint != pin.fingerprint:
            raise VerificationError("incompatible_source_contract")


def _is_exact_manifest(
    compatibility: ProgramCompatibility,
    *,
    requirements: CompatibilityRequirements,
) -> bool:
    function_keys = tuple(
        sorted(item.function_key for item in compatibility.function_semantics)
    )
    source_keys = tuple(
        sorted(
            (
                SourceContractKey(kind=item.kind, source_id=item.source_id)
                for item in compatibility.source_contracts
            ),
            key=lambda item: (item.kind.value, item.source_id),
        )
    )
    return (
        function_keys == requirements.function_keys
        and source_keys == requirements.source_keys
    )


def _uses_row_filtering(program: AnswerProgram) -> bool:
    return any(
        relation.source.row_filters
        or relation.source.applied_filters
        or any(
            choice.controller_kind is PopulationChoiceControllerKind.ROW_PREDICATE
            for choice in relation.source.population_choices
        )
        for relation in program.relations
    )


def _source_contract_key(source: RelationSource) -> SourceContractKey:
    if source.kind is SourceKind.API_READ:
        return SourceContractKey(
            kind=SourceContractKind.CATALOG_READ,
            source_id=source.read_id,
        )
    if source.kind is SourceKind.GENERATED_CALENDAR:
        return SourceContractKey(
            kind=SourceContractKind.GENERATED_SOURCE,
            source_id=CALENDAR_ROW_SOURCE_ID,
        )
    if source.kind is SourceKind.MEMORY_READ:
        return SourceContractKey(
            kind=SourceContractKind.MEMORY_RELATION,
            source_id=memory_row_source_id(source.memory_relation_id),
        )
    assert_never(source.kind)


def _current_source_fingerprint(
    key: SourceContractKey,
    *,
    catalog: RelationCatalog,
    row_sources: RowSourceCatalog,
    memory_relations: tuple[RelationRows, ...],
) -> str:
    if key.kind is SourceContractKind.CATALOG_READ:
        return canonical_contract_fingerprint(catalog.read(key.source_id))
    elif key.kind is SourceContractKind.GENERATED_SOURCE:
        return canonical_contract_fingerprint(row_sources.source(key.source_id))
    elif key.kind is SourceContractKind.MEMORY_RELATION:
        relation = next(
            relation
            for relation in memory_relations
            if memory_row_source_id(relation.id) == key.source_id
        )
        return canonical_contract_fingerprint(relation)
    else:
        assert_never(key.kind)
