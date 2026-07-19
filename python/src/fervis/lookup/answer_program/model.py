"""Canonical deterministic answer-program graph."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum

from fervis.lookup.answer_program.operations import Operation
from fervis.lookup.answer_program.capability_contracts import (
    NarrowPopulationCapability,
)
from fervis.lookup.answer_program.relations import Relation
from fervis.lookup.answer_program.result_projection import ResultProjection
from fervis.lookup.answer_program.values import (
    ParameterDeclaration,
)
from fervis.lookup.question_contract import RequestedFact


ANSWER_PROGRAM_SCHEMA_REVISION = 6


@dataclass(frozen=True)
class FactFulfillment:
    requested_fact_id: str
    answer_output_id: str
    result_output_id: str


@dataclass(frozen=True)
class FunctionSemanticVersion:
    function_key: str
    version: str


class SourceContractKind(StrEnum):
    CATALOG_READ = "catalog_read"
    GENERATED_SOURCE = "generated_source"
    MEMORY_RELATION = "memory_relation"


@dataclass(frozen=True)
class SourceContractPin:
    kind: SourceContractKind
    source_id: str
    fingerprint: str


@dataclass(frozen=True)
class ProgramCompatibility:
    schema_revision: int = ANSWER_PROGRAM_SCHEMA_REVISION
    compiler_version: str = ""
    function_semantics: tuple[FunctionSemanticVersion, ...] = ()
    source_contracts: tuple[SourceContractPin, ...] = ()


@dataclass(frozen=True)
class AnswerProgram:
    fact_template: tuple[RequestedFact, ...] = ()
    fulfillment: tuple[FactFulfillment, ...] = ()
    parameters: tuple[ParameterDeclaration, ...] = ()
    capabilities: tuple[NarrowPopulationCapability, ...] = ()
    relations: tuple[Relation, ...] = ()
    operations: tuple[Operation, ...] = ()
    result_projection: ResultProjection = ResultProjection()
    compatibility: ProgramCompatibility = ProgramCompatibility()
