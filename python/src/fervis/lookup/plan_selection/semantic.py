"""Fact-local source alignment and strategy selection contract."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.available_sources import AvailableSourceCatalog
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.answer_program.values import (
    IdentitySetValuePayload,
    IdentityValuePayload,
)
from fervis.lookup.question_contract import RequestedFactSemanticIndex
from fervis.types.enums import StrEnum


class SourceAlignment(StrEnum):
    DIRECT = "DIRECT"
    PARTIAL = "PARTIAL"
    NOT_ALIGNED = "NOT_ALIGNED"


@dataclass(frozen=True)
class SourceAlignmentAssessment:
    source_ref: str
    basis: str
    alignment: SourceAlignment


@dataclass(frozen=True)
class SourceStrategyBranch:
    branch_id: str
    source_refs: tuple[str, ...]
    relation_evidence_refs: tuple[str, ...]
    qualification_clause_refs: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSourceStrategy:
    requested_fact_id: str
    source_assessments: tuple[SourceAlignmentAssessment, ...]
    strategy_basis: str
    branches: tuple[SourceStrategyBranch, ...]


@dataclass(frozen=True)
class SemanticPlanSelectionRequest:
    indexes: tuple[RequestedFactSemanticIndex, ...]
    source_catalog: AvailableSourceCatalog
    canonical_values: tuple[CanonicalInputValue, ...] = ()

    def allowed_alignments(
        self,
        *,
        requested_fact_id: str,
        source_ref: str,
    ) -> tuple[SourceAlignment, ...]:
        index = next(
            item
            for item in self.indexes
            if item.requested_fact_id == requested_fact_id
        )
        use_refs = {item.use_ref for item in index.input_use_sites}
        required_contracts = {
            (payload.entity_kind, payload.key_id)
            for value in self.canonical_values
            if use_refs.intersection(value.use_refs)
            for payload in (value.typed_value.payload,)
            if isinstance(payload, (IdentityValuePayload, IdentitySetValuePayload))
        }
        if not required_contracts:
            return tuple(SourceAlignment)
        source = self.source_catalog.source(source_ref)
        source_contracts = {
            (evidence.entity_kind, evidence.key_id)
            for evidence in source.identity_evidence
        } | {
            (param.entity_target.entity_kind, param.entity_target.key_id)
            for param in source.params
            if param.entity_target is not None
        }
        if required_contracts <= source_contracts:
            return tuple(SourceAlignment)
        return (SourceAlignment.PARTIAL, SourceAlignment.NOT_ALIGNED)


__all__ = tuple(name for name in globals() if not name.startswith("_"))
