"""Terminal source-binding outcome parsing."""

from __future__ import annotations

from typing import Any

from fervis.lookup.fact_planning.blocked_evidence import (
    canonical_blocked_evidence_refs,
    source_binding_evidence_refs,
)
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    BlockedFactField,
    MissingCatalogChoiceInput,
    MissingCatalogInput,
    MissingCatalogInputKind,
    MissingCatalogRequiredInput,
    PlanClarification,
    PlanImpossible,
)
from fervis.lookup.source_binding.candidates import source_candidates
from fervis.lookup.source_binding.model import SourceBindingRequest

from .parser_common import _dicts, _required_dicts, _required_strings, _strings, _text


def _plan_clarification(payload: dict[str, Any]) -> PlanClarification:
    return PlanClarification(
        missing_catalog_inputs=tuple(
            _missing_catalog_input(item)
            for item in _required_dicts(
                payload.get("missing_catalog_inputs"),
                "missing_catalog_inputs",
            )
        )
    )


def _missing_catalog_input(payload: dict[str, Any]) -> MissingCatalogInput:
    kind = MissingCatalogInputKind(_text(payload.get("kind")))
    if kind == MissingCatalogInputKind.REQUIRED_INPUT:
        return MissingCatalogRequiredInput(
            id=_text(payload.get("id")),
            requested_fact_id=_text(payload.get("requested_fact_id")),
            required_catalog_input_id=_text(payload.get("required_catalog_input_id")),
        )
    if kind == MissingCatalogInputKind.CHOICE_INPUT:
        return MissingCatalogChoiceInput(
            id=_text(payload.get("id")),
            requested_fact_id=_text(payload.get("requested_fact_id")),
            required_catalog_choice_input_id=_text(
                payload.get("required_catalog_choice_input_id")
            ),
        )
    raise ValueError(f"unsupported missing catalog input kind: {kind.value}")


def _plan_impossible(
    payload: dict[str, Any],
    *,
    request: SourceBindingRequest,
) -> PlanImpossible:
    evidence_resolver = _source_evidence_resolver(request)
    requested_fact_ids = tuple(fact.id for fact in request.requested_facts)
    return PlanImpossible(
        blocked_facts=tuple(
            _blocked_fact(
                item,
                evidence_resolver=evidence_resolver,
                requested_fact_ids=requested_fact_ids,
            )
            for item in _required_dicts(payload.get("blocked_facts"), "blocked_facts")
        )
    )


def _blocked_fact(
    payload: dict[str, Any],
    *,
    evidence_resolver: dict[str, tuple[str, ...]],
    requested_fact_ids: tuple[str, ...],
) -> BlockedFact:
    requested_fact_id = _text(payload.get("requested_fact_id"))
    if requested_fact_id not in set(requested_fact_ids):
        raise ValueError("blocked fact references unknown requested fact")
    return BlockedFact(
        requested_fact_id=requested_fact_id,
        basis=BlockedFactBasis(_text(payload.get("basis"))),
        evidence_refs=_canonical_impossible_evidence_refs(
            payload.get("evidence_refs"),
            evidence_resolver=evidence_resolver,
            requested_fact_ids=requested_fact_ids,
        ),
        reviewed_read_ids=tuple(_strings(payload.get("reviewed_read_ids"))),
        nearest_fields=tuple(
            _blocked_fact_field(item) for item in _dicts(payload.get("nearest_fields"))
        ),
        explanation=str(payload.get("explanation") or "").strip(),
    )


def _canonical_impossible_evidence_refs(
    raw_evidence_refs: Any,
    *,
    evidence_resolver: dict[str, tuple[str, ...]],
    requested_fact_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return canonical_blocked_evidence_refs(
        _required_strings(raw_evidence_refs, "evidence_refs"),
        source_evidence_refs=evidence_resolver,
        requested_fact_ids=requested_fact_ids,
    )


def _source_evidence_resolver(
    request: SourceBindingRequest,
) -> dict[str, tuple[str, ...]]:
    return source_binding_evidence_refs(
        source_candidates(request).values(),
        relation_catalog=request.relation_catalog,
    )


def _blocked_fact_field(payload: dict[str, Any]) -> BlockedFactField:
    return BlockedFactField(
        read_id=_text(payload.get("read_id")),
        field_id=_text(payload.get("field_id")),
    )
