"""Parse provider-authored fact plans into typed planning models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.fact_planning.blocked_evidence import (
    bound_source_evidence_refs,
    canonical_blocked_evidence_refs,
)
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    BlockedFactField,
    FactPlan,
    MissingCatalogChoiceInput,
    MissingCatalogInput,
    MissingCatalogInputKind,
    MissingCatalogRequiredInput,
    PlanClarification,
    PlanImpossible,
    PlanOutcomeKind,
)
from fervis.lookup.fact_planning import provider_contract as provider_output
from fervis.lookup.fact_planning.pattern_plan import compile_pattern_answer_program
from fervis.lookup.source_binding import BoundSource
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.compilation import compile_answer_program
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.plan_execution.relations import RelationRows


def parse_fact_plan(
    payload: dict[str, Any],
    *,
    bound_sources: tuple[BoundSource, ...] = (),
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    relation_catalog: RelationCatalog | None = None,
    question_contract: QuestionContract,
    memory_relations: tuple[RelationRows, ...] = (),
    input_context: CompilerInputContext,
) -> FactPlan:
    requested_fact_ids = tuple(fact.id for fact in question_contract.requested_facts)
    evidence_resolver = _bound_source_evidence_resolver(
        bound_sources,
        relation_catalog=relation_catalog,
    )
    output = provider_output.FactPlanOutput.parse(payload)
    outcome = _dict(output.outcome, "outcome")
    kind = _enum(PlanOutcomeKind, outcome.get("kind"), "outcome.kind")
    if kind == PlanOutcomeKind.FACT_PLAN:
        provider_output.FactPlanAnswerOutput.parse(outcome)
        if relation_catalog is None:
            raise ValueError("answer-program compilation requires relation catalog")
        draft, draft_bindings = _answer_plan(
                outcome,
                bound_sources=bound_sources,
                source_binding_ids_by_requested_fact_id=(
                    source_binding_ids_by_requested_fact_id
                ),
                source_binding_ids_by_requirement_by_requested_fact_id=(
                    source_binding_ids_by_requirement_by_requested_fact_id
                ),
                input_context=input_context,
            )
        program, bindings = compile_answer_program(
            draft,
            question_contract=question_contract,
            catalog=relation_catalog,
            bindings=draft_bindings,
            memory_relations=memory_relations,
        )
        return FactPlan(
            outcome=program,
            bindings=bindings,
        )
    if kind == PlanOutcomeKind.NEEDS_CLARIFICATION:
        clarification = provider_output.PlanClarificationOutput.parse(outcome)
        return FactPlan(
            outcome=PlanClarification(
                missing_catalog_inputs=tuple(
                    _missing_catalog_input(item)
                    for item in _required_dicts(
                        clarification.missing_catalog_inputs,
                        "missing_catalog_inputs",
                    )
                )
            )
        )
    if kind == PlanOutcomeKind.IMPOSSIBLE:
        impossible = provider_output.PlanImpossibleOutput.parse(outcome)
        return FactPlan(
            outcome=PlanImpossible(
                blocked_facts=tuple(
                    _blocked_fact(
                        item,
                        evidence_resolver=evidence_resolver,
                        requested_fact_ids=requested_fact_ids,
                    )
                    for item in _required_dicts(
                        impossible.blocked_facts,
                        "blocked_facts",
                    )
                )
            )
        )
    raise ValueError(f"unsupported fact plan outcome: {kind.value}")


def _answer_plan(
    payload: dict[str, Any],
    *,
    bound_sources: tuple[BoundSource, ...],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    input_context: CompilerInputContext,
) -> tuple[AnswerProgram, BindingSet]:
    if "values" in payload:
        raise ValueError("fact plan values are not model-authored")
    if "answers" in payload:
        return compile_pattern_answer_program(
            payload,
            bound_sources=bound_sources,
            source_binding_ids_by_requested_fact_id=(
                source_binding_ids_by_requested_fact_id
            ),
                source_binding_ids_by_requirement_by_requested_fact_id=(
                    source_binding_ids_by_requirement_by_requested_fact_id
                ),
                input_context=input_context,
            )
    raise ValueError("fact plan answers must be a list")


def _blocked_fact(
    raw: dict[str, Any],
    *,
    evidence_resolver: dict[str, tuple[str, ...]],
    requested_fact_ids: tuple[str, ...],
) -> BlockedFact:
    payload = provider_output.BlockedFactOutput.parse(raw)
    basis = _enum(BlockedFactBasis, payload.basis, "blocked_fact.basis")
    requested_fact_id = _text(payload.requested_fact_id)
    if requested_fact_ids and requested_fact_id not in set(requested_fact_ids):
        raise ValueError("blocked fact references unknown requested fact")
    return BlockedFact(
        requested_fact_id=requested_fact_id,
        basis=basis,
        evidence_refs=canonical_blocked_evidence_refs(
            _required_strings(payload.evidence_refs, "evidence_refs"),
            source_evidence_refs=evidence_resolver,
            requested_fact_ids=requested_fact_ids,
        ),
        reviewed_read_ids=tuple(_strings(payload.reviewed_read_ids)),
        nearest_fields=tuple(
            _blocked_fact_field(item) for item in _dicts(payload.nearest_fields)
        ),
        explanation=_text(payload.explanation),
    )


def _bound_source_evidence_resolver(
    bound_sources: tuple[BoundSource, ...],
    *,
    relation_catalog: RelationCatalog | None,
) -> dict[str, tuple[str, ...]]:
    if relation_catalog is None:
        return {}
    return bound_source_evidence_refs(
        bound_sources,
        relation_catalog=relation_catalog,
    )


def _blocked_fact_field(raw: dict[str, Any]) -> BlockedFactField:
    payload = provider_output.BlockedFactFieldOutput.parse(raw)
    return BlockedFactField(
        read_id=_text(payload.read_id),
        field_id=_text(payload.field_id),
    )


def _missing_catalog_input(raw: dict[str, Any]) -> MissingCatalogInput:
    kind = _enum(
        MissingCatalogInputKind,
        raw.get("kind"),
        "missing_catalog_input.kind",
    )
    if kind == MissingCatalogInputKind.REQUIRED_INPUT:
        payload = provider_output.MissingCatalogRequiredInputOutput.parse(raw)
        return MissingCatalogRequiredInput(
            id=_text(payload.id),
            requested_fact_id=_text(payload.requested_fact_id),
            required_catalog_input_id=_text(payload.required_catalog_input_id),
        )
    if kind == MissingCatalogInputKind.CHOICE_INPUT:
        payload = provider_output.MissingCatalogChoiceInputOutput.parse(raw)
        return MissingCatalogChoiceInput(
            id=_text(payload.id),
            requested_fact_id=_text(payload.requested_fact_id),
            required_catalog_choice_input_id=_text(
                payload.required_catalog_choice_input_id
            ),
        )
    raise ValueError(f"unsupported missing catalog input: {kind.value}")


def _enum(enum_type: type, value: Any, label: str):
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} has unsupported value: {value!r}") from exc


def _dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("expected list of objects")
    return tuple(_dict(item, "list item") for item in value)


def _required_dicts(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(_dict(item, label) for item in value)


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("expected list of strings")
    return tuple(_text(item) for item in value)


def _required_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(_text(item) for item in value)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)
