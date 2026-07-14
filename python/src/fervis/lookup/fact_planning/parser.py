"""Parse provider-authored fact plans into typed planning models."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypeVar
from fervis.lookup.provider_contract import ProviderObject

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
    payload: dict[str, object],
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
    selected_source_strategy_ids: tuple[str, ...] = (),
) -> FactPlan:
    requested_fact_ids = tuple(fact.id for fact in question_contract.requested_facts)
    evidence_resolver = _bound_source_evidence_resolver(
        bound_sources,
        relation_catalog=relation_catalog,
    )
    output = provider_output.FactPlanOutput.parse(payload)
    outcome = output.outcome
    kind = _enum(
        PlanOutcomeKind,
        outcome.discriminator("kind"),
        "outcome.kind",
    )
    if kind == PlanOutcomeKind.FACT_PLAN:
        answer_output = outcome.parse_as(provider_output.FactPlanAnswerOutput)
        if relation_catalog is None:
            raise ValueError("answer-program compilation requires relation catalog")
        draft, draft_bindings = _answer_plan(
            answer_output,
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
        clarification = outcome.parse_as(provider_output.PlanClarificationOutput)
        return FactPlan(
            outcome=PlanClarification(
                missing_catalog_inputs=tuple(
                    _missing_catalog_input(item)
                    for item in clarification.missing_catalog_inputs
                )
            )
        )
    if kind == PlanOutcomeKind.IMPOSSIBLE:
        impossible = outcome.parse_as(provider_output.PlanImpossibleOutput)
        return FactPlan(
            outcome=PlanImpossible(
                blocked_facts=tuple(
                    _blocked_fact(
                        item,
                        evidence_resolver=evidence_resolver,
                        requested_fact_ids=requested_fact_ids,
                        selected_source_strategy_ids=selected_source_strategy_ids,
                    )
                    for item in impossible.blocked_facts
                )
            )
        )
    raise ValueError(f"unsupported fact plan outcome: {kind.value}")


def _answer_plan(
    payload: provider_output.FactPlanAnswerOutput,
    *,
    bound_sources: tuple[BoundSource, ...],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    input_context: CompilerInputContext,
) -> tuple[AnswerProgram, BindingSet]:
    answers = tuple(
        provider_output.parse_pattern_answer(answer) for answer in payload.answers
    )
    return compile_pattern_answer_program(
        answers,
        bound_sources=bound_sources,
        source_binding_ids_by_requested_fact_id=(
            source_binding_ids_by_requested_fact_id
        ),
        source_binding_ids_by_requirement_by_requested_fact_id=(
            source_binding_ids_by_requirement_by_requested_fact_id
        ),
        input_context=input_context,
    )


def _blocked_fact(
    payload: provider_output.BlockedFactOutput,
    *,
    evidence_resolver: dict[str, tuple[str, ...]],
    requested_fact_ids: tuple[str, ...],
    selected_source_strategy_ids: tuple[str, ...],
) -> BlockedFact:
    basis = _enum(BlockedFactBasis, payload.basis, "blocked_fact.basis")
    requested_fact_id = _text(payload.requested_fact_id)
    if requested_fact_ids and requested_fact_id not in set(requested_fact_ids):
        raise ValueError("blocked fact references unknown requested fact")
    return BlockedFact(
        requested_fact_id=requested_fact_id,
        basis=basis,
        evidence_refs=canonical_blocked_evidence_refs(
            payload.evidence_refs,
            source_evidence_refs=evidence_resolver,
            requested_fact_ids=requested_fact_ids,
            non_catalog_evidence_refs=selected_source_strategy_ids,
        ),
        reviewed_read_ids=payload.reviewed_read_ids or (),
        nearest_fields=tuple(
            _blocked_fact_field(item) for item in payload.nearest_fields or ()
        ),
        explanation=(payload.explanation or "").strip(),
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


def _blocked_fact_field(
    payload: provider_output.BlockedFactFieldOutput,
) -> BlockedFactField:
    return BlockedFactField(
        read_id=_text(payload.read_id),
        field_id=_text(payload.field_id),
    )


def _missing_catalog_input(raw: ProviderObject) -> MissingCatalogInput:
    kind = _enum(
        MissingCatalogInputKind,
        raw.discriminator("kind"),
        "missing_catalog_input.kind",
    )
    if kind == MissingCatalogInputKind.REQUIRED_INPUT:
        required_input = raw.parse_as(provider_output.MissingCatalogRequiredInputOutput)
        return MissingCatalogRequiredInput(
            id=_text(required_input.id),
            requested_fact_id=_text(required_input.requested_fact_id),
            required_catalog_input_id=_text(required_input.required_catalog_input_id),
        )
    if kind == MissingCatalogInputKind.CHOICE_INPUT:
        choice_input = raw.parse_as(provider_output.MissingCatalogChoiceInputOutput)
        return MissingCatalogChoiceInput(
            id=_text(choice_input.id),
            requested_fact_id=_text(choice_input.requested_fact_id),
            required_catalog_choice_input_id=_text(
                choice_input.required_catalog_choice_input_id
            ),
        )
    raise ValueError(f"unsupported missing catalog input: {kind.value}")


_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum(enum_type: type[_EnumT], value: str, label: str) -> _EnumT:
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} has unsupported value: {value!r}") from exc


def _text(value: str) -> str:
    return value.strip()
