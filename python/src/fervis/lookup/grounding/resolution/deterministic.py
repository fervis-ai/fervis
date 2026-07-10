"""Deterministic grounding for time and literal inputs."""

from __future__ import annotations

from typing import Any

from fervis.lookup.grounding.time_resolution import Field as TimeField
from fervis.lookup.grounding.time_resolution import IntentField, Unit
from fervis.lookup.grounding.time_resolution import Status as TimeStatus
from fervis.lookup.grounding.time_resolution import resolve_time
from fervis.lookup.grounding.model import (
    CanonicalInputLedger,
    GroundedInputUse,
    GroundingIssue,
    GroundingTerminalKind,
    GroundingRequestedFactCard,
    KnownTimeResolutionTask,
    TimeResolutionIntent,
)
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.fact_plan.row_sources import (
    CALENDAR_END_PARAM_ID,
    CALENDAR_ROW_SOURCE_ID,
    CALENDAR_START_PARAM_ID,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    LiteralType,
    TimeComponent,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFactKnownInput,
)

from .values import _grounded_value_id


def _deterministic_known_inputs(
    question_contract: QuestionContract,
    *,
    runtime_values: RuntimeValueContext | None,
    active_time_anchor_periods: dict[str, dict[str, str]] | None = None,
) -> CanonicalInputLedger:
    values: list[FactValue] = []
    for known, requested_fact_ids in _known_input_bindings(question_contract):
        if known.is_time_value:
            continue
        if known.is_result_limit:
            literal = _result_limit_value(
                known,
                applies_to_requested_fact_ids=requested_fact_ids,
            )
            values.append(literal)
            continue
    return CanonicalInputLedger(
        values=tuple(values),
        issues=(),
    )


def time_resolution_tasks(
    question_contract: QuestionContract,
) -> tuple[KnownTimeResolutionTask, ...]:
    facts_by_id = {fact.id: fact for fact in question_contract.requested_facts}
    return tuple(
        KnownTimeResolutionTask(
            known_input_id=known.id,
            known_input_text=known.text,
            requested_fact_id=requested_fact_ids[0] if requested_fact_ids else "",
            time_expression=_time_expression(known),
            applies_to_requested_fact_ids=requested_fact_ids,
            requested_facts=tuple(
                _requested_fact_card(facts_by_id[fact_id])
                for fact_id in requested_fact_ids
                if fact_id in facts_by_id
            ),
        )
        for known, requested_fact_ids in _known_input_bindings(question_contract)
        if known.is_time_value
    )


def resolve_time_resolutions(
    resolutions: tuple[TimeResolutionIntent, ...],
    *,
    question_contract: QuestionContract,
    runtime_values: RuntimeValueContext | None,
    active_time_anchor_periods: dict[str, dict[str, str]] | None = None,
) -> CanonicalInputLedger:
    values: list[FactValue] = []
    uses: list[GroundedInputUse] = []
    issues: list[GroundingIssue] = []
    inputs_by_id = {
        known.id: (known, fact_ids)
        for known, fact_ids in _known_input_bindings(question_contract)
    }
    intents_by_id = {
        resolution.known_input_id: resolution for resolution in resolutions
    }
    anchor_periods = tuple((active_time_anchor_periods or {}).values())
    for known_input_id, resolution in intents_by_id.items():
        known, requested_fact_ids = inputs_by_id[known_input_id]
        requested_fact_id = requested_fact_ids[0] if requested_fact_ids else ""
        time_value = _ground_time_value(
            known,
            date_intent=resolution.date_intent,
            requested_fact_id=requested_fact_id,
            applies_to_requested_fact_ids=requested_fact_ids,
            runtime_values=runtime_values,
            active_time_anchor_periods=anchor_periods,
        )
        if isinstance(time_value, GroundingIssue):
            issues.append(time_value)
            continue
        values.append(time_value)
        uses.extend(_calendar_param_uses(time_value))
    return CanonicalInputLedger(
        values=tuple(values),
        uses=tuple(uses),
        issues=tuple(issues),
    )


def _calendar_param_uses(value: FactValue) -> tuple[GroundedInputUse, ...]:
    return (
        GroundedInputUse(
            id=f"use_{value.id}_calendar_start",
            value_id=value.id,
            row_source_id=CALENDAR_ROW_SOURCE_ID,
            param_id=CALENDAR_START_PARAM_ID,
            value_component=TimeComponent.START,
        ),
        GroundedInputUse(
            id=f"use_{value.id}_calendar_end",
            value_id=value.id,
            row_source_id=CALENDAR_ROW_SOURCE_ID,
            param_id=CALENDAR_END_PARAM_ID,
            value_component=TimeComponent.END,
        ),
    )


def _ground_time_value(
    known: RequestedFactKnownInput,
    *,
    date_intent: dict[str, object],
    requested_fact_id: str,
    applies_to_requested_fact_ids: tuple[str, ...],
    runtime_values: RuntimeValueContext | None,
    active_time_anchor_periods: tuple[dict[str, str], ...],
) -> FactValue | GroundingIssue:
    if runtime_values is None:
        return GroundingIssue(
            kind=GroundingTerminalKind.TIME_RESOLUTION_FAILED,
            known_input_id=known.id,
            requested_fact_id=requested_fact_id,
            message="runtime anchors are required to resolve time input",
            proof_refs=(f"known_input:{known.id}",),
        )
    resolved = _resolve_time(
        known,
        date_intent=date_intent,
        runtime_values=runtime_values,
        active_time_anchor_periods=active_time_anchor_periods,
    )
    if resolved.get(TimeField.STATUS) != TimeStatus.RESOLVED:
        return GroundingIssue(
            kind=GroundingTerminalKind.TIME_RESOLUTION_FAILED,
            known_input_id=known.id,
            requested_fact_id=requested_fact_id,
            message="time input could not be resolved",
            proof_refs=(f"known_input:{known.id}",),
        )
    return FactValue.time(
        id=_grounded_value_id(known.id),
        known_input_id=known.id,
        expression=_time_expression(known),
        intent=dict(resolved.get(TimeField.INTENT) or {}),
        resolved_start=str(resolved.get(TimeField.START) or ""),
        resolved_end=str(resolved.get(TimeField.END) or ""),
        granularity=_time_granularity(resolved),
        proof_refs=(f"known_input:{known.id}",),
        applies_to_requested_fact_ids=applies_to_requested_fact_ids,
    )


def _resolve_time(
    known: RequestedFactKnownInput,
    *,
    date_intent: dict[str, object],
    runtime_values: RuntimeValueContext,
    active_time_anchor_periods: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    try:
        resolved = resolve_time(
            _time_expression(known),
            intent=date_intent,
            anchor_date=runtime_values.runtime_date,
            timezone=runtime_values.timezone,
        )
        if (
            resolved.get(TimeField.STATUS) != TimeStatus.NEEDS_CLARIFICATION
            or len(active_time_anchor_periods) != 1
        ):
            return resolved
        return resolve_time(
            _time_expression(known),
            anchor_date=runtime_values.runtime_date,
            timezone=runtime_values.timezone,
            intent={
                **date_intent,
                IntentField.ANCHOR_PERIOD: active_time_anchor_periods[0],
            },
        )
    except ValueError:
        return {}


def _time_expression(known: RequestedFactKnownInput) -> str:
    return known.resolved_value_text


def time_anchor_period_from_memory_address(address: Any) -> dict[str, str] | None:
    scalar_value = getattr(address, "scalar_value", {}) or {}
    if str(scalar_value.get("type") or "") != "time_scope":
        return None
    start = str(scalar_value.get("resolvedStart") or "").strip()
    end = str(scalar_value.get("resolvedEnd") or "").strip()
    if not start or not end:
        return None
    return {
        IntentField.UNIT: str(scalar_value.get("granularity") or Unit.DAY),
        IntentField.START_DATE: start,
        IntentField.END_DATE: end,
    }


def _time_granularity(resolved: dict[str, Any]) -> str:
    intent = resolved.get(TimeField.INTENT)
    if not isinstance(intent, dict):
        return ""
    return str(
        intent.get("precision")
        or intent.get("unit")
        or (intent.get("anchor_period") or {}).get("unit")
        or ""
    )


def _result_limit_value(
    known: RequestedFactKnownInput,
    *,
    applies_to_requested_fact_ids: tuple[str, ...],
) -> FactValue:
    return FactValue.literal(
        id=_grounded_value_id(known.id),
        known_input_id=known.id,
        literal_type=LiteralType.NUMBER,
        value=known.resolved_value_text,
        label=known.text,
        proof_refs=(f"known_input:{known.id}",),
        applies_to_requested_fact_ids=applies_to_requested_fact_ids,
    )


def _known_input_bindings(
    question_contract: QuestionContract,
) -> tuple[tuple[RequestedFactKnownInput, tuple[str, ...]], ...]:
    if question_contract.question_inputs:
        return tuple(
            (
                known,
                question_contract.requested_fact_ids_for_input(known.id),
            )
            for known in question_contract.question_inputs
        )
    return tuple(
        (
            known,
            (fact.id,),
        )
        for fact in question_contract.requested_facts
        for known in fact.known_inputs
    )


def _requested_fact_card(fact: Any) -> GroundingRequestedFactCard:
    answer_population = getattr(fact, "answer_population", None)
    return GroundingRequestedFactCard(
        requested_fact_id=fact.id,
        answer_fact=fact.description,
        answer_population_label=(
            answer_population.population_label if answer_population is not None else ""
        ),
        answer_population_counted_unit=(
            answer_population.counted_unit if answer_population is not None else ""
        ),
        answer_outputs=tuple(
            {
                "id": output.id,
                "description": output.description,
            }
            for output in fact.answer_outputs
        ),
    )
