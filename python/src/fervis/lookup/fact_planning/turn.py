"""Planner model turn for typed Lookup fact plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.fact_plan.fact_plan import FactPlan
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
)
from fervis.lookup.model_turn import (
    LookupModelTurnError,
    ModelTurnGenerationFailure,
    generation_error_kwargs,
    run_one_of_tool_model_turn,
)
from fervis.lookup.fact_planning.parser import parse_fact_plan
from fervis.lookup.fact_planning.request import (
    FactPlanRequest,
    PatternFactPlanTurnPrompt,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.plan_selection import BoundPlanSelectionSet
from fervis.lookup.fact_planning.grouped_aggregate_choices import (
    GROUPED_AGGREGATE_PLAN_SHAPES,
)
from fervis.lookup.answer_program.compiler_inputs import compiler_input_context


@dataclass(frozen=True)
class FactPlanTurnResult:
    plan: FactPlan
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class FactPlanGenerationError(LookupModelTurnError):
    pass


def generate_pattern_fact_plan(
    *,
    request: FactPlanRequest,
    plan_selection: BoundPlanSelectionSet,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> FactPlanTurnResult:
    invocation = PatternFactPlanTurnPrompt(
        request,
        plan_selection=plan_selection,
    ).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
        )
    )
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message="pattern fact plan prompt budget exceeded",
            model_error_message="pattern fact plan model turn failed",
        )
    except ModelTurnGenerationFailure as exc:
        raise FactPlanGenerationError(**generation_error_kwargs(exc)) from exc
    try:
        plan_payload = _with_selected_plan_shapes(output.arguments, plan_selection)
        plan = parse_fact_plan(
            plan_payload,
            bound_sources=request.bound_sources,
            source_binding_ids_by_requested_fact_id={
                plan.requested_fact_id: plan.source_binding_ids
                for plan in plan_selection.plan_selections
            },
            source_binding_ids_by_requirement_by_requested_fact_id=(
                plan_selection.source_binding_ids_by_requirement_by_requested_fact_id()
            ),
            relation_catalog=request.relation_catalog,
            question_contract=request.question_contract,
            memory_relations=request.memory_relations,
            input_context=compiler_input_context(
                values=request.available_values,
                question_contract=request.question_contract,
            ),
            selected_source_strategy_ids=tuple(
                plan.source_strategy_id for plan in plan_selection.plan_selections
            ),
        )
    except Exception as exc:
        raise FactPlanGenerationError(
            message=f"pattern fact plan parse failed: {exc}",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=output.artifact,
            error_context={"message": str(exc)},
        ) from exc
    return FactPlanTurnResult(
        plan=plan,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=output.artifact,
    )


def _with_selected_plan_shapes(
    payload: dict[str, Any],
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, Any]:
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict) or not isinstance(outcome.get("answers"), list):
        return payload
    normalized_answers: list[Any] = []
    covered_answer_output_ids_by_fact: dict[str, set[str]] = {}
    required_answer_output_ids_by_fact: dict[str, tuple[str, ...]] = {}
    for answer in outcome["answers"]:
        if not isinstance(answer, dict):
            normalized_answers.append(answer)
            continue
        requested_fact_id = str(answer.get("requested_fact_id") or "")
        selected_shapes = plan_selection.plan_shapes_for(requested_fact_id)
        if not selected_shapes:
            raise ValueError("fact plan has no aligned plan shapes")
        authored_shape = str(answer.get("pattern") or "")
        if authored_shape and authored_shape not in selected_shapes:
            raise ValueError("pattern fact plan changed selected plan shape")
        if authored_shape:
            selected_shape = authored_shape
        elif len(selected_shapes) == 1:
            selected_shape = selected_shapes[0]
        else:
            raise ValueError("fact plan must choose a pattern from aligned plan shapes")
        normalized_answer = {**answer, "pattern": selected_shape}
        selected_answer_output_ids = plan_selection.required_answer_output_ids_for(
            requested_fact_id
        )
        required_answer_output_ids_by_fact[requested_fact_id] = (
            selected_answer_output_ids
        )
        authored_answer_output_ids = tuple(
            str(item) for item in normalized_answer.get("answer_output_ids") or ()
        )
        if selected_shape in GROUPED_AGGREGATE_PLAN_SHAPES:
            if authored_answer_output_ids:
                raise ValueError(
                    "grouped aggregate fact plan authored backend-selected outputs"
                )
            covered_answer_output_ids_by_fact.setdefault(
                requested_fact_id,
                set(),
            ).update(selected_answer_output_ids)
        else:
            selected_answer_output_id_set = set(selected_answer_output_ids)
            if (
                authored_answer_output_ids
                and not set(authored_answer_output_ids) <= selected_answer_output_id_set
            ):
                raise ValueError(
                    "fact plan authored answer outputs outside selected plan"
                )
            effective_answer_output_ids = (
                authored_answer_output_ids or selected_answer_output_ids
            )
            normalized_answer["answer_output_ids"] = list(effective_answer_output_ids)
            covered_answer_output_ids_by_fact.setdefault(
                requested_fact_id, set()
            ).update(effective_answer_output_ids)
        normalized_answers.append(normalized_answer)
    for (
        requested_fact_id,
        required_answer_output_ids,
    ) in required_answer_output_ids_by_fact.items():
        required = set(required_answer_output_ids)
        covered = covered_answer_output_ids_by_fact.get(requested_fact_id, set())
        if required and covered and not required <= covered:
            raise ValueError("fact plan does not cover selected answer outputs")
    return {
        **payload,
        "outcome": {
            **outcome,
            "answers": normalized_answers,
        },
    }
