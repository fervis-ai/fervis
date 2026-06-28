"""Planner model turn for typed Lookup fact plans."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from fervis.lookup.fact_plan.fact_plan import FactPlan
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
    model_turn_artifact,
)
from fervis.lookup.fact_planning.parser import parse_fact_plan
from fervis.lookup.fact_planning.request import (
    FactPlanRequest,
    PatternFactPlanTurnPrompt,
)
from fervis.lookup.errors import ErrorCode
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.plan_selection import BoundPlanSelectionSet
from fervis.lookup.fact_planning.grouped_ranked_choices import (
    GROUPED_RANKED_PLAN_SHAPES,
)
from fervis.model_io.structured_output.errors import RequiredToolOutputError
from fervis.model_io.structured_output.generation import (
    generate_one_of_tool_output,
)
from fervis.model_io.telemetry import (
    ModelTurnPromptBudgetError,
    enforce_model_turn_prompt_budget,
)


@dataclass(frozen=True)
class FactPlanTurnResult:
    plan: FactPlan
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


@dataclass(frozen=True)
class FactPlanGenerationError(Exception):
    message: str
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    error_code: str = ErrorCode.PLANNING_FAILED
    error_context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


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
    prompt = invocation.prompt_text
    system_prompt = invocation.system_prompt
    schema = invocation.provider_schema
    tool_specs = invocation.tool_specs
    try:
        enforce_model_turn_prompt_budget(prompt=prompt, tool_specs=tool_specs)
    except ModelTurnPromptBudgetError as exc:
        raise FactPlanGenerationError(
            message="pattern fact plan prompt budget exceeded",
            usage={},
            duration_ms=0,
            artifact=ModelTurnArtifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=schema,
                tool_specs=tool_specs,
                submitted_payload={},
            ),
        ) from exc
    started = time.monotonic()
    try:
        output = generate_one_of_tool_output(
            model_port=model_port,
            provider=provider,
            system_prompt=system_prompt,
            prompt=prompt,
            max_thinking_tokens=max_thinking_tokens,
            tool_specs=tool_specs,
        )
    except RequiredToolOutputError as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        raise FactPlanGenerationError(
            message="pattern fact plan model turn failed",
            usage=dict(exc.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=model_turn_artifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=schema,
                tool_specs=tool_specs,
                submitted_payload=exc.arguments,
                raw_output=exc.raw_output,
            ),
            error_code=exc.error_code or ErrorCode.PLANNING_FAILED,
            error_context=dict(exc.error_context or {}),
        ) from exc
    duration_ms = int((time.monotonic() - started) * 1000)
    artifact = model_turn_artifact(
        system_prompt=system_prompt,
        prompt_text=prompt,
        provider_schema=schema,
        tool_specs=tool_specs,
        submitted_payload=output.arguments,
        raw_output=output.raw_output,
        selected_tool_name=output.tool_spec.name,
    )
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
            requested_fact_ids=tuple(
                fact.id for fact in request.question_contract.requested_facts
            ),
        )
    except Exception as exc:
        raise FactPlanGenerationError(
            message=f"pattern fact plan parse failed: {exc}",
            usage=dict(output.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=artifact,
            error_context={"message": str(exc)},
        ) from exc
    return FactPlanTurnResult(
        plan=plan,
        usage=dict(output.output.get("usage") or {}),
        duration_ms=duration_ms,
        artifact=artifact,
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
        if selected_shape in GROUPED_RANKED_PLAN_SHAPES:
            if authored_answer_output_ids:
                raise ValueError(
                    "grouped/ranked fact plan authored backend-selected outputs"
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
