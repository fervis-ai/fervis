from __future__ import annotations

from typing import Any

from fervis.lookup.clarification import (
    clarification_from_payload,
    clarification_payload,
)
from fervis.lookup.plan_execution.operation_runtime import RelationEngineOutput
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.outcomes.model import (
    EmptyRelationKind,
    FactResult,
    NeedsClarification,
)
from fervis.lookup.outcomes.errors import ExecutionIssue
from fervis.lookup.outcomes.answerability import classify_plan_impossible
from fervis.lookup.outcomes.classification import (
    classify_answer_result,
    classify_binding_candidates,
    classify_empty_relation,
)
from fervis.lookup.fact_plan.fact_plan import (
    AnswerPlan,
    BlockedFact,
    BlockedFactBasis,
    BlockedFactField,
    FactFulfillment,
    PlanImpossible,
)
from fervis.lookup.fact_plan.render_spec import (
    RenderRelationOutput,
    RenderSpec,
)
from fervis.lookup.answer_rendering import (
    render_fact_result,
    rendered_fact_payload,
    rendered_fact_text,
)

from tests.testkit.algorithms.relation_engine import (
    engine_input_from_payload,
)
from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)
from tests.testkit.question_contract import question_contract_from_payload


def run_outcomes_classify_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    mode = str(input_payload.get("mode") or "answer")
    try:
        if mode == "engine":
            result = execute_operations(engine_input_from_payload(input_payload))
            rendered = None
            actual = _engine_output_payload(result)
            return subset_mismatches(
                actual=actual,
                expected_subset=payload["expect"]["result_contains"],
            )
        if mode == "answer":
            result = _classify_answer(input_payload)
        elif mode == "empty_relation":
            relation = engine_input_from_payload(input_payload).relations[0]
            result = classify_empty_relation(
                relation,
                kind=EmptyRelationKind(
                    str(input_payload.get("empty_kind") or "answer_rows")
                ),
            )
        elif mode == "binding_candidates":
            relation = engine_input_from_payload(input_payload).relations[0]
            result = classify_binding_candidates(
                requested_fact_id=str(input_payload["requested_fact_id"]),
                binding_target_id=str(input_payload["binding_target_id"]),
                known_input_id=str(input_payload["known_input_id"]),
                candidate_relation=relation,
                display_fields=tuple(
                    str(item) for item in input_payload.get("display_fields") or ()
                ),
            )
        elif mode == "impossible":
            result = classify_plan_impossible(
                _plan_impossible(input_payload["plan_impossible"]),
                question_contract=_question_contract(input_payload),
            )
        elif mode == "clarification":
            result = FactResult(
                outcome=NeedsClarification(
                    clarifications=tuple(
                        clarification_from_payload(item)
                        for item in input_payload.get("clarifications") or ()
                    )
                )
            )
        else:
            return [f"unsupported outcomes mode: {mode}"]
        rendered = render_fact_result(result) if isinstance(result, FactResult) else None
    except Exception as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    actual = _result_payload(result, rendered)
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _classify_answer(payload: dict[str, Any]) -> Any:
    engine_input = engine_input_from_payload(payload)
    engine_output = (
        RelationEngineOutput(relations=engine_input.relations)
        if payload.get("precomputed")
        else execute_operations(engine_input)
    )
    return classify_answer_result(
        _answer_plan(payload),
        engine_output=engine_output,
        final_relation_id=str(payload.get("final_relation_id") or ""),
    )


def _answer_plan(payload: dict[str, Any]) -> AnswerPlan:
    render_outputs = tuple(
        RenderRelationOutput(
            id=str(item["id"]),
            relation_id=str(item["relation_id"]),
            field_id=str(item["field_id"]),
            label=str(item.get("label") or ""),
        )
        for item in payload.get("render_outputs") or ()
    )
    return AnswerPlan(
        fulfillment=tuple(
            FactFulfillment(
                requested_fact_id=str(item.get("requested_fact_id") or "rf_answer"),
                answer_output_id=str(item.get("answer_output_id") or item["id"]),
                render_output_id=str(item["id"]),
            )
            for item in payload.get("render_outputs") or ()
        ),
        operations=engine_input_from_payload(payload).operations,
        render_spec=RenderSpec(relation_outputs=render_outputs),
    )


def _plan_impossible(payload: dict[str, Any]) -> PlanImpossible:
    return PlanImpossible(
        blocked_facts=tuple(
            BlockedFact(
                requested_fact_id=str(item["requested_fact_id"]),
                basis=BlockedFactBasis(str(item["basis"])),
                evidence_refs=tuple(str(ref) for ref in item.get("evidence_refs") or ()),
                reviewed_read_ids=tuple(
                    str(ref) for ref in item.get("reviewed_read_ids") or ()
                ),
                nearest_fields=tuple(
                    BlockedFactField(
                        read_id=str(field["read_id"]),
                        field_id=str(field["field_id"]),
                    )
                    for field in item.get("nearest_fields") or ()
                ),
            )
            for item in payload.get("blocked_facts") or ()
        )
    )


def _question_contract(payload: dict[str, Any]) -> Any:
    requested_facts = payload.get("requested_facts") or (
        {"id": "rf_answer", "description": "answer", "answer_outputs": ["answer"]},
    )
    return question_contract_from_payload({"requested_facts": requested_facts})


def _result_payload(result: Any, rendered: Any) -> dict[str, Any]:
    if result is None:
        return {"result": None}
    if isinstance(result, ExecutionIssue):
        return {
            "issue_kind": result.kind.value,
            "relation_id": result.relation_id,
            "proof_refs": list(result.proof_refs),
        }
    if isinstance(result, FactResult):
        outcome = result.outcome
        actual: dict[str, Any] = {"outcome_kind": outcome.kind.value}
        if hasattr(outcome, "operation"):
            actual["undefined"] = {
                "operation_id": outcome.operation.operation_id,
                "reason_code": outcome.operation.reason_code.value,
                "input_refs": list(outcome.operation.input_refs),
                "proof_refs": list(outcome.operation.proof_refs),
            }
            actual["proof_refs"] = list(outcome.proof_refs)
        if hasattr(outcome, "blocked_requirements"):
            actual["blocked_requirements"] = [
                {
                    "requested_fact_id": item.requested_fact_id,
                    "fact_ref": item.fact_ref,
                    "reviewed_read_ids": list(item.reviewed_read_ids),
                    "nearest_fields": [
                        {"read_id": field.read_id, "field_id": field.field_id}
                        for field in item.nearest_fields
                    ],
                    "proof_refs": list(item.proof_refs),
                }
                for item in outcome.blocked_requirements
            ]
        if hasattr(outcome, "empty_relation"):
            actual["empty_relation"] = {
                "kind": outcome.empty_relation.kind.value,
                "relation_id": outcome.empty_relation.relation_id,
            }
        if hasattr(outcome, "clarifications"):
            actual["clarifications"] = [
                clarification_payload(clarification)
                for clarification in outcome.clarifications
            ]
        if rendered is not None:
            actual["rendered_text"] = rendered_fact_text(rendered)
            actual["rendered_payload"] = rendered_fact_payload(rendered)
            actual["rendered_rows"] = [dict(row) for row in rendered.rows]
            actual["rendered_message"] = rendered.message
            actual["rendered_details"] = dict(rendered.details or {})
            actual["proof_refs"] = list(rendered.proof_refs)
        return actual


def _engine_output_payload(output: Any) -> dict[str, Any]:
    actual: dict[str, Any] = {}
    if output.undefined is not None:
        actual["undefined"] = {
            "operation_id": output.undefined.operation.operation_id,
            "reason_code": output.undefined.operation.reason_code.value,
            "input_refs": list(output.undefined.operation.input_refs),
            "proof_refs": list(output.undefined.operation.proof_refs),
        }
        actual["proof_refs"] = list(output.undefined.proof_refs)
    if output.issue is not None:
        actual["issue_kind"] = output.issue.kind.value
        actual["relation_id"] = output.issue.relation_id
    return actual
