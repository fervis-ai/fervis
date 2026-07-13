"""Deterministic rendering projection for fact results."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from fervis.lookup.answer_rendering.model import RenderedFact
from fervis.lookup.answer_program.result_projection import (
    EntityKeyValue,
    RelationResultOutput,
    ResultValue,
    ScalarResultOutput,
)
from fervis.lookup.clarification import render_clarification_question
from fervis.lookup.outcomes.model import (
    AnswerResult,
    FactResult,
    Impossible,
    NeedsClarification,
    NoData,
    ResultOutcome,
    Undefined,
)
from fervis.lookup.outcomes.terminal_details import fact_result_terminal_details
from fervis.lookup.canonical_data import RuntimeValue


_PUBLIC_RESULT_ROLES = frozenset({"answer_value", "ranking_metric"})


def render_fact_result(result: FactResult) -> RenderedFact:
    outcome = result.outcome
    if isinstance(outcome, AnswerResult):
        return RenderedFact(
            kind=outcome.kind,
            rows=_render_rows(outcome),
            row_labels=_render_row_labels(outcome),
            scalars=_render_scalars(outcome),
            proof_refs=outcome.proof_refs,
            render_outputs=_render_output_manifest(outcome),
        )
    return RenderedFact(
        kind=outcome.kind,
        message=_terminal_message(outcome),
        details=_terminal_details(outcome),
        proof_refs=outcome.proof_refs,
    )


def rendered_fact_text(rendered: RenderedFact) -> str:
    lines: list[str] = []
    if rendered.rows:
        lines.extend(
            _render_row_text(row, labels=rendered.row_labels) for row in rendered.rows
        )
    if rendered.scalars:
        lines.extend(f"{key}: {value}" for key, value in dict(rendered.scalars).items())
    if lines:
        return "\n".join(lines)
    return rendered.message or rendered.kind.value


def rendered_fact_payload(rendered: RenderedFact) -> dict[str, Any]:
    payload = {
        "kind": rendered.kind.value,
        "rows": _json_safe([dict(row) for row in rendered.rows]),
        "scalars": _json_safe(dict(rendered.scalars or {})),
        "message": rendered.message,
        "details": _json_safe(dict(rendered.details or {})),
    }
    if rendered.proof_refs:
        payload["proofRefs"] = list(rendered.proof_refs)
    if rendered.row_labels:
        payload["rowLabels"] = dict(rendered.row_labels)
    if rendered.render_outputs:
        payload["renderOutputs"] = _json_safe(rendered.render_outputs)
    return payload


def _render_rows(outcome: AnswerResult) -> tuple[Mapping[str, RuntimeValue], ...]:
    public_output_ids = {
        output.id for output in _public_relation_outputs(outcome)
    }
    return tuple(
        {
            output_id: _render_value(value)
            for output_id, value in projected_row.values.items()
            if output_id in public_output_ids
        }
        for projected_row in outcome.projected_rows
    )


def _render_scalars(outcome: AnswerResult) -> Mapping[str, RuntimeValue] | None:
    if not outcome.scalars:
        return None
    rendered = {
        output.id: outcome.scalars[output.scalar_id]
        for output in _public_scalar_outputs(outcome)
    }
    return rendered or None


def _render_row_labels(outcome: AnswerResult) -> Mapping[str, str] | None:
    labels = {
        relation_output.id: relation_output.label
        for relation_output in _public_relation_outputs(outcome)
        if relation_output.label
    }
    return labels or None


def _render_output_manifest(
    outcome: AnswerResult,
) -> tuple[Mapping[str, RuntimeValue], ...]:
    output: list[Mapping[str, RuntimeValue]] = []
    for relation_output in _public_relation_outputs(outcome):
        role = _render_output_role(relation_output)
        item: dict[str, RuntimeValue] = {
            "key": relation_output.id,
            "role": role,
        }
        if relation_output.label:
            item["label"] = relation_output.label
        if relation_output.entity_key is not None:
            item["entityKind"] = relation_output.entity_key.entity_kind
            item["keyId"] = relation_output.entity_key.key_id
        output.append(item)
    for scalar_output in _public_scalar_outputs(outcome):
        role = _render_output_role(scalar_output)
        item = {
            "key": scalar_output.id,
            "role": role,
        }
        if scalar_output.label:
            item["label"] = scalar_output.label
        output.append(item)
    return tuple(output)


def _public_relation_outputs(
    outcome: AnswerResult,
) -> tuple[RelationResultOutput, ...]:
    return tuple(
        output
        for output in outcome.result_projection.relation_outputs
        if output.role in _PUBLIC_RESULT_ROLES
    )


def _public_scalar_outputs(outcome: AnswerResult) -> tuple[ScalarResultOutput, ...]:
    return tuple(
        output
        for output in outcome.result_projection.scalar_outputs
        if output.role in _PUBLIC_RESULT_ROLES
    )


def _render_value(value: ResultValue) -> RuntimeValue:
    if not isinstance(value, EntityKeyValue):
        return value
    return {
        "entityKind": value.entity_kind,
        "keyId": value.key_id,
        "components": value.component_values(),
    }


def _render_output_role(
    output: RelationResultOutput | ScalarResultOutput,
) -> str:
    return output.role


def _render_row_text(
    row: Mapping[str, RuntimeValue],
    *,
    labels: Mapping[str, str] | None,
) -> str:
    if labels:
        return ", ".join(
            f"{labels.get(key, key)}: {value}" for key, value in row.items()
        )
    return ": ".join(str(value) for value in row.values())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _terminal_message(outcome: ResultOutcome) -> str:
    if isinstance(outcome, NeedsClarification):
        return _clarification_message(outcome)
    if isinstance(outcome, Impossible):
        return _impossible_message(outcome)
    if isinstance(outcome, NoData):
        return "No matching data was found."
    if isinstance(outcome, Undefined):
        return outcome.operation.reason_code.value
    return ""


def _clarification_message(outcome: NeedsClarification) -> str:
    questions = [render_clarification_question(item) for item in outcome.clarifications]
    return "\n".join(questions) if questions else "Can you clarify the requested value?"


def _impossible_message(outcome: Impossible) -> str:
    blocked = [
        item.required_for or item.fact_ref or item.requested_fact_id
        for item in outcome.blocked_requirements
        if item.required_for or item.fact_ref or item.requested_fact_id
    ]
    if not blocked:
        return "I cannot answer that from the available API evidence."
    return (
        "I cannot answer "
        + "; ".join(str(item) for item in blocked)
        + " from the available API evidence."
    )


def _terminal_details(outcome: ResultOutcome) -> Mapping[str, RuntimeValue] | None:
    return fact_result_terminal_details(FactResult(outcome=outcome))
