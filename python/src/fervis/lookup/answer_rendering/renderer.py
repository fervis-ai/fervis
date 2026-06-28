"""Deterministic rendering projection for fact results."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from fervis.lookup.answer_rendering.model import RenderedFact
from fervis.lookup.outcomes.model import (
    AnswerResult,
    FactResult,
    Impossible,
    NeedsClarification,
    NoData,
    Undefined,
)
from fervis.lookup.outcomes.terminal_details import fact_result_terminal_details


def render_fact_result(result: FactResult) -> RenderedFact:
    outcome = result.outcome
    if isinstance(outcome, AnswerResult):
        return RenderedFact(
            kind=outcome.kind,
            rows=_render_rows(outcome),
            row_labels=_render_row_labels(outcome),
            scalars=outcome.scalars,
            proof_refs=outcome.proof_refs,
            render_outputs=_render_output_manifest(outcome),
        )
    return RenderedFact(
        kind=outcome.kind,
        message=_terminal_message(outcome),
        details=_terminal_details(outcome),
        proof_refs=getattr(outcome, "proof_refs", ()),
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


def _render_rows(outcome: AnswerResult) -> tuple[Mapping[str, object], ...]:
    if outcome.render_spec is None or not outcome.render_spec.relation_outputs:
        return ()
    rows: list[dict[str, object]] = []
    for relation_id, relation_outputs in _relation_outputs_by_relation(outcome).items():
        relation = next(
            (item for item in outcome.relations if item.id == relation_id),
            None,
        )
        if relation is None:
            raise ValueError("render relation is unavailable")
        for row in relation.rows:
            rendered: dict[str, object] = {}
            for relation_output in relation_outputs:
                if relation_output.field_id not in row:
                    raise ValueError("render field is unavailable")
                rendered[relation_output.id] = row[relation_output.field_id]
            rows.append(rendered)
    return tuple(rows)


def _relation_outputs_by_relation(outcome: AnswerResult) -> dict[str, list[object]]:
    outputs: dict[str, list[object]] = {}
    if outcome.render_spec is None:
        return outputs
    for relation_output in outcome.render_spec.relation_outputs:
        outputs.setdefault(relation_output.relation_id, []).append(relation_output)
    return outputs


def _render_row_labels(outcome: AnswerResult) -> Mapping[str, str] | None:
    if outcome.render_spec is None:
        return None
    labels = {
        relation_output.id: relation_output.label
        for relation_output in outcome.render_spec.relation_outputs
        if relation_output.label
    }
    return labels or None


def _render_output_manifest(outcome: AnswerResult) -> tuple[Mapping[str, object], ...]:
    if outcome.render_spec is None:
        return ()
    output: list[Mapping[str, object]] = []
    for relation_output in outcome.render_spec.relation_outputs:
        role = _render_output_role(relation_output)
        if not role:
            continue
        item: dict[str, object] = {
            "key": relation_output.id,
            "role": role,
        }
        if relation_output.label:
            item["label"] = relation_output.label
        output.append(item)
    for scalar_output in outcome.render_spec.scalar_outputs:
        role = _render_output_role(scalar_output)
        if not role:
            continue
        item = {
            "key": scalar_output.id,
            "role": role,
        }
        if scalar_output.label:
            item["label"] = scalar_output.label
        output.append(item)
    return tuple(output)


def _render_output_role(output: object) -> str:
    return str(getattr(output, "role", "") or "")


def _render_row_text(
    row: Mapping[str, object],
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


def _terminal_message(outcome: object) -> str:
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
    questions = [item.question for item in outcome.clarifications if item.question]
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


def _terminal_details(outcome: object) -> Mapping[str, object] | None:
    return fact_result_terminal_details(FactResult(outcome=outcome))
