from __future__ import annotations

from typing import Any

from fervis.lookup.question_contract import (
    KnownInputKind,
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
    RequestedFactRowSetReferenceInput,
)


def question_contract_from_payload(payload: dict[str, Any]) -> QuestionContract:
    return QuestionContract(
        requested_facts=tuple(
            requested_fact_from_payload(item)
            for item in payload.get("requested_facts") or ()
        )
    )


def requested_fact_from_payload(payload: dict[str, Any]) -> RequestedFact:
    known_inputs = tuple(
        known_input_from_payload(item) for item in payload.get("known_inputs") or ()
    )
    return RequestedFact(
        id=str(payload["id"]),
        description=str(payload.get("description") or payload["id"]),
        required_for=str(payload.get("required_for") or ""),
        answer_subject=answer_subject_from_payload(payload.get("answer_subject")),
        answer_outputs=tuple(
            answer_output_from_payload(item)
            for item in payload.get("answer_outputs") or ()
        ),
        known_inputs=known_inputs,
        input_refs=tuple(
            payload.get("input_refs") or (item.id for item in known_inputs)
        ),
    )


def answer_subject_from_payload(payload: Any) -> RequestedFactAnswerSubject | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return RequestedFactAnswerSubject(subject_text=payload)
    if not isinstance(payload, dict):
        raise ValueError("answer_subject must be a string or object")
    return RequestedFactAnswerSubject(
        subject_text=str(payload["subject_text"]),
    )


def answer_output_from_payload(payload: Any) -> RequestedFactAnswerOutput:
    if isinstance(payload, str):
        return RequestedFactAnswerOutput(id=payload)
    if not isinstance(payload, dict):
        raise ValueError("answer output must be a string or object")
    return RequestedFactAnswerOutput(
        id=str(payload["id"]),
        description=str(payload.get("description") or ""),
    )


def known_input_from_payload(payload: dict[str, Any]) -> RequestedFactKnownInput:
    kind = KnownInputKind(str(payload["kind"]))
    if kind == KnownInputKind.LITERAL:
        return RequestedFactLiteralInput(
            id=str(payload["id"]),
            source=KnownInputSource(str(payload.get("source") or "question_context")),
            text=str(payload.get("text") or ""),
            resolved_input_ref=str(payload.get("resolved_input_ref") or ""),
            resolved_value_text=str(payload.get("resolved_value_text") or ""),
            field_label_text=str(payload.get("field_label_text") or ""),
            value_meaning_hint=str(payload.get("value_meaning_hint") or ""),
            role=LiteralInputRole(str(payload["role"])),
        )
    return RequestedFactRowSetReferenceInput(
        id=str(payload["id"]),
        text=str(payload.get("text") or ""),
        occurrence=int(payload.get("occurrence") or 1),
        resolved_input_ref=str(payload.get("resolved_input_ref") or ""),
    )
