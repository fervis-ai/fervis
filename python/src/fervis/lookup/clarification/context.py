"""Ordered clarification context and provenance."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.memory.conversation_context import (
    ConversationContextSource,
    ConversationMeaningAnchor,
)

from .model import ClarificationOwnerResponse, ConversationResolutionResponse


@dataclass(frozen=True)
class ClarificationExchange:
    response_id: str
    clarification_id: str
    question: str
    answer: str

    def __post_init__(self) -> None:
        values = (
            self.response_id,
            self.clarification_id,
            self.question,
            self.answer,
        )
        if any(not value.strip() for value in values):
            raise ValueError("clarification exchange requires complete attribution")

    @property
    def lineage_ref(self) -> str:
        return clarification_response_ref(self.response_id)

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "response_id": self.response_id,
            "clarification_questions": [self.question],
            "answer": self.answer,
        }


@dataclass(frozen=True)
class ActiveClarification:
    original_question: str
    exchanges: tuple[ClarificationExchange, ...]

    def __post_init__(self) -> None:
        if not self.original_question.strip() or not self.exchanges:
            raise ValueError("active clarification requires a complete chain")
        response_ids = tuple(exchange.response_id for exchange in self.exchanges)
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("active clarification contains duplicate responses")

    @property
    def lineage_refs(self) -> tuple[str, ...]:
        return tuple(exchange.lineage_ref for exchange in self.exchanges)

    def to_prompt_payload(self) -> dict[str, object]:
        exchanges = [exchange.to_prompt_payload() for exchange in self.exchanges]
        return {
            "original_question": self.original_question,
            "exchanges": exchanges,
        }


@dataclass(frozen=True)
class _AnchorText:
    anchor_id: str
    kind: str
    label: str
    text: str
    start: int


def clarification_response_ref(response_id: str) -> str:
    if not response_id.strip():
        raise ValueError("clarification response reference requires an identity")
    return f"clarification_response:{response_id}"


def active_clarification(
    responses: tuple[ConversationResolutionResponse, ...],
) -> ActiveClarification:
    first_annotation = responses[0].annotation if responses else None
    if first_annotation is None:
        raise ValueError("active clarification requires an original question")
    exchanges = tuple(_exchange(response) for response in responses)
    return ActiveClarification(
        original_question=first_annotation.suspended_question_text,
        exchanges=exchanges,
    )


def extend_clarification_chain(
    existing: tuple[ClarificationOwnerResponse, ...],
    response: ConversationResolutionResponse,
) -> tuple[ConversationResolutionResponse, ...]:
    chain = tuple(
        item
        for item in existing
        if isinstance(item, ConversationResolutionResponse)
        and item.annotation is not None
    )
    if response.annotation is None:
        raise ValueError("clarification chain requires an attributed response")
    return (*chain, response)


def clarification_context_source(
    clarification: ActiveClarification,
) -> ConversationContextSource:
    source_id = f"active_clarification:{clarification.exchanges[-1].response_id}"
    lines, anchor_texts = _render_context(clarification, source_id=source_id)
    source_text = "\n".join(lines)
    anchors = tuple(
        _meaning_anchor(item, source_text=source_text) for item in anchor_texts
    )
    return ConversationContextSource(
        source_id=source_id,
        kind="active_clarification",
        text=source_text,
        meaning_anchors=anchors,
    )


def active_clarification_from_source(
    source: ConversationContextSource,
) -> ActiveClarification:
    if source.kind != "active_clarification":
        raise ValueError("context source is not an active clarification")
    original_question = _single_anchor_text(source, kind="original_question")
    exchange_anchors = tuple(
        anchor
        for anchor in source.meaning_anchors
        if anchor.kind in {"clarification_question", "clarification_answer"}
    )
    if len(exchange_anchors) % 2:
        raise ValueError("active clarification has an incomplete exchange")
    exchanges = tuple(
        _exchange_from_anchors(question, answer)
        for question, answer in zip(
            exchange_anchors[::2],
            exchange_anchors[1::2],
            strict=True,
        )
    )
    return ActiveClarification(
        original_question=original_question,
        exchanges=exchanges,
    )


def _exchange(response: ConversationResolutionResponse) -> ClarificationExchange:
    annotation = response.annotation
    if annotation is None:
        raise ValueError("active clarification exchange requires an annotation")
    return ClarificationExchange(
        response_id=response.source.response_id,
        clarification_id=response.source.clarification_id,
        question=annotation.clarification_question_text,
        answer=response.source.exact_user_text,
    )


def _render_context(
    clarification: ActiveClarification,
    *,
    source_id: str,
) -> tuple[list[str], list[_AnchorText]]:
    original_prefix = "Original question: "
    lines = [f"{original_prefix}{clarification.original_question}"]
    anchors = [
        _AnchorText(
            anchor_id=f"{source_id}:original_question",
            kind="original_question",
            label="original question",
            text=clarification.original_question,
            start=len(original_prefix),
        )
    ]
    source_length = len(lines[0])
    for index, exchange in enumerate(clarification.exchanges, start=1):
        question_prefix = f"Clarification question {index}: "
        answer_prefix = f"Answer {index}: "
        question_line = f"{question_prefix}{exchange.question}"
        answer_line = f"{answer_prefix}{exchange.answer}"
        question_start = source_length + 1 + len(question_prefix)
        answer_start = source_length + 1 + len(question_line) + 1 + len(answer_prefix)
        lines.extend((question_line, answer_line))
        anchors.extend(
            (
                _AnchorText(
                    anchor_id=f"request:{exchange.clarification_id}",
                    kind="clarification_question",
                    label="clarification question",
                    text=exchange.question,
                    start=question_start,
                ),
                _AnchorText(
                    anchor_id=f"response:{exchange.response_id}",
                    kind="clarification_answer",
                    label="clarification answer",
                    text=exchange.answer,
                    start=answer_start,
                ),
            )
        )
        source_length += 1 + len(question_line) + 1 + len(answer_line)
    return lines, anchors


def _meaning_anchor(
    item: _AnchorText,
    *,
    source_text: str,
) -> ConversationMeaningAnchor:
    occurrence = source_text[: item.start].count(item.text) + 1
    return ConversationMeaningAnchor(
        anchor_id=item.anchor_id,
        text=item.text,
        occurrence=occurrence,
        kind=item.kind,
        label=item.label,
    )


def _exchange_from_anchors(
    question: ConversationMeaningAnchor,
    answer: ConversationMeaningAnchor,
) -> ClarificationExchange:
    if (
        question.kind != "clarification_question"
        or answer.kind != "clarification_answer"
    ):
        raise ValueError("active clarification exchange roles are out of order")
    return ClarificationExchange(
        response_id=_anchor_identity(answer.anchor_id, prefix="response:"),
        clarification_id=_anchor_identity(question.anchor_id, prefix="request:"),
        question=question.text,
        answer=answer.text,
    )


def _anchor_identity(anchor_id: str, *, prefix: str) -> str:
    if not anchor_id.startswith(prefix):
        raise ValueError("clarification anchor has an invalid identity")
    identity = anchor_id.removeprefix(prefix)
    if not identity:
        raise ValueError("clarification anchor requires an identity")
    return identity


def _single_anchor_text(source: ConversationContextSource, *, kind: str) -> str:
    matches = tuple(
        anchor.text for anchor in source.meaning_anchors if anchor.kind == kind
    )
    if len(matches) != 1:
        raise ValueError("active clarification requires one anchor per text role")
    return matches[0]
