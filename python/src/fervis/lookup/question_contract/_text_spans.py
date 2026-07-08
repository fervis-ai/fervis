"""Copied-span matching helpers for question-contract boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CopiedSpan:
    text: str
    context: str
    start: int
    end: int

    @classmethod
    def parse_from(cls, *, text: str, contexts: tuple[str, ...]) -> "CopiedSpan":
        for context in contexts:
            span = cls.find(text=text, context=context)
            if span is not None:
                return span
        raise ValueError("text must come from question context")

    @classmethod
    def find(cls, *, text: str, context: str) -> "CopiedSpan | None":
        spans = cls.find_all(text=text, context=context)
        return spans[0] if spans else None

    @classmethod
    def find_all(cls, *, text: str, context: str) -> tuple["CopiedSpan", ...]:
        output: list[CopiedSpan] = []
        for match in re.finditer(re.escape(text), context):
            start = match.start()
            end = match.end()
            if _has_valid_token_edges(text=text, context=context, start=start, end=end):
                output.append(cls(text=text, context=context, start=start, end=end))
        return tuple(output)


def contains_copied_span(container: str, text: str) -> bool:
    return CopiedSpan.find(text=text, context=container) is not None


def copied_span(
    value: str,
    *,
    question_context_texts: tuple[str, ...],
) -> CopiedSpan:
    return CopiedSpan.parse_from(text=value, contexts=question_context_texts)


def _has_valid_token_edges(
    *,
    text: str,
    context: str,
    start: int,
    end: int,
) -> bool:
    if _has_token_edge(text, at_start=True) and start > 0:
        if _is_token_char(context[start - 1]):
            return False
    if _has_token_edge(text, at_start=False) and end < len(context):
        if _is_token_char(context[end]):
            return False
    return True


def _has_token_edge(text: str, *, at_start: bool) -> bool:
    value = text.strip()
    if not value:
        return False
    char = value[0] if at_start else value[-1]
    return _is_token_char(char)


def _is_token_char(char: str) -> bool:
    return char.isalnum() or char in {"_", "-"}
