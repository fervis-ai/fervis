"""Active clarification projection for conversation resolution."""

from __future__ import annotations

from typing import Any, Mapping

from fervis.lookup.turn_prompts.context import active_clarification_context


def active_clarification_contract(
    conversation_context: Mapping[str, object],
    *,
    current_question: str = "",
) -> dict[str, Any] | None:
    context = active_clarification_context(
        conversation_context,
        current_question=current_question,
    )
    if context is None:
        return None
    return {
        "original_question": context.original_question,
        "exchanges": [
            {
                "clarification_questions": list(exchange.questions),
                "answer": exchange.answer,
            }
            for exchange in context.exchanges
        ],
    }
