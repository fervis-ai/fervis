"""Input context for interpreting one factual question."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.lookup.clarification.model import QuestionContractResponse
from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
)
from fervis.lookup.turn_prompts.context import HostPromptContext


@dataclass(frozen=True)
class QuestionContractRequest:
    current_question: str
    conversation_context: dict[str, Any]
    conversation_resolution: CompiledConversationResolution | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)
    clarification_responses: tuple[QuestionContractResponse, ...] = ()


__all__ = ["QuestionContractRequest"]
