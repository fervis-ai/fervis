"""Typed shared context for Lookup prompt turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class MemoryPromptValue:
    memory_id: str
    label: str
    value: object


@dataclass(frozen=True)
class MemoryPromptContext:
    values: tuple[MemoryPromptValue, ...] = ()
    payload: Mapping[str, object] | None = None


@dataclass(frozen=True)
class HostPromptContext:
    organization_name: str = ""
    about_api: str = ""


@dataclass(frozen=True)
class TurnPromptContext:
    current_question: str
    host: HostPromptContext = field(default_factory=HostPromptContext)
    memory: MemoryPromptContext | None = None
    conversation_context: Mapping[str, object] = field(default_factory=dict)


def build_turn_prompt_context(
    *,
    current_question: str,
    conversation_context: Mapping[str, object],
    host: HostPromptContext | None = None,
    memory_payload: Mapping[str, object] | None = None,
) -> TurnPromptContext:
    return TurnPromptContext(
        current_question=current_question,
        host=host or HostPromptContext(),
        memory=(
            MemoryPromptContext(payload=memory_payload) if memory_payload else None
        ),
        conversation_context=conversation_context,
    )
