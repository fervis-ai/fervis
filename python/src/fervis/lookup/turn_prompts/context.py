"""Typed shared context for Lookup prompt turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ClarificationExchangePromptContext:
    questions: tuple[str, ...] = ()
    answer: str = ""


@dataclass(frozen=True)
class ActiveClarificationPromptContext:
    original_question: str
    exchanges: tuple[ClarificationExchangePromptContext, ...] = ()


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
    active_clarification: ActiveClarificationPromptContext | None = None
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
        active_clarification=active_clarification_context(
            conversation_context,
            current_question=current_question,
        ),
        memory=(
            MemoryPromptContext(payload=memory_payload) if memory_payload else None
        ),
        conversation_context=conversation_context,
    )


def active_clarification_context(
    conversation_context: Mapping[str, object],
    *,
    current_question: str = "",
) -> ActiveClarificationPromptContext | None:
    chain = _active_clarification_artifact_chain(conversation_context)
    if not chain:
        return None
    original_question = _source_question(chain[0])
    if not original_question:
        return None
    exchanges: list[ClarificationExchangePromptContext] = []
    for index, artifact in enumerate(chain):
        answer = ""
        if index + 1 < len(chain):
            answer = _source_question(chain[index + 1])
        else:
            answer = current_question.strip()
        exchanges.append(
            ClarificationExchangePromptContext(
                questions=tuple(_clarification_questions(artifact)),
                answer=answer,
            )
        )
    return ActiveClarificationPromptContext(
        original_question=original_question,
        exchanges=tuple(exchanges),
    )


def _active_clarification_artifact_chain(
    conversation_context: Mapping[str, object],
) -> tuple[dict[str, Any], ...]:
    artifacts = conversation_context.get("factArtifacts")
    if not isinstance(artifacts, list | tuple):
        return ()
    chain_newest_first: list[dict[str, Any]] = []
    for artifact in reversed(artifacts):
        if not isinstance(artifact, dict):
            break
        if artifact.get("outcome") != "needs_clarification":
            break
        if not _source_question(artifact):
            break
        if not _clarification_questions(artifact):
            break
        chain_newest_first.append(artifact)
    return tuple(reversed(chain_newest_first))


def _source_question(artifact: dict[str, Any]) -> str:
    return str(artifact.get("sourceQuestion") or "").strip()


def _clarification_questions(artifact: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    for address in artifact.get("addresses") or ():
        if not isinstance(address, dict):
            continue
        for item in address.get("clarificationQuestions") or ():
            text = str(item or "").strip()
            if text and text not in questions:
                questions.append(text)
    return questions
