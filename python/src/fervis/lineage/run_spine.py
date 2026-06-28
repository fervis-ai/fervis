"""Framework-neutral writers for the canonical conversation/question/run spine."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import AbstractContextManager
from typing import Protocol

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.lineage.enums import ConversationOriginKind, RunTriggerKind
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.recorder import (
    ClarificationResponseWrite,
    ConversationWrite,
    QuestionRunWrite,
    QuestionWrite,
)


@dataclass(frozen=True)
class QuestionStart:
    conversation_id: str
    tenant_id: str
    read_context_ref: ReadContextRef
    question_id: str
    origin_message_ref: str
    question: str


@dataclass(frozen=True)
class QuestionRunStart:
    question_id: str
    run_id: str
    trigger_kind: RunTriggerKind
    integrated_question: str
    adapter_ref: str
    runtime_version: str
    previous_run_id: str | None = None
    trigger_clarification_response_run_id: str | None = None
    trigger_clarification_response_id: str | None = None


@dataclass(frozen=True)
class ClarificationResponseStart:
    response_id: str
    run_id: str
    clarification_id: str
    response_text: str


@dataclass(frozen=True)
class QuestionRunStartRequest:
    run: QuestionRunStart
    question: QuestionStart | None = None
    clarification_response: ClarificationResponseStart | None = None


class QuestionRunSequenceStore(Protocol):
    def transaction(self) -> AbstractContextManager[object]: ...

    def next_conversation_sequence(self, conversation_id: str) -> int: ...

    def next_question_run_number(self, question_id: str) -> int: ...


def record_question_run_start(
    request: QuestionRunStartRequest,
    *,
    sequence_store: QuestionRunSequenceStore,
    recorder: LineageRecorderPort,
) -> None:
    with sequence_store.transaction():
        if request.question is not None:
            _record_conversation(
                conversation_id=request.question.conversation_id,
                tenant_id=request.question.tenant_id,
                read_context_ref=request.question.read_context_ref,
                recorder=recorder,
            )
            recorder.record_question(
                QuestionWrite(
                    question_id=request.question.question_id,
                    conversation_id=request.question.conversation_id,
                    conversation_sequence=sequence_store.next_conversation_sequence(
                        request.question.conversation_id
                    ),
                    origin_message_ref=request.question.origin_message_ref,
                    original_question=request.question.question,
                )
            )
        if request.clarification_response is not None:
            recorder.record_clarification_response(
                ClarificationResponseWrite(
                    response_id=request.clarification_response.response_id,
                    run_id=request.clarification_response.run_id,
                    clarification_id=request.clarification_response.clarification_id,
                    evidence_ref=(
                        f"clarification_response:"
                        f"{request.clarification_response.response_id}"
                    ),
                    response_text=request.clarification_response.response_text,
                )
            )
        recorder.start_run(
            QuestionRunWrite(
                run_id=request.run.run_id,
                question_id=request.run.question_id,
                run_number=sequence_store.next_question_run_number(
                    request.run.question_id
                ),
                trigger_kind=request.run.trigger_kind,
                integrated_question=request.run.integrated_question,
                adapter_ref=request.run.adapter_ref,
                runtime_version=request.run.runtime_version,
                previous_run_id=request.run.previous_run_id,
                trigger_clarification_response_run_id=(
                    request.run.trigger_clarification_response_run_id
                ),
                trigger_clarification_response_id=(
                    request.run.trigger_clarification_response_id
                ),
            )
        )


def _record_conversation(
    *,
    conversation_id: str,
    tenant_id: str,
    read_context_ref: ReadContextRef,
    recorder: LineageRecorderPort,
) -> None:
    recorder.ensure_conversation(
        ConversationWrite(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            read_context_ref=read_context_ref,
            origin_kind=ConversationOriginKind.INITIAL,
        )
    )
