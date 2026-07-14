"""Framework-neutral writers for the canonical conversation/question/run spine."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import AbstractContextManager
from typing import Protocol

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.lineage.enums import ConversationOriginKind, QuestionRunKind, RunTriggerKind
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.recorder import ConversationWrite, QuestionRunWrite, QuestionWrite


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
    kind: QuestionRunKind
    trigger_kind: RunTriggerKind
    adapter_ref: str
    runtime_version: str
    base_run_id: str | None = None
    trigger_clarification_response_id: str = ""


@dataclass(frozen=True)
class QuestionRunStartRequest:
    run: QuestionRunStart
    question: QuestionStart | None = None


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
        recorder.start_run(
            QuestionRunWrite(
                run_id=request.run.run_id,
                question_id=request.run.question_id,
                run_number=sequence_store.next_question_run_number(
                    request.run.question_id
                ),
                kind=request.run.kind,
                trigger_kind=request.run.trigger_kind,
                adapter_ref=request.run.adapter_ref,
                runtime_version=request.run.runtime_version,
                base_run_id=request.run.base_run_id,
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
