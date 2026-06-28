"""Django delivery adapter for the canonical lineage run spine."""

from __future__ import annotations

from contextlib import AbstractContextManager

from django.conf import settings
from django.db import transaction
from django.db.models import Max

from fervis.lineage import models
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.run_spine import (
    QuestionRunSequenceStore,
    QuestionRunStartRequest,
    record_question_run_start as record_question_run_start_spine,
)
from fervis.lineage.django.recorder import DjangoLineageRecorder


DJANGO_DRF_ADAPTER_REF = "django_drf"
DEFAULT_RUNTIME_VERSION = "development"


def runtime_version_from_settings() -> str:
    value = str(getattr(settings, "FERVIS_RUNTIME_VERSION", "") or "").strip()
    return value or DEFAULT_RUNTIME_VERSION


def record_question_run_start(
    request: QuestionRunStartRequest,
    *,
    recorder: LineageRecorderPort | None = None,
) -> None:
    active_recorder = recorder or DjangoLineageRecorder()
    record_question_run_start_spine(
        request,
        sequence_store=DjangoQuestionRunSequenceStore(),
        recorder=active_recorder,
    )


class DjangoQuestionRunSequenceStore(QuestionRunSequenceStore):
    def transaction(self) -> AbstractContextManager[object]:
        return transaction.atomic()

    def next_conversation_sequence(self, conversation_id: str) -> int:
        models.Conversation.objects.select_for_update().get(
            conversation_id=conversation_id,
        )
        current = (
            models.Question.objects.filter(conversation_id=conversation_id).aggregate(
                current=Max("conversation_sequence")
            )["current"]
            or 0
        )
        return int(current) + 1

    def next_question_run_number(self, question_id: str) -> int:
        models.Question.objects.select_for_update().get(question_id=question_id)
        current = (
            models.QuestionRun.objects.filter(question_id=question_id).aggregate(
                current=Max("run_number")
            )["current"]
            or 0
        )
        return int(current) + 1
