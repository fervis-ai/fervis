from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Protocol

import sqlalchemy as sa
from django.utils import timezone

from fervis.interfaces.cli.dispatch import run_init_command, run_migrate_command
from fervis.lineage.django.runtime_failures import (
    record_worker_runtime_error,
)
from fervis.interfaces.django.question_run_ports import (
    DjangoQuestionLineagePort,
    DjangoQuestionLifecyclePort,
)
from fervis.run_work.queue.django.models import RunWorkItem
from fervis.run_work.queue.django.queue import claim_run_work_items
from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.lineage.enums import RunResultKind
from fervis.lineage.enums import RunStepKey, RunStepKind
from fervis.lineage.models import (
    AnswerOutput,
    AnswerPresentation,
    Question,
    QuestionRun,
    RunResult,
)
from fervis.lineage.recorder_core import LineageRecorder
from fervis.lineage.recorder import ClarificationRequestWrite, RunStepWrite
from fervis.project import discover_project
from fervis.project.configuration import load_fervis_project_config
from fervis.project.persistence.schema import metadata
from fervis.run_work.service import RunWorkService
from fervis.questions.service import QuestionService
from fervis.questions.ports import (
    LookupExecutionRequest,
    LookupExecutionResult,
    QuestionLookupPort,
)
from fervis.storage.sql.bundle import sql_storage_bundle
from fervis.storage.sql.lineage_store import SQLLineageRecorderStore
from fervis.storage.sql.rows import now_utc
from fervis.storage.sql.terminal import (
    record_runtime_error_result,
    terminal_result_for_run,
)
from fervis.storage.sql.work_items import SQLWorkItemQueue
from tests.testkit.django import SEEDED_USER_PK
from tests.testkit.terminal_lineage import (
    TerminalAnswerWriter,
    make_terminal_answer_writer,
)

_CLARIFICATION_PAYLOAD: dict[str, object] = {
    "id": "clarification_area",
    "need": "target_reference",
    "reason": "multiple_matching_entities",
    "owner": "grounding",
    "continuation": {
        "kind": "grounding",
        "knownInputId": "area_input",
        "acceptsFreeText": False,
    },
    "requestedFactId": "fact_1",
    "question": "Which matching area should I use?",
    "subjects": [
        {
            "kind": "question_input",
            "id": "area_input",
            "label": "area",
            "sourceText": "selected area",
            "options": [
                {
                    "id": "area_nairobi",
                    "label": "Nairobi",
                    "entityKind": "area",
                    "keyId": "area_id",
                    "matchedField": "id",
                    "matchedValue": "area_1",
                    "resolverReadId": "list_areas",
                }
            ],
        }
    ],
    "evidence": [
        {"kind": "resolver_read", "id": "list_areas", "readId": "list_areas"}
    ],
}


def _clarification_writer(recorder):
    def write(request: LookupExecutionRequest, payload: dict[str, object]) -> None:
        attempt = request.active_attempt or 1
        step_id = f"{request.run_id}.clarification.{attempt}"
        recorder.record_step(
            RunStepWrite(
                step_id=step_id,
                run_id=request.run_id,
                sequence=(attempt - 1) * 10_000 + 1,
                step_key=RunStepKey.GROUNDING,
                kind=RunStepKind.DETERMINISTIC,
                input_summary_json={},
                output_summary_json={},
                error_json={},
            )
        )
        recorder.record_clarification_request(
            ClarificationRequestWrite(
                clarification_id=str(payload["id"]),
                run_id=request.run_id,
                step_id=step_id,
                payload_json=payload,
            )
        )

    return write


@dataclass(frozen=True)
class ContractAdapterFactory:
    name: str
    build: Callable[[Path], "ContractAdapter"]

    def __call__(self, tmp_path: Path) -> "ContractAdapter":
        return self.build(tmp_path)


@dataclass
class ContractAdapter:
    name: str
    questions: QuestionService
    run_work: RunWorkService
    lookup: "ScriptedLookup"
    work_items: "WorkItemHandle"
    probe: "AdapterProbe"
    tenant_id: str = "tenant_1"
    principal_id: str = "user_1"
    provider: str = "anthropic"
    model_key: str = "HAIKU"


class WorkItemHandle(Protocol):
    def claim_one(self, *, worker_id: str = "worker_1") -> "ClaimedWorkItem": ...


@dataclass(frozen=True)
class ClaimedWorkItem:
    run_id: str
    active_attempt: int


class AdapterProbe(Protocol):
    def question_count(self, conversation_id: str) -> int: ...

    def run_count(self, conversation_id: str) -> int: ...

    def work_item_status(self, run_id: str) -> str | None: ...

    def terminal_result(self, run_id: str) -> dict[str, object] | None: ...

    def record_terminal_error(
        self, run_id: str, error: str = "already_terminal"
    ) -> None: ...

    def expire_running_lease(self, run_id: str) -> None: ...


@dataclass
class ScriptedLookup(QuestionLookupPort):
    mode: str = "complete_without_terminal"
    error: str = "lookup_failed"
    answer: str = "42"
    result_data: dict[str, object] = field(default_factory=lambda: {"value": 42})
    clarification_payload: dict[str, object] = field(
        default_factory=lambda: dict(_CLARIFICATION_PAYLOAD)
    )
    calls: list[LookupExecutionRequest] = field(default_factory=list)
    terminal_answer_writer: TerminalAnswerWriter | None = None
    clarification_writer: Callable[[LookupExecutionRequest, dict[str, object]], None] | None = None

    def complete_without_terminal(self, *, answer: str = "42") -> None:
        self.mode = "complete_without_terminal"
        self.answer = answer

    def complete_with_terminal(self, *, answer: str = "42") -> None:
        self.mode = "complete_with_terminal"
        self.answer = answer
        self.result_data = {"value": answer}

    def fail_result_without_terminal(self, *, error: str = "lookup_failed") -> None:
        self.mode = "failed_without_terminal"
        self.error = error

    def raise_error(self, *, error: str = "provider timeout") -> None:
        self.mode = "raise"
        self.error = error

    def needs_clarification(
        self,
        *,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.mode = "needs_clarification"
        self.clarification_payload = dict(payload or _CLARIFICATION_PAYLOAD)
        self.result_data = {
            "kind": "needs_clarification",
            "details": {"clarifications": [self.clarification_payload]},
        }

    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del progress_sink
        self.calls.append(request)
        if self.mode == "raise":
            raise RuntimeError(self.error)
        if self.mode == "failed_without_terminal":
            return LookupExecutionResult(
                status="FAILED",
                error=self.error,
                terminal_lineage_recorded=False,
            )
        if self.mode == "needs_clarification":
            if self.clarification_writer is None:
                raise RuntimeError("clarification writer is not configured")
            self.clarification_writer(request, self.clarification_payload)
            return LookupExecutionResult(
                status="NEEDS_CLARIFICATION",
                result_data=dict(self.result_data),
                terminal_lineage_recorded=False,
            )
        terminal_lineage_recorded = self.mode == "complete_with_terminal"
        if terminal_lineage_recorded:
            if self.terminal_answer_writer is None:
                raise RuntimeError("terminal answer writer is not configured")
            self.terminal_answer_writer(request, self.answer, dict(self.result_data))
        return LookupExecutionResult(
            status="COMPLETED",
            answer=self.answer,
            result_data=dict(self.result_data),
            terminal_lineage_recorded=terminal_lineage_recorded,
        )

    def call_count(self) -> int:
        return len(self.calls)

    def run_program(self, request, *, progress_sink=None) -> LookupExecutionResult:
        del request, progress_sink
        raise RuntimeError("scripted model lookup cannot execute answer programs")

    def last_request(self) -> LookupExecutionRequest | None:
        return self.calls[-1] if self.calls else None


@dataclass
class DeterministicIds:
    question_count: int = 0
    run_count: int = 0
    conversation_count: int = 0
    clarification_response_count: int = 0

    def new_conversation_id(self) -> str:
        self.conversation_count += 1
        return f"conversation_{self.conversation_count}"

    def new_question_id(self) -> str:
        self.question_count += 1
        return f"question_{self.question_count}"

    def new_run_id(self) -> str:
        self.run_count += 1
        return f"run_{self.run_count}"

    def new_clarification_response_id(self) -> str:
        self.clarification_response_count += 1
        return f"clarification_response_{self.clarification_response_count}"


def sql_adapter(tmp_path: Path) -> ContractAdapter:
    root = _migrated_fastapi_project(tmp_path)
    loaded = load_fervis_project_config(discover_project(root))
    lookup = ScriptedLookup()
    bundle = sql_storage_bundle(
        project=discover_project(root),
        loaded_config=loaded,
        lookup=lookup,
    )
    lookup.terminal_answer_writer = make_terminal_answer_writer(
        LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    )
    lookup.clarification_writer = _clarification_writer(
        LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    )
    return ContractAdapter(
        name="sql",
        questions=bundle.questions,
        run_work=bundle.run_work,
        lookup=lookup,
        work_items=SQLWorkItems(bundle.engine),
        probe=SQLProbe(engine=bundle.engine),
        principal_id="user_1",
    )


def django_adapter(tmp_path: Path) -> ContractAdapter:
    del tmp_path
    lookup = ScriptedLookup(
        terminal_answer_writer=make_terminal_answer_writer(DjangoLineageRecorder()),
        clarification_writer=_clarification_writer(DjangoLineageRecorder()),
    )
    questions = QuestionService(
        lineage=DjangoQuestionLineagePort(),
        runs=DjangoQuestionLifecyclePort(),
        lookup=lookup,
        program=lookup,
        ids=DeterministicIds(),
        adapter_ref="django_drf:test",
        runtime_version="test-runtime",
    )
    return ContractAdapter(
        name="django",
        questions=questions,
        run_work=questions.run_work,
        lookup=lookup,
        work_items=DjangoWorkItems(),
        probe=DjangoProbe(),
        principal_id=str(SEEDED_USER_PK),
    )


CONTRACT_ADAPTERS = (
    ContractAdapterFactory("sql", sql_adapter),
    ContractAdapterFactory("django", django_adapter),
)


@dataclass(frozen=True)
class SQLWorkItems:
    engine: Any

    def claim_one(self, *, worker_id: str = "worker_1") -> ClaimedWorkItem:
        [item] = SQLWorkItemQueue(self.engine).claim_run_work_items(
            worker_id=worker_id,
            batch_size=1,
            lease_seconds=60,
        )
        return ClaimedWorkItem(run_id=item.run_id, active_attempt=item.active_attempt)


class DjangoWorkItems:
    def claim_one(self, *, worker_id: str = "worker_1") -> ClaimedWorkItem:
        [item] = claim_run_work_items(
            worker_id=worker_id,
            batch_size=1,
            lease_seconds=60,
        )
        return ClaimedWorkItem(run_id=item.run_id, active_attempt=item.active_attempt)


@dataclass(frozen=True)
class SQLProbe:
    engine: Any

    def question_count(self, conversation_id: str) -> int:
        return self._count_where("fervis_question", "conversation_id", conversation_id)

    def run_count(self, conversation_id: str) -> int:
        question = metadata.tables["fervis_question"]
        run = metadata.tables["fervis_question_run"]
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    sa.select(sa.func.count())
                    .select_from(
                        run.join(question, run.c.question_id == question.c.question_id)
                    )
                    .where(question.c.conversation_id == conversation_id)
                ).scalar_one()
            )

    def work_item_status(self, run_id: str) -> str | None:
        return self._work_item_value(run_id, "status")

    def terminal_result(self, run_id: str) -> dict[str, object] | None:
        terminal = terminal_result_for_run(self.engine, run_id)
        if terminal is None:
            return None
        return {
            "status": terminal.status,
            "answer": terminal.answer,
            "result_data": self._answer_output_value(run_id),
            "error": terminal.error,
        }

    def record_terminal_error(
        self, run_id: str, error: str = "already_terminal"
    ) -> None:
        record_runtime_error_result(
            engine=self.engine,
            run_id=run_id,
            error_code=error,
        )

    def expire_running_lease(self, run_id: str) -> None:
        table = metadata.tables["fervis_run_work_item"]
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(table)
                .where(table.c.run_id == run_id)
                .values(
                    status="RUNNING",
                    lease_expires_at=now_utc() - timedelta(seconds=1),
                    attempt_count=1,
                    active_attempt=1,
                )
            )

    def _count_where(self, table_name: str, field_name: str, value: str) -> int:
        table = metadata.tables[table_name]
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    sa.select(sa.func.count()).where(table.c[field_name] == value)
                ).scalar_one()
            )

    def _work_item_value(self, run_id: str, field_name: str) -> str | None:
        table = metadata.tables["fervis_run_work_item"]
        with self.engine.connect() as connection:
            value = connection.execute(
                sa.select(table.c[field_name]).where(table.c.run_id == run_id)
            ).scalar()
        return None if value is None else str(value)

    def _answer_output_value(self, run_id: str) -> dict[str, object] | None:
        table = metadata.tables["fervis_answer_output"]
        with self.engine.connect() as connection:
            value = connection.execute(
                sa.select(table.c.value_json)
                .where(table.c.run_id == run_id)
                .order_by(table.c.created_at)
            ).scalar()
        return value if isinstance(value, dict) else None


class DjangoProbe:
    def question_count(self, conversation_id: str) -> int:
        return Question.objects.filter(conversation_id=conversation_id).count()

    def run_count(self, conversation_id: str) -> int:
        return QuestionRun.objects.filter(
            question__conversation_id=conversation_id
        ).count()

    def work_item_status(self, run_id: str) -> str | None:
        item = RunWorkItem.objects.filter(run_id=run_id).first()
        return item.status if item is not None else None

    def terminal_result(self, run_id: str) -> dict[str, object] | None:
        result = RunResult.objects.filter(run_id=run_id).first()
        if result is None:
            return None
        if result.result_kind == RunResultKind.RUNTIME_ERROR.value:
            error = getattr(result, "runtime_error_detail", None)
            return {
                "status": "FAILED",
                "answer": None,
                "result_data": None,
                "error": getattr(error, "message", None),
            }
        answer_text = (
            AnswerPresentation.objects.filter(run_id=run_id)
            .exclude(rendered_value__isnull=True)
            .order_by("created_at")
            .values_list("rendered_value", flat=True)
            .first()
        )
        result_data = (
            AnswerOutput.objects.filter(run_id=run_id)
            .order_by("created_at")
            .values_list("value_json", flat=True)
            .first()
        )
        return {
            "status": "COMPLETED"
            if result.result_kind == RunResultKind.ANSWERED.value
            else "WAITING_FOR_CLARIFICATION",
            "answer": answer_text,
            "result_data": result_data,
            "error": None,
        }

    def record_terminal_error(
        self, run_id: str, error: str = "already_terminal"
    ) -> None:
        record_worker_runtime_error(
            run_id=run_id,
            error_code=error,
            message=error,
        )

    def expire_running_lease(self, run_id: str) -> None:
        RunWorkItem.objects.filter(run_id=run_id).update(
            status="RUNNING",
            lease_expires_at=timezone.now() - timedelta(seconds=1),
            attempt_count=1,
            active_attempt=1,
        )


def _migrated_fastapi_project(tmp_path: Path) -> Path:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi>=0.1']\n",
        encoding="utf-8",
    )
    app_dir = root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    return root
