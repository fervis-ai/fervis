"""SQL storage writability checks for Fervis doctor."""

from __future__ import annotations

import uuid

from fervis.interfaces.agent.actions import run_migrate_action
from fervis.host_api.contracts.authority import ReadContextRef
from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection
from fervis.project.persistence.contracts import PersistenceCheck
from fervis.lineage.recorder_core import LineageRecorder
from fervis.lineage.enums import QuestionRunKind, RunTriggerKind
from fervis.lineage.run_spine import (
    QuestionRunStart,
    QuestionRunStartRequest,
    QuestionStart,
    record_question_run_start,
)
from fervis.questions import (
    AskRequest,
    ExecutionMode,
    QuestionPrincipal,
)
from fervis.questions.ports import (
    LookupExecutionRequest,
    LookupExecutionResult,
    ProgramExecutionRequest,
    QuestionLookupPort,
    QuestionProgramPort,
    ModelAssistedRunSpec,
    RunSubmission,
)

from .engine import resolve_sql_storage_target
from .lineage_store import SQLLineageRecorderStore
from .question_run_ports import sql_question_service
from .question_run_ports import SQLQuestionRunSequenceStore
from .transaction import rollback_sql_transaction
from .work_items import SQLWorkItemQueue


def sql_storage_writability_checks(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
) -> list[PersistenceCheck]:
    target = resolve_sql_storage_target(
        project=project,
        loaded_config=loaded_config,
    )
    return [
        _check(
            "persistence.lineage_writable",
            "Fervis lineage storage accepts rolled-back run spine writes.",
            lambda: _dry_run_lineage(target.engine),
            target.engine,
        ),
        _check(
            "persistence.queue_writable",
            "Fervis queue storage accepts rolled-back work items.",
            lambda: _dry_run_queue(target.engine),
            target.engine,
        ),
        _check(
            "persistence.question_run_dry_run",
            "Fervis question-run lifecycle can create rolled-back queued runs.",
            lambda: _dry_run_question_lifecycle(target.engine),
            target.engine,
        ),
    ]


def _check(id_: str, message: str, probe, engine) -> PersistenceCheck:
    try:
        with rollback_sql_transaction(engine):
            probe()
    except Exception as exc:
        return PersistenceCheck(
            id=id_,
            passed=False,
            message=str(exc) or exc.__class__.__name__,
            fix=run_migrate_action(),
        )
    return PersistenceCheck(id=id_, passed=True, message=message)


def _dry_run_question_lifecycle(engine) -> None:
    service = sql_question_service(
        engine=engine,
        lookup=_DoctorLookup(),
        program=_DoctorProgram(),
        adapter_ref="fervis_doctor",
    )
    suffix = uuid.uuid4().hex
    result = service.ask(
        AskRequest(
            question="doctor storage dry run",
            principal=QuestionPrincipal(
                principal_id=f"doctor-principal-{suffix}",
                tenant_id=f"doctor-tenant-{suffix}",
            ),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id=f"doctor-conversation-{suffix}",
            idempotency_key=f"doctor-{suffix}",
        )
    )
    if result.status != "QUEUED":
        raise RuntimeError(f"unexpected doctor dry-run status: {result.status}")


def _dry_run_lineage(engine) -> None:
    suffix = uuid.uuid4().hex
    record_question_run_start(
        QuestionRunStartRequest(
            question=QuestionStart(
                conversation_id=f"doctor-lineage-conversation-{suffix}",
                tenant_id=f"doctor-lineage-tenant-{suffix}",
                read_context_ref=ReadContextRef(scheme="anonymous"),
                question_id=f"doctor-lineage-question-{suffix}",
                origin_message_ref=f"doctor-lineage-message-{suffix}",
                question="doctor lineage dry run",
            ),
            run=QuestionRunStart(
                question_id=f"doctor-lineage-question-{suffix}",
                run_id=f"doctor-lineage-run-{suffix}",
                kind=QuestionRunKind.MODEL_ASSISTED,
                trigger_kind=RunTriggerKind.INITIAL,
                adapter_ref="fervis_doctor",
                runtime_version="doctor",
            ),
        ),
        sequence_store=SQLQuestionRunSequenceStore(engine),
        recorder=LineageRecorder(SQLLineageRecorderStore(engine)),
    )


def _dry_run_queue(engine) -> None:
    suffix = uuid.uuid4().hex
    SQLWorkItemQueue(engine).enqueue_run_work_item(
        submission=RunSubmission(
            conversation_id=f"doctor-queue-conversation-{suffix}",
            tenant_id=f"doctor-queue-tenant-{suffix}",
            question_id=f"doctor-queue-question-{suffix}",
            run_id=f"doctor-queue-run-{suffix}",
            principal=QuestionPrincipal(
                principal_id=f"doctor-queue-principal-{suffix}",
                tenant_id=f"doctor-queue-tenant-{suffix}",
            ),
            spec=ModelAssistedRunSpec(
                integrated_question="doctor queue dry run",
                provider=None,
                model_key="DOCTOR",
            ),
            execution_mode=ExecutionMode.QUEUED,
        )
    )


class _DoctorLookup(QuestionLookupPort):
    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del request, progress_sink
        raise RuntimeError("doctor dry run must not execute lookup")


class _DoctorProgram(QuestionProgramPort):
    def run_program(
        self,
        request: ProgramExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del request, progress_sink
        raise RuntimeError("doctor dry run must not execute a program")
