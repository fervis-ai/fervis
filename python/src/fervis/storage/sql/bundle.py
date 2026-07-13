"""Composition root for SQL-backed Fervis storage ports."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine

from fervis.host_api.context import HostApiContext
from fervis.lineage.views.query import LineageQueryPort
from fervis.observability.prompt_captures import PromptCaptureQueryPort
from fervis.observability.query import ObservabilityQueryPort
from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection
from fervis.project.host_api_context import host_api_context_from_config
from fervis.questions.service import QuestionService
from fervis.run_work.service import RunWorkService
from fervis.questions.ports import (
    LookupExecutionRequest,
    LookupExecutionResult,
    QuestionLookupPort,
)

from .engine import resolve_sql_storage_target
from .lineage_query import SQLLineageQuery
from .lookup_runtime import sql_configured_lookup_port, sql_configured_program_port
from .observability_query import SQLObservabilityQuery
from .prompt_captures import SQLPromptCaptureQuery
from .question_run_ports import sql_question_service, sql_run_work_service


@dataclass(frozen=True)
class SQLStorageBundle:
    engine: Engine
    host_api_context: HostApiContext
    lineage_query: LineageQueryPort
    observability_query: ObservabilityQueryPort
    prompt_capture_query: PromptCaptureQueryPort
    questions: QuestionService
    run_work: RunWorkService
    kind: str
    location: str

    def close(self) -> None:
        self.host_api_context.close()

    def __enter__(self) -> "SQLStorageBundle":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def sql_storage_bundle(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
    lookup: QuestionLookupPort | None = None,
) -> SQLStorageBundle:
    target = resolve_sql_storage_target(
        project=project,
        loaded_config=loaded_config,
    )
    host_api_context = host_api_context_from_config(
        project=project,
        loaded_config=loaded_config,
    )
    lookup_port = lookup or _LazyConfiguredLookup(
        project=project,
        loaded_config=loaded_config,
        engine=target.engine,
        host_api_context=host_api_context,
    )
    program_port = sql_configured_program_port(
        project=project,
        loaded_config=loaded_config,
        engine=target.engine,
        host_api_context=host_api_context,
    )
    return SQLStorageBundle(
        engine=target.engine,
        host_api_context=host_api_context,
        kind=target.kind,
        location=target.location,
        lineage_query=SQLLineageQuery(target.engine),
        observability_query=SQLObservabilityQuery(target.engine),
        prompt_capture_query=SQLPromptCaptureQuery(target.engine),
        questions=sql_question_service(
            engine=target.engine,
            lookup=lookup_port,
            program=program_port,
        ),
        run_work=sql_run_work_service(
            engine=target.engine,
            lookup=lookup_port,
            program=program_port,
        ),
    )


class _LazyConfiguredLookup(QuestionLookupPort):
    def __init__(
        self,
        *,
        project: ProjectInspection,
        loaded_config: LoadedFervisConfig,
        engine: Engine,
        host_api_context: HostApiContext,
    ) -> None:
        self.project = project
        self.loaded_config = loaded_config
        self.engine = engine
        self.host_api_context = host_api_context
        self._lookup: QuestionLookupPort | None = None

    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        return self._configured().run_lookup(request, progress_sink=progress_sink)

    def _configured(self) -> QuestionLookupPort:
        if self._lookup is None:
            self._lookup = sql_configured_lookup_port(
                project=self.project,
                loaded_config=self.loaded_config,
                engine=self.engine,
                host_api_context=self.host_api_context,
            )
        return self._lookup
