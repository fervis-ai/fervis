"""SQL-backed local lookup runtime composition."""

from __future__ import annotations

from sqlalchemy.engine import Engine

from fervis.lineage.recorder_core import LineageRecorder
from fervis.lookup.orchestration.question_lookup_port import (
    LookupServiceQuestionLookupPort,
)
from fervis.lookup.orchestration.question_program_port import AnswerProgramQuestionPort
from fervis.lookup.orchestration.program_service import AnswerProgramService
from fervis.lookup.orchestration.service import LookupService
from fervis.model_io.backbone.factory import build_provider_backbone
from fervis.model_io.models import ModelRef
from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection
from fervis.project.host_api_context import host_api_context_from_config
from fervis.questions.ports import QuestionLookupPort, QuestionProgramPort

from .lineage_store import SQLLineageRecorderStore
from .observability_query import SQLObservabilityQuery
from .terminal import run_has_terminal_result


def sql_configured_lookup_port(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
    engine: Engine,
) -> QuestionLookupPort:
    provider_name = ModelRef.parse(
        loaded_config.config.model.default_model_ref
    ).provider
    lookup_service = LookupService(
        provider_backbone=build_provider_backbone(provider_name),
        host_api_context=host_api_context_from_config(
            project=project,
            loaded_config=loaded_config,
        ),
        observability_query=SQLObservabilityQuery(engine),
        lineage_recorder=LineageRecorder(SQLLineageRecorderStore(engine)),
    )
    return LookupServiceQuestionLookupPort(
        lookup_service=lookup_service,
        terminal_lineage_recorded=lambda request: run_has_terminal_result(
            engine,
            request.run_id,
        ),
    )


def sql_configured_program_port(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
    engine: Engine,
) -> QuestionProgramPort:
    return AnswerProgramQuestionPort(
        program_service=AnswerProgramService(
            host_api_context=host_api_context_from_config(
                project=project,
                loaded_config=loaded_config,
            ),
            lineage_recorder=LineageRecorder(SQLLineageRecorderStore(engine)),
        ),
        terminal_lineage_recorded=lambda request: run_has_terminal_result(
            engine,
            request.run_id,
        ),
    )
