"""CLI adapter contracts and stable value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.types.enums import StrEnum
from typing import Protocol

from fervis.interfaces.cli.runtime_ask import (
    RuntimeAskFollower,
    RuntimeAskQuestions,
)
from fervis.interfaces.common.admission import ConfiguredModelPolicy
from fervis.lineage.views.detail import LineageRenderDetail
from fervis.lineage.views.query import LineageQueryPort
from fervis.observability.prompt_captures import PromptCaptureQueryPort
from fervis.observability.query import ObservabilityQueryPort
from fervis.project import ProjectInspection, discover_project
from fervis.questions import AskRequestLimits


class FervisCommandKind(StrEnum):
    AUTH = "auth"
    BLOCKED = "blocked"
    CATALOG = "catalog"
    CONFIG = "config"
    DEBUG = "debug"
    DEBUG_ARTIFACT = "debug_artifact"
    DEBUG_PROMPTS = "debug_prompts"
    DOCTOR = "doctor"
    EXPLAIN = "explain"
    GOLDSET = "goldset"
    INIT = "init"
    MIGRATE = "migrate"
    MODEL = "model"
    MODELS = "models"
    PROJECT_INSPECT = "project_inspect"
    RUNTIME_ASK = "runtime_ask"
    SOURCES = "sources"
    USAGE = "usage"
    WORKER = "worker"


class FervisViewKind(StrEnum):
    COMMAND = "command"
    LINEAGE = "lineage"
    INPUT_LINEAGE = "input_lineage"
    QUESTION_RUN = "question_run"
    USAGE = "usage"


class FervisOutputFormat(StrEnum):
    AGENT = "agent"
    TEXT = "text"


class FervisRootKind(StrEnum):
    ANSWER = "answer"
    QUESTION = "question"
    RUN = "run"
    CONVERSATION = "conversation"


class FervisRunWorkerCycle(Protocol):
    def to_payload(self) -> dict[str, object]: ...


class FervisRunWorker(Protocol):
    def process_once(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ) -> FervisRunWorkerCycle: ...


@dataclass(frozen=True)
class FervisRoot:
    kind: FervisRootKind
    root_id: str


@dataclass(frozen=True)
class FervisRenderOptions:
    answer_output: str | None = None
    fact_filter: str | None = None
    step: str | None = None
    errors_only: bool = False
    inputs_only: bool = False
    detail: LineageRenderDetail = LineageRenderDetail.COMPACT
    output_format: FervisOutputFormat = FervisOutputFormat.TEXT


@dataclass(frozen=True)
class FervisCommandResult:
    kind: FervisCommandKind
    payload: object
    view_kind: FervisViewKind
    render_options: FervisRenderOptions = field(default_factory=FervisRenderOptions)


@dataclass(frozen=True)
class FervisCliPorts:
    lineage_query: LineageQueryPort
    observability_query: ObservabilityQueryPort
    prompt_capture_query: PromptCaptureQueryPort
    questions: RuntimeAskQuestions
    project: ProjectInspection = field(default_factory=discover_project)
    question_run_limits: AskRequestLimits = field(default_factory=AskRequestLimits)
    question_run_follower: RuntimeAskFollower | None = None
    model_policy: ConfiguredModelPolicy = field(default_factory=ConfiguredModelPolicy)
    run_worker: FervisRunWorker | None = None
