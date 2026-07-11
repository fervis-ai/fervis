from __future__ import annotations

import importlib

import json

import os

import subprocess

import sys

import threading

import time

from io import StringIO

from pathlib import Path

from types import SimpleNamespace

import pytest

from fervis.interfaces.agent.actions import (
    inspect_question_action,
    provide_clarification_action,
)

from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)

from fervis.interfaces.cli.dispatch import (
    evaluate_fervis,
    run_fervis,
)

from fervis.interfaces.cli.rendering import render_fervis_result

from fervis.interfaces.cli.runtime_ask import RuntimeAskEventStream
from fervis.interfaces.common.admission import ConfiguredModelPolicy

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunTriggerKind,
    RunStepKey,
)

from fervis.lineage.step_summary import (
    StepSummaryDetail,
    StepSummaryItem,
    step_summary_json,
)

from fervis.lineage.views.detail import LineageRenderDetail

from fervis.observability.query import (
    ObservabilityArtifact,
    ObservabilityArtifactContent,
    ModelCallDetailLevel,
    ObservabilityModelCall,
    ObservabilityQueryPort,
    ObservabilityRun,
    ObservabilityUsage,
)

from fervis.observability.prompt_captures import (
    ModelTurnPromptCapture,
    PromptCaptureQueryPort,
)

from fervis.questions import AskRequestLimits, AskResult

from fervis.project import ProjectInspection

from tests.testkit.algorithms.lineage import fixture_lineage_query

API_DIR = Path(__file__).resolve().parents[3]

def _agent_step(run: dict[str, object], step_key: str) -> dict[str, object]:
    for step in run["steps"]:
        if step["step_key"] == step_key:
            return step
    raise AssertionError(f"agent view missing step {step_key!r}")

def _command_envelope(rendered: str, *, command: str) -> dict[str, object]:
    envelope = json.loads(rendered)
    assert envelope["schema"] == "fervis-command-result.v0.1"
    assert envelope["command"] == command
    assert envelope["status"] == "succeeded"
    assert envelope["exit_code"] == 0
    return envelope

def _command_payload(rendered: str, *, command: str) -> dict[str, object]:
    return _command_envelope(rendered, command=command)["payload"]

def _jsonl_events(rendered: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in rendered.splitlines() if line.strip()]

def _blocked_envelope(rendered: str, *, command: str) -> dict[str, object]:
    envelope = json.loads(rendered)
    assert envelope["schema"] == "fervis-command-result.v0.1"
    assert envelope["command"] == command
    assert envelope["status"] == "blocked"
    assert envelope["exit_code"] == 2
    assert envelope["payload_schema"] == "fervis-command-error.v0.1"
    return envelope

def _ports(
    questions=None,
    question_run_limits=None,
    project=None,
    lineage_query=None,
    question_run_follower=None,
    run_worker=None,
    model_policy=None,
) -> FervisCliPorts:
    return FervisCliPorts(
        lineage_query=lineage_query or fixture_lineage_query(_lineage_dataset()),
        observability_query=_ObservabilityQuery(),
        prompt_capture_query=_PromptCaptureQuery(),
        questions=questions or _QuestionService(),
        project=project
        or ProjectInspection(
            framework="django",
            root_path=API_DIR,
            config_path=Path("config") / "fervis.py",
            expected_config_path=Path("config") / "fervis.py",
            confidence="high",
        ),
        question_run_limits=question_run_limits or AskRequestLimits(),
        question_run_follower=question_run_follower,
        model_policy=model_policy
        or ConfiguredModelPolicy(
            default_provider="openai",
            default_model_key="gpt-5.4-mini",
            allowed_model_keys_by_provider={
                "openai": frozenset({"gpt-5.4-mini"}),
                "anthropic": frozenset({"claude-haiku-4-5-20251001"}),
            },
        ),
        run_worker=run_worker,
    )

def _lineage_dataset() -> dict[str, object]:
    return {
        "conversations": [
            {"conversation_id": "conversation_1", "tenant_id": "tenant_1"},
            {"conversation_id": "conversation_2", "tenant_id": "tenant_1"},
        ],
        "questions": [
            {
                "question_id": "question_1",
                "conversation_id": "conversation_1",
                "conversation_sequence": 1,
                "original_question": "Which staff earned the most this month?",
            },
            {
                "question_id": "question_2",
                "conversation_id": "conversation_2",
                "conversation_sequence": 1,
                "original_question": "How many stores are open?",
            },
        ],
        "runs": [
            {
                "run_id": "run_1",
                "question_id": "question_1",
                "run_number": 1,
                "kind": "model_assisted",
                "trigger_kind": "initial",
            },
            {
                "run_id": "run_2",
                "question_id": "question_1",
                "run_number": 2,
                "kind": "model_assisted",
                "trigger_kind": "retry",
                "base_run_id": "run_1",
            },
            {
                "run_id": "run_3",
                "question_id": "question_2",
                "run_number": 1,
                "kind": "model_assisted",
                "trigger_kind": "initial",
            },
        ],
        "steps": [
            {
                "step_id": "step_question_contract",
                "run_id": "run_1",
                "sequence": 1,
                "step_key": "question_contract",
                "kind": "model_turn",
            },
            {
                "step_id": "step_source_binding",
                "run_id": "run_1",
                "sequence": 2,
                "step_key": "source_binding",
                "kind": "model_turn",
                "output_summary_json": step_summary_json(
                    StepSummaryItem(
                        text="calculated_pay: row-level payroll amount -> fits_requested_answer",
                        is_explanation=True,
                    ),
                    StepSummaryItem(
                        text="Binding: group=staff_id metric=calculated_pay function=sum"
                    ),
                    StepSummaryItem(
                        text="Source binding source_6: USE_SOURCE for fact_1",
                        detail=StepSummaryDetail.VERBOSE,
                    ),
                    StepSummaryItem(
                        text=(
                            "Population basis: Shift compensation rows match the "
                            "staff population for this month."
                        ),
                        detail=StepSummaryDetail.VERBOSE,
                        is_explanation=True,
                    ),
                    StepSummaryItem(
                        text=(
                            "Fulfillment basis answer_1/choice_staff_id: staff_id "
                            "is the canonical returned staff identity."
                        ),
                        detail=StepSummaryDetail.VERBOSE,
                        is_explanation=True,
                    ),
                ),
            },
            {
                "step_id": "step_error",
                "run_id": "run_2",
                "sequence": 1,
                "step_key": "source_binding",
                "kind": "model_turn",
            },
            {
                "step_id": "step_render",
                "run_id": "run_1",
                "sequence": 3,
                "step_key": "render",
                "kind": "deterministic",
            },
            {
                "step_id": "step_execute",
                "run_id": "run_1",
                "sequence": 9000,
                "step_key": "execute",
                "kind": "deterministic",
            },
        ],
        "run_results": [
            {
                "run_result_id": "run_result_1",
                "run_id": "run_1",
                "result_kind": "answered",
            },
            {
                "run_result_id": "run_result_2",
                "run_id": "run_2",
                "result_kind": "runtime_error",
            },
            {
                "run_result_id": "run_result_3",
                "run_id": "run_3",
                "result_kind": "factual_terminal",
            },
        ],
        "runtime_errors": [
            {
                "runtime_error_detail_id": "runtime_error_1",
                "run_id": "run_2",
                "run_result_id": "run_result_2",
                "failed_step_id": "step_error",
                "error_kind": "provider_runtime_failed",
                "message": "provider timed out",
            }
        ],
        "requested_facts": [
            {
                "requested_fact_id": "fact_1",
                "run_id": "run_1",
                "produced_by_step_id": "step_source_binding",
                "fact_key": "fact_1",
                "description": "staff member who earned the most compensation",
                "answer_expression_family": "ranked_selection",
            }
        ],
        "fact_results": [
            {
                "fact_result_id": "fact_result_1",
                "run_id": "run_1",
                "requested_fact_id": "fact_1",
                "produced_by_step_id": "step_source_binding",
                "result_kind": "answered",
            }
        ],
        "answers": [
            {
                "answer_id": "answer_1",
                "run_id": "run_1",
                "run_result_id": "run_result_1",
            }
        ],
        "answer_outputs": [
            {
                "answer_output_id": "answer_output_1",
                "run_id": "run_1",
                "answer_id": "answer_1",
                "fact_result_id": "fact_result_1",
                "output_key": "answer_1",
                "value_kind": "entity",
                "value_json": {"entity_type": "staff", "entity_id": "staff_9393"},
                "proof_node_refs_json": ["answer_output:fact_1:answer_1"],
            }
        ],
        "answer_presentations": [
            {
                "presentation_id": "presentation_1",
                "run_id": "run_1",
                "answer_id": "answer_1",
                "client_key": "default",
                "locale": "default",
                "presentation_kind": "text",
                "render_step_id": "step_render",
                "rendered_value": "Staff staff_9393 earned the most compensation.",
            }
        ],
        "catalog_endpoints": [
            {
                "catalog_endpoint_id": "66666666-6666-4666-8666-666666666666",
                "run_id": "run_1",
                "catalog_endpoint_key": "django_retail_ops_list_shift_compensation_list:test",
                "endpoint_name": "list_shift_compensation_list",
                "framework_kind": "django_drf",
                "source_namespace_kind": "django_app",
                "source_namespace_path_json": ["retail_ops"],
                "route_method": "GET",
                "route_path_template": "/v1/shift-compensations/",
                "route_name": "shift-compensation-list",
                "handler_ref": "apps.retail_ops.views.ShiftCompensationListView",
                "domain_resource_names_json": ["shift compensation"],
            },
            {
                "catalog_endpoint_id": "55555555-5555-4555-8555-555555555555",
                "run_id": "run_1",
                "catalog_endpoint_key": "django_retail_ops_list_payroll_summary:test",
                "endpoint_name": "list_payroll_summary",
                "framework_kind": "django_drf",
                "source_namespace_kind": "django_app",
                "source_namespace_path_json": ["retail_ops"],
                "route_method": "GET",
                "route_path_template": "/v1/payroll-summary/",
                "route_name": "payroll-summary-list",
                "handler_ref": "apps.retail_ops.views.PayrollSummaryView",
                "domain_resource_names_json": ["payroll summary"],
            },
        ],
        "source_reads": [
            {
                "source_read_id": "source_read_1",
                "run_id": "run_1",
                "step_id": "step_execute",
                "catalog_endpoint_id": "66666666-6666-4666-8666-666666666666",
                "args_json": {"month": "2026-06"},
                "status": "succeeded",
                "row_count": 3,
                "response_hash": "sha256:source",
            }
        ],
        "proof_graphs": [
            {
                "proof_graph_id": "proof_1",
                "run_id": "run_1",
                "fact_result_id": "fact_result_1",
                "compile_step_id": "step_compile",
                "execute_step_id": "step_execute",
                "payload_schema": "fervis.execution_proof_graph",
                "payload_schema_rev": 1,
                "payload_json": {
                    "nodes": [
                        {
                            "id": "relation:source_1",
                            "kind": "relation",
                            "proof_refs": ["source_read:source_read_1"],
                        },
                        {
                            "id": "endpoint_arg:source_1:month",
                            "kind": "endpoint_arg",
                            "proof_refs": ["known_input:month_1"],
                            "label": "month=2026-06",
                            "value": "2026-06",
                        },
                        {"id": "operation:op_1", "kind": "operation", "proof_refs": []},
                        {
                            "id": "answer_output:fact_1:answer_1",
                            "kind": "answer_output",
                            "proof_refs": [],
                        },
                    ],
                    "edges": [
                        {
                            "source": "endpoint_arg:source_1:month",
                            "target": "relation:source_1",
                            "role": "scopes",
                        },
                        {
                            "source": "relation:source_1",
                            "target": "operation:op_1",
                            "role": "input",
                        },
                        {
                            "source": "operation:op_1",
                            "target": "answer_output:fact_1:answer_1",
                            "role": "produces",
                        },
                    ],
                    "contributions": [
                        {
                            "origin": "explicit",
                            "label": "June 2026",
                            "node_refs": ["endpoint_arg:source_1:month"],
                            "proof_refs": ["known_input:month_1"],
                        },
                        {
                            "origin": "derived",
                            "label": "month=2026-06",
                            "node_refs": ["endpoint_arg:source_1:month"],
                            "proof_refs": ["known_input:month_1"],
                        },
                    ],
                },
            }
        ],
    }

class _ObservabilityQuery(ObservabilityQueryPort):
    def run_id_for_answer(self, answer_id: str) -> str | None:
        return "run_1" if answer_id == "answer_1" else None

    def run_by_id(self, run_id: str) -> ObservabilityRun | None:
        return ObservabilityRun(run_id=run_id) if run_id == "run_1" else None

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        return (run_id,) if run_id == "run_1" else ()

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        return ("run_1",) if question_id == "question_1" else ()

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        return ("run_1",) if conversation_id == "conversation_1" else ()

    def model_calls_for_run_ids(
        self, run_ids: tuple[str, ...], *, detail: ModelCallDetailLevel = "inspection"
    ) -> tuple[ObservabilityModelCall, ...]:
        if "run_1" not in run_ids:
            return ()
        return (_model_call(), _anthropic_model_call())

    def model_calls_for_run(
        self,
        run_id: str,
        step_key: RunStepKey | None = None,
        *,
        detail: ModelCallDetailLevel = "inspection",
    ) -> tuple[ObservabilityModelCall, ...]:
        calls = (_model_call(), _anthropic_model_call())
        if run_id != "run_1":
            return ()
        return tuple(
            call for call in calls if step_key is None or call.step_key == step_key
        )

    def artifact_content(self, artifact_id: str) -> ObservabilityArtifactContent | None:
        if artifact_id != "artifact_parsed":
            return None
        return ObservabilityArtifactContent(
            artifact_id="artifact_parsed",
            artifact_kind=ArtifactKind.PARSED_PAYLOAD,
            content_hash="sha256:parsed",
            content_type="application/json",
            size_bytes=17,
            content='{"answer": "parsed"}',
            storage_ref=None,
        )

class _PromptCaptureQuery(PromptCaptureQueryPort):
    def model_turn_prompt_captures_for_run(
        self, run_id: str
    ) -> tuple[ModelTurnPromptCapture, ...]:
        return ()

class _QuestionService:
    def __init__(
        self,
        result: AskResult | None = None,
        *,
        progress_events: tuple[dict[str, object], ...] = (),
        accepted_trigger: dict[str, object] | None = None,
    ) -> None:
        self.requests = []
        self.continue_requests = []
        self.result = result
        self.progress_events = progress_events
        self.accepted_trigger = accepted_trigger

    def ask(self, request, event_sink=None):
        self.requests.append(request)
        return self._emit_result(request, event_sink=event_sink)

    def continue_question(self, request, event_sink=None):
        self.continue_requests.append(request)
        return self._emit_result(request, event_sink=event_sink)

    def _emit_result(self, request, *, event_sink=None):
        result = self.result or AskResult(
            status="COMPLETED",
            conversation_id=getattr(request, "conversation_id", "") or "conversation_cli",
            question_id=getattr(request, "question_id", "") or "question_cli",
            run_id="run_cli",
            answer="42",
            result_data={"value": 42},
        )
        if event_sink is not None:
            if result.status == "ACTIVE_RUN_CONFLICT":
                event_sink.emit(
                    {
                        "event": "run.active_conflict",
                        "conversation_id": result.conversation_id,
                        "question_id": result.question_id,
                        "run_id": result.run_id,
                        "active_run_id": result.active_run_id or result.run_id,
                        "status": "ACTIVE_RUN_CONFLICT",
                        "error": {
                            "code": result.error or "active_run_conflict",
                            "message": result.error or "active_run_conflict",
                            "retryable": True,
                        },
                    }
                )
                return result
            accepted = {
                "event": "run.accepted",
                "conversation_id": result.conversation_id,
                "question_id": result.question_id,
                "run_id": result.run_id,
                "status": "QUEUED" if result.status == "QUEUED" else "RUNNING",
            }
            if self.accepted_trigger:
                accepted["trigger"] = dict(self.accepted_trigger)
            event_sink.emit(accepted)
            for event in self.progress_events:
                event_sink.emit(event)
            event_sink.emit(_terminal_event(result))
        return result

def _terminal_event(result: AskResult) -> dict[str, object]:
    if result.status == "COMPLETED":
        return {
            "event": "run.completed",
            "run_id": result.run_id,
            "status": result.status,
            "answer": result.answer,
            "result_data": result.result_data or {},
        }
    if result.status == "NEEDS_CLARIFICATION":
        return {
            "event": "run.needs_clarification",
            "conversation_id": result.conversation_id,
            "question_id": result.question_id,
            "run_id": result.run_id,
            "status": result.status,
            "clarifications": list(
                (result.result_data or {}).get("clarifications") or []
            ),
        }
    if result.status == "QUEUED":
        return {
            "event": "run.queued",
            "run_id": result.run_id,
            "status": result.status,
        }
    return {
        "event": "run.failed",
        "run_id": result.run_id,
        "status": result.status,
        "error": {
            "code": result.error or "runtime_ask_failed",
            "message": result.error or "runtime ask failed",
            "retryable": False,
        },
    }

class _BlockingEventedQuestionService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def ask(self, request, event_sink=None):
        if event_sink is not None:
            event_sink.emit(
                {
                    "event": "run.accepted",
                    "conversation_id": "conversation_stream",
                    "question_id": "question_stream",
                    "run_id": "run_stream",
                    "status": "RUNNING",
                }
            )
        self.started.set()
        self.release.wait(timeout=5)
        return AskResult(
            status="COMPLETED",
            conversation_id="conversation_stream",
            question_id="question_stream",
            run_id="run_stream",
            answer="42",
            result_data={"value": 42},
        )

class _ExplodingLineageQuery:
    def __getattr__(self, name):
        raise AssertionError(f"runtime ask should not read explain lineage: {name}")

class _QuestionRunFollower:
    def __init__(
        self,
        result: AskResult | None = None,
        *,
        progress_events: tuple[dict[str, object], ...] = (),
    ) -> None:
        self.result = result
        self.progress_events = progress_events
        self.calls: list[tuple[str, float]] = []

    def follow(self, result, *, event_sink=None, wait_seconds=0.0):
        self.calls.append((result.run_id, wait_seconds))
        followed = self.result or result
        if event_sink is not None:
            for event in self.progress_events:
                event_sink.emit(event)
            event_sink.emit(_terminal_event(followed))
        return followed

class _TimedOutQuestionRunFollower:
    def follow(self, result, *, event_sink=None, wait_seconds=0.0):
        del event_sink, wait_seconds
        return result

class _FailingQuestionRunFollower:
    def follow(self, result, *, event_sink=None, wait_seconds=0.0):
        del result, event_sink, wait_seconds
        raise RuntimeError("follower unavailable")

class _FailingQuestionService:
    def ask(self, request, event_sink=None):
        del event_sink
        del request
        raise RuntimeError("service unavailable")

class _ValidationFailingQuestionService:
    def ask(self, request, event_sink=None):
        del event_sink
        del request
        raise ValueError("ask request question must not be empty")

def _model_call() -> ObservabilityModelCall:
    return ObservabilityModelCall(
        model_call_id="call_1",
        run_id="run_1",
        step_id="step_source_binding",
        step_key=RunStepKey.SOURCE_BINDING,
        step_sequence=1,
        call_index=1,
        provider="openai",
        model_key="gpt-test",
        status=ModelCallStatus.SUCCEEDED,
        prompt_chars=100,
        schema_chars=50,
        tool_spec_chars=75,
        usage_rows=(
            ObservabilityUsage(
                usage_kind=ModelUsageKind.INPUT_TOKENS,
                quantity=20,
                unit=ModelUsageUnit.TOKENS,
                provider_usage_key="input_tokens",
                cost_micros=1000,
                currency="USD",
            ),
            ObservabilityUsage(
                usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                quantity=5,
                unit=ModelUsageUnit.TOKENS,
                provider_usage_key="output_tokens",
                cost_micros=500,
                currency="USD",
            ),
        ),
        artifacts=(
            ObservabilityArtifact(
                artifact_id="artifact_prompt",
                artifact_kind=ArtifactKind.PROMPT,
                content_hash="sha256:prompt",
                content_type="text/plain",
                size_bytes=11,
                has_content=True,
                storage_ref=None,
            ),
            ObservabilityArtifact(
                artifact_id="artifact_parsed",
                artifact_kind=ArtifactKind.PARSED_PAYLOAD,
                content_hash="sha256:parsed",
                content_type="application/json",
                size_bytes=17,
                has_content=True,
                storage_ref=None,
            ),
        ),
    )

def _anthropic_model_call() -> ObservabilityModelCall:
    return ObservabilityModelCall(
        model_call_id="call_2",
        run_id="run_1",
        step_id="step_fact_planning",
        step_key=RunStepKey.FACT_PLANNING,
        step_sequence=3,
        call_index=1,
        provider="anthropic",
        model_key="claude-test",
        status=ModelCallStatus.SUCCEEDED,
        prompt_chars=80,
        schema_chars=40,
        tool_spec_chars=55,
        usage_rows=(
            ObservabilityUsage(
                usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                quantity=2,
                unit=ModelUsageUnit.TOKENS,
                provider_usage_key="output_tokens",
                cost_micros=80,
                currency="USD",
            ),
        ),
    )

__all__ = [name for name in globals() if not name.startswith('__')]
