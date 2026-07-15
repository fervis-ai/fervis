from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from fervis.delivery.answer_question import (
    AnswerQuestionGenerationError,
    AnswerQuestionService,
    GeneratedAnswerAudio,
    RecordedAnswerQuestion,
    answer_computation_xml_v1,
)
from fervis.lineage.enums import ContributionOrigin
from fervis.lineage.views.explanation import (
    AnswerExplanationView,
    LineageExplanationView,
)
from fervis.lineage.views.json_payload import view_json
from fervis.lineage.views.model import (
    CatalogEndpointView,
    ContributionView,
    ExecutionProofView,
    InputLineageResultView,
    InputLineageView,
    LineageRootKind,
    LineageTimelineView,
    ProofAppliedInputView,
    ProofComputationLinkView,
    SemanticConversationClauseView,
    SemanticInterpretedInputView,
    SemanticKnownInputView,
    SemanticRequestedFactView,
    SemanticResolverCandidateView,
    SourceReadView,
    StepSemanticView,
    StepDecisionItemView,
    StepDecisionView,
    TimelineAnswerOutputView,
    TimelineFactResultView,
    TimelineQuestionView,
    TimelineRunView,
    TimelineStepView,
)
from fervis.model_io.providers.chat_runtime import ChatProviderConfig
from fervis.model_io.providers.openai_answer_question import (
    OpenAIRealtimeAnswerQuestionGenerator,
)


def test_audio_delivery_uses_the_exact_desktop_inputs_and_verbose_explanation() -> (
    None
):
    explanation = _contextual_explanation()
    generator = _CapturingGenerator()

    audio = AnswerQuestionService(
        explanations=_StaticExplanationLoader(explanation),
        generator=generator,
    ).answer("run-current", _recorded_question())

    assert audio.data == b"spoken explanation"
    assert "fewest words that fully answer the question" in generator.instructions
    assert "never exceed 60 words" in generator.instructions
    assert "Conversation Resolution" in generator.instructions
    root = ET.fromstring(generator.input_text)
    assert root.tag == "AnswerComputation"
    assert [child.tag for child in root] == ["Inputs", "Explanation"]
    assert json.loads(root.findtext("Inputs", default="")) == view_json(
        explanation.inputs
    )
    assert json.loads(root.findtext("Explanation", default="")) == view_json(
        explanation.lineage.verbose
    )
    assert generator.question == _recorded_question()


def test_context_free_audio_identifies_requested_facts_without_claiming_memory() -> (
    None
):
    explanation = _contextual_explanation()
    run = explanation.lineage.verbose.questions[0].runs[0]
    first_step = replace(
        run.steps[0],
        semantic=StepSemanticView(
            requested_facts=(
                SemanticRequestedFactView(
                    requested_fact_id="fact-current",
                    description="completed sales total",
                ),
                SemanticRequestedFactView(
                    requested_fact_id="fact-count",
                    description="completed sales count",
                ),
            )
        ),
    )
    context_free_run = replace(
        run,
        activated_memory_ids=(),
        steps=(first_step, *run.steps[1:]),
    )
    timeline = replace(
        explanation.lineage.verbose,
        questions=(
            replace(
                explanation.lineage.verbose.questions[0],
                runs=(context_free_run,),
            ),
        ),
    )

    projected = replace(
        explanation,
        lineage=LineageExplanationView(compact=timeline, verbose=timeline),
        inputs=replace(
            explanation.inputs,
            results=(
                replace(
                    explanation.inputs.results[0],
                    fact_description="completed sales total",
                    explicit=("status completed",),
                ),
            ),
        ),
    )
    root = ET.fromstring(answer_computation_xml_v1(projected))

    assert json.loads(root.findtext("Inputs", default="")) == view_json(
        projected.inputs
    )
    assert json.loads(root.findtext("Explanation", default="")) == view_json(
        projected.lineage.verbose
    )


def test_openai_realtime_adapter_returns_a_playable_wav_from_one_response() -> None:
    pcm = b"\x00\x01\x02\x03" * 4
    connection = _FakeConnection(
        [
            SimpleNamespace(
                type="response.output_audio.delta",
                delta=base64.b64encode(pcm).decode("ascii"),
            ),
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(status="completed"),
            ),
        ]
    )
    client = _FakeClient(connection)
    generator = OpenAIRealtimeAnswerQuestionGenerator(
        provider_config=_provider_config(),
        client=client,
    )

    result = generator.generate(
        instructions="Speak briefly.",
        input_text="<answer />",
        question=_recorded_question(),
    )

    with wave.open(io.BytesIO(result.data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24_000
        assert wav.readframes(wav.getnframes()) == pcm
    assert connection.response.requests == [
        {
            "audio": {
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24_000},
                    "voice": "cedar",
                }
            },
            "conversation": "none",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "<answer />"},
                        {
                            "type": "input_audio",
                            "audio": base64.b64encode(_recorded_question().pcm).decode(
                                "ascii"
                            ),
                        },
                    ],
                }
            ],
            "instructions": "Speak briefly.",
            "max_output_tokens": 1000,
            "output_modalities": ["audio"],
        }
    ]
    assert client.realtime.connect_calls == [
        {
            "model": "gpt-realtime-2.1",
            "websocket_connection_options": {
                "open_timeout": 15,
                "close_timeout": 5,
            },
        }
    ]


def test_openai_realtime_adapter_fails_when_no_audio_completes() -> None:
    generator = OpenAIRealtimeAnswerQuestionGenerator(
        provider_config=_provider_config(),
        client=_FakeClient(
            _FakeConnection(
                [
                    SimpleNamespace(
                        type="response.done",
                        response=SimpleNamespace(status="failed"),
                    )
                ]
            )
        ),
    )

    with pytest.raises(
        AnswerQuestionGenerationError,
        match="did not complete",
    ):
        generator.generate(
            instructions="Speak.",
            input_text="<answer />",
            question=_recorded_question(),
        )


def test_openai_realtime_adapter_maps_quota_failure_to_actionable_message() -> None:
    generator = OpenAIRealtimeAnswerQuestionGenerator(
        provider_config=_provider_config(),
        client=_FakeClient(
            _FakeConnection(
                [
                    SimpleNamespace(
                        type="error",
                        error=SimpleNamespace(
                            message="insufficient_quota.insufficient_quota"
                        ),
                    )
                ]
            )
        ),
    )

    with pytest.raises(AnswerQuestionGenerationError) as raised:
        generator.generate(
            instructions="Speak.",
            input_text="<answer />",
            question=_recorded_question(),
        )

    assert raised.value.public_message == (
        "OpenAI quota is exhausted. Add credits or increase the project budget, "
        "then try again."
    )


class _StaticExplanationLoader:
    def __init__(self, explanation: AnswerExplanationView) -> None:
        self.explanation = explanation

    def for_run(self, run_id: str) -> AnswerExplanationView:
        assert run_id == "run-current"
        return self.explanation


class _CapturingGenerator:
    def __init__(self) -> None:
        self.instructions = ""
        self.input_text = ""
        self.question = None

    def generate(self, *, instructions: str, input_text: str, question):
        self.instructions = instructions
        self.input_text = input_text
        self.question = question
        return GeneratedAnswerAudio(b"spoken explanation")


def _recorded_question() -> RecordedAnswerQuestion:
    return RecordedAnswerQuestion(pcm=b"\x00\x00" * 4_800, duration_ms=200)


class _FakeResponse:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, *, response) -> None:
        self.requests.append(response)


class _FakeConnection:
    def __init__(self, events) -> None:
        self.events = events
        self.response = _FakeResponse()

    def __iter__(self):
        return iter(self.events)


class _FakeConnectionManager:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> _FakeConnection:
        return self.connection

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _FakeRealtime:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.connect_calls: list[dict[str, object]] = []

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        return _FakeConnectionManager(self.connection)


class _FakeClient:
    def __init__(self, connection: _FakeConnection) -> None:
        self.realtime = _FakeRealtime(connection)


def _provider_config() -> ChatProviderConfig:
    return ChatProviderConfig(
        provider_name="openai",
        model_name="gpt-5.4-mini",
        api_key_env_var="OPENAI_API_KEY",
        sdk_name="openai_chat_completions",
    )


def _contextual_explanation() -> AnswerExplanationView:
    source = SourceReadView(
        source_read_id="source-read-secret",
        step_id="step-execute",
        catalog_endpoint=CatalogEndpointView(
            catalog_endpoint_id="endpoint-secret",
            catalog_endpoint_key="sales-summary",
            endpoint_name="list_sales_summary",
            framework_kind="django",
            source_namespace_kind="viewset",
            source_namespace_path=("sales",),
        ),
        args={
            "endDate": "2026-09-30",
            "granularity": "day",
            "groupBy": "date",
            "startDate": "2026-07-01",
            "status": "COMPLETED",
        },
        row_count=4,
        response_hash="response-hash-secret",
        status="succeeded",
        completeness={"truncated": False},
        artifact_id="source-response-artifact",
    )
    proof = ExecutionProofView(
        proof_graph_id="proof-secret",
        evidence_handles=("source-read-secret",),
        endpoint_args=(),
        computation_links=(
            ProofComputationLinkView(
                source="daily-sales",
                target="answer-total",
                role="sum",
            ),
        ),
        computation_summaries=("Combined the daily aggregates into one total.",),
        debug_evidence_handles=(),
        debug_computation_links=(),
        contributions=(
            ContributionView(
                origin=ContributionOrigin.CONTEXTUAL,
                label="last quarter",
                node_refs=("daily-sales",),
            ),
        ),
        applied_inputs=(
            ProofAppliedInputView(
                handle="daily-sales",
                label="date range",
                action="2026-07-01 through 2026-09-30",
            ),
        ),
        source_reads=(source,),
    )
    steps = (
        TimelineStepView(
            step_id="step-conversation",
            step_key="conversation_resolution",
            sequence=1,
            semantic=StepSemanticView(
                conversation_clauses=(
                    SemanticConversationClauseView(
                        current_clause_text="how about this quarter?",
                        resolved_text="how much did we make in sales this quarter?",
                        resolved_values=("this quarter",),
                    ),
                )
            ),
        ),
        TimelineStepView(
            step_id="step-grounding",
            step_key="grounding",
            sequence=2,
            decisions=(
                StepDecisionView(
                    step_key="grounding",
                    lines=("Retained the completed-sales summary.",),
                    items=(
                        StepDecisionItemView(
                            text="Retained the completed-sales summary.",
                            is_explanation=True,
                            subject="sales/list_sales_summary",
                            disposition="RETAIN",
                            basis=(
                                "Daily completed-sales totals match the requested "
                                "measure."
                            ),
                        ),
                    ),
                ),
            ),
            semantic=StepSemanticView(
                known_inputs=(
                    SemanticKnownInputView(
                        input_id="input-secret",
                        text="this quarter",
                        kind="date_range",
                    ),
                ),
                resolver_candidates=(
                    SemanticResolverCandidateView(
                        input_id="input-secret",
                        resolver_read_id="calendar-secret",
                        resolver_label="calendar",
                        basis="The phrase names the active calendar quarter.",
                    ),
                ),
                interpreted_inputs=(
                    SemanticInterpretedInputView(
                        input_id="input-secret",
                        input_text="this quarter",
                        kind="date_range",
                        value="2026-07-01 to 2026-09-30",
                    ),
                ),
            ),
        ),
        TimelineStepView(
            step_id="step-execute",
            step_key="execute",
            sequence=3,
            source_reads=(source,),
            fact_results=(
                TimelineFactResultView(
                    fact_result_id="result-secret",
                    requested_fact_id="fact-secret",
                    result_kind="value",
                    proof=proof,
                ),
            ),
            answer_outputs=(
                TimelineAnswerOutputView(
                    fact_result_id="result-secret",
                    output_key="total_sales",
                    value_kind="decimal",
                    value="111155.00",
                    value_json={"kind": "decimal", "value": "111155.00"},
                    proof_node_refs=("answer-total",),
                    proof=proof,
                ),
            ),
        ),
    )
    run = TimelineRunView(
        run_id="run-current",
        run_number=2,
        kind="deterministic",
        trigger_kind="rerun",
        result_kind="answered",
        activated_memory_ids=("memory-secret",),
        memory_artifacts=(),
        steps=steps,
    )
    timeline = LineageTimelineView(
        root_kind=LineageRootKind.RUN,
        root_id="run-current",
        questions=(
            TimelineQuestionView(
                question_id="question-secret",
                conversation_id="conversation-secret",
                text="how about this quarter?",
                runs=(run,),
            ),
        ),
    )
    return AnswerExplanationView(
        inputs=InputLineageView(
            root_kind=LineageRootKind.RUN,
            root_id="run-current",
            results=(
                InputLineageResultView(
                    fact_result_id="result-secret",
                    requested_fact_id="fact-secret",
                    fact_description="total completed sales last quarter",
                    explicit=("last quarter",),
                    contextual=("last quarter from the prior exchange",),
                    applied=("date range 2026-07-01 through 2026-09-30",),
                ),
            ),
        ),
        lineage=LineageExplanationView(compact=timeline, verbose=timeline),
    )
