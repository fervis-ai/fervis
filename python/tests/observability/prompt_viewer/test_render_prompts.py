from __future__ import annotations

import json

import pytest

from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
)
from fervis.lineage.recorder import (
    ConversationWrite,
    ModelCallUsageWrite,
    ModelCallWrite,
    QuestionRunWrite,
    QuestionWrite,
    RunArtifactWrite,
    RunStepWrite,
)
from fervis.observability.prompt_viewer.render_prompts import (
    ModelTurnCapture,
    PromptInspectionFormat,
    RunOption,
    build_prompt_inspection_document,
    load_model_turn_captures,
    parse_jsonish,
    render_prompt_inspection,
    render_static_site,
    split_prompt_sections,
)
from fervis.observability.django_prompt_captures import DjangoPromptCaptureQuery


def test_split_prompt_sections_discovers_named_sections():
    sections = split_prompt_sections(_prompt_with_json_section())

    assert [section.title for section in sections] == [
        "Preamble",
        "Question",
        "Catalog Selection",
    ]


def test_split_prompt_sections_parses_json_section_value():
    sections = split_prompt_sections(_prompt_with_json_section())

    assert sections[2].value == {"facts": [{"id": "metric_total"}]}


def test_parse_jsonish_accepts_leading_json_with_prose_tail():
    assert parse_jsonish('{"metric": "count"}\n\nInstructions:\nExplain briefly.') == {
        "metric": "count"
    }


def test_render_static_site_writes_index_page(tmp_path):
    site = _render_site(tmp_path)

    assert "run-a" in (site / "index.html").read_text(encoding="utf-8")


def test_render_static_site_writes_run_page(tmp_path):
    site = _render_site(tmp_path)

    run_page = (site / "runs" / "run-a" / "index.html").read_text(encoding="utf-8")
    assert "fact_plan" in run_page


def test_render_static_site_writes_call_page_schema_section(tmp_path):
    site = _render_site(tmp_path)

    call_page = _call_page(site)
    assert "Typed Grammar / Schema" in call_page


def test_render_static_site_writes_call_page_argument_sections(tmp_path):
    site = _render_site(tmp_path)

    call_page = _call_page(site)
    assert "Submitted Arguments" in call_page
    assert "Parsed Arguments" in call_page


def test_render_static_site_uses_collapsible_call_sections(tmp_path):
    site = _render_site(tmp_path)

    assert "<details" in _call_page(site)


def test_render_prompt_inspection_raw_writes_agent_readable_json(tmp_path):
    document = build_prompt_inspection_document(
        runs=[RunOption(run_id="run-a")],
        turn_loader=lambda _run_id: [
            ModelTurnCapture(
                run_id="run-a",
                sequence=3,
                event_type="model_turn.completed",
                purpose="fact_plan",
                provider="test_provider",
                model_key="GPT_TEST",
                raw_system_prompt="System <instructions>",
                raw_prompt=_prompt_with_json_section(),
                raw_schema={"type": "object"},
                raw_tool_specs=[{"name": "FactPlan"}],
                arguments={"status": "needs_clarification"},
                parsed_arguments={"status": "needs_clarification"},
                usage={"inputTokens": 10},
                prompt_frame={"phase": "fact_plan"},
            )
        ],
    )

    index_path = render_prompt_inspection(
        document=document,
        output_dir=tmp_path / "raw",
        title="Prompt Viewer",
        output_format=PromptInspectionFormat.RAW,
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    [turn] = payload["runs"][0]["model_turns"]
    assert index_path.name == "index.json"
    assert turn["system_prompt"] == "System <instructions>"
    assert turn["prompt"] == _prompt_with_json_section()
    assert turn["prompt_sections"][2]["parsed_value"] == {
        "facts": [{"id": "metric_total"}]
    }
    assert turn["submitted_arguments"] == {"status": "needs_clarification"}


@pytest.mark.django_db
def test_load_model_turn_captures_reads_lineage_model_call_artifacts():
    _record_prompt_capture_lineage()

    [turn] = load_model_turn_captures(
        "run_prompt",
        prompt_capture_query=DjangoPromptCaptureQuery(),
    )

    assert turn.event_type == "model_turn.completed"
    assert turn.purpose == "source_binding"
    assert turn.selected_tool_name == "submit_source_binding"
    assert turn.raw_system_prompt == "System instructions"
    assert turn.raw_prompt == "Question:\nHow many stores are open?"
    assert turn.raw_schema == {"type": "object"}
    assert turn.raw_tool_specs == [{"name": "submit_source_binding"}]
    assert turn.arguments == {"source": "source_1"}
    assert turn.parsed_arguments == {"accepted": True}
    assert turn.usage == {"input_tokens": 11}
    assert turn.prompt_frame == {"phase": "source_binding"}
    assert turn.metadata["promptChars"] == 34
    assert turn.metadata["toolSpecChars"] == 41


def _record_prompt_capture_lineage() -> None:
    recorder = DjangoLineageRecorder()
    recorder.ensure_conversation(
        ConversationWrite(conversation_id="conversation_1", tenant_id="tenant_1")
    )
    recorder.record_question(
        QuestionWrite(
            question_id="question_1",
            conversation_id="conversation_1",
            conversation_sequence=1,
            original_question="How many stores are open?",
        )
    )
    recorder.start_run(
        QuestionRunWrite(
            run_id="run_prompt",
            question_id="question_1",
            run_number=1,
            trigger_kind=RunTriggerKind.INITIAL,
            integrated_question="How many stores are open?",
            adapter_ref="django_drf:test",
            runtime_version="test-runtime",
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_source_binding",
            run_id="run_prompt",
            sequence=2,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
            input_summary_json={"promptFrame": {"phase": "source_binding"}},
            output_summary_json={"selectedToolName": "submit_source_binding"},
        )
    )
    recorder.record_model_call(
        ModelCallWrite(
            model_call_id="call_1",
            run_id="run_prompt",
            step_id="step_source_binding",
            call_index=1,
            provider="openai",
            model_key="gpt-test",
            status=ModelCallStatus.SUCCEEDED,
            prompt_chars=34,
            schema_chars=18,
            tool_spec_chars=41,
        )
    )
    recorder.record_model_call_usage(
        ModelCallUsageWrite(
            usage_id="usage_1",
            run_id="run_prompt",
            model_call_id="call_1",
            usage_kind=ModelUsageKind.INPUT_TOKENS,
            quantity=11,
            unit=ModelUsageUnit.TOKENS,
            provider_usage_key="input_tokens",
        )
    )
    for artifact_kind, content in _prompt_capture_artifacts().items():
        recorder.record_artifact(
            RunArtifactWrite(
                artifact_id=f"artifact_{artifact_kind.value}",
                run_id="run_prompt",
                step_id="step_source_binding",
                model_call_id="call_1",
                artifact_kind=artifact_kind,
                content_hash=f"sha256:{artifact_kind.value}",
                content_type="application/json"
                if content.lstrip().startswith(("{", "["))
                else "text/plain",
                size_bytes=len(content),
                content=content,
            )
        )


def _prompt_capture_artifacts() -> dict[ArtifactKind, str]:
    return {
        ArtifactKind.SYSTEM_PROMPT: "System instructions",
        ArtifactKind.PROMPT: "Question:\nHow many stores are open?",
        ArtifactKind.SCHEMA: json.dumps({"type": "object"}),
        ArtifactKind.TOOL_SPEC: json.dumps([{"name": "submit_source_binding"}]),
        ArtifactKind.SUBMITTED_PAYLOAD: json.dumps({"source": "source_1"}),
        ArtifactKind.PARSED_PAYLOAD: json.dumps({"accepted": True}),
    }


def _prompt_with_json_section() -> str:
    return """Planner instruction.

Question:
What happened?

Catalog Selection:
{"facts": [{"id": "metric_total"}]}
"""


def _render_site(tmp_path):
    run = RunOption(run_id="run-a")
    turn = ModelTurnCapture(
        run_id="run-a",
        sequence=3,
        event_type="model_turn.completed",
        purpose="fact_plan",
        provider="test_provider",
        model_key="GPT_TEST",
        raw_prompt="Question:\nWhat happened?",
        raw_schema={"type": "object"},
        raw_tool_specs=[{"name": "FactPlan"}],
        arguments={"status": "needs_clarification"},
        parsed_arguments={"status": "needs_clarification"},
        usage={"inputTokens": 10},
        prompt_frame={"phase": "fact_plan"},
    )
    site = tmp_path / "site"
    render_static_site(
        runs=[run],
        output_dir=site,
        title="Prompt Viewer",
        turn_loader=lambda _run_id: [turn],
    )
    return site


def _call_page(site) -> str:
    return (site / "runs" / "run-a" / "0003-fact-plan.html").read_text(
        encoding="utf-8"
    )
