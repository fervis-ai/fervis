from __future__ import annotations

import json

from fervis.lineage.enums import ArtifactKind, ModelCallStatus, RunStepKey, RunStepKind
from fervis.lineage.model_calls import ModelCallCapture, model_call_audit_write
from fervis.lineage.recorder import RunStepWrite


def test_model_call_schema_artifact_preserves_submitted_property_order() -> None:
    schema = {
        "type": "object",
        "properties": {
            "answer_requests_count": {"type": "integer"},
            "answer_requests": {"type": "array"},
            "question_inputs": {"type": "array"},
        },
    }
    audit = model_call_audit_write(
        run_id="run_1",
        step=RunStepWrite(
            step_id="step_1",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        ),
        capture=ModelCallCapture(
            provider="openai",
            model_key="openai:test",
            status=ModelCallStatus.SUCCEEDED,
            duration_ms=1,
            system_prompt="system",
            prompt_text="prompt",
            provider_schema=schema,
            tool_specs=(),
            usage={},
            submitted_payload={},
        ),
    )

    artifact = next(
        item for item in audit.artifacts if item.artifact_kind is ArtifactKind.SCHEMA
    )
    properties = json.loads(artifact.content)["properties"]

    assert list(properties) == [
        "answer_requests_count",
        "answer_requests",
        "question_inputs",
    ]
