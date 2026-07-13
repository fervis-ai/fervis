"""Render Fervis CLI command results."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from decimal import Decimal
from fervis.types.enums import StrEnum

from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
    FervisRenderOptions,
    FervisViewKind,
)
from fervis.interfaces.cli.envelope import CommandEnvelope
from fervis.interfaces.cli.runtime_ask import RuntimeAskEventStream
from fervis.lineage.views.agent import (
    agent_input_lineage_view,
    agent_lineage_view,
)
from fervis.lineage.views.explain import ExplainView
from fervis.lineage.views.input_lineage import (
    input_lineage_view,
    render_input_lineage,
)
from fervis.lineage.views.render import render_lineage
from fervis.observability.query import (
    ObservabilityArtifact,
    ObservabilityArtifactContent,
)
from fervis.observability.render import UsageRenderDetail, render_usage_report
from fervis.observability.usage import RuntimeUsageReport
from fervis.observability.prompt_viewer.render_prompts import PromptViewerResult


def render_fervis_result(result: FervisCommandResult) -> str:
    if isinstance(result.payload, RuntimeAskEventStream):
        return "\n".join(
            json.dumps(_jsonable(event), sort_keys=True)
            for event in result.payload.events
        )
    if isinstance(result.payload, CommandEnvelope):
        return json.dumps(
            _jsonable(result.payload.to_payload()),
            indent=2,
            sort_keys=True,
        )
    if result.kind is FervisCommandKind.EXPLAIN:
        payload = result.payload
        if not isinstance(payload, ExplainView):
            raise ValueError("explain command requires an ExplainView payload")
        options = result.render_options
        if options.inputs_only:
            return render_input_lineage(
                input_lineage_view(
                    payload.lineage,
                    answer_output=options.answer_output,
                ),
                detail=options.detail,
            )
        return render_lineage(
            payload.timeline,
            answer_output=options.answer_output,
            fact_filter=options.fact_filter,
            step=options.step,
            errors_only=options.errors_only,
            detail=options.detail,
        )
    if result.kind is FervisCommandKind.USAGE:
        payload = result.payload
        if not isinstance(payload, RuntimeUsageReport):
            raise ValueError("usage command requires a RuntimeUsageReport payload")
        return render_usage_report(
            payload,
            detail=UsageRenderDetail(result.render_options.detail.value),
        )
    if result.kind is FervisCommandKind.INSPECT_PROMPTS:
        payload = result.payload
        if not isinstance(payload, PromptViewerResult):
            raise ValueError("inspect prompts requires a PromptViewerResult payload")
        return f"Wrote {payload.run_count} run(s) to {payload.index_path}"
    if result.kind is FervisCommandKind.INSPECT_ARTIFACT:
        payload = result.payload
        if not isinstance(payload, ObservabilityArtifactContent):
            raise ValueError(
                "inspect artifact requires an ObservabilityArtifactContent payload"
            )
        return render_artifact_content(payload)
    raise ValueError(f"unsupported command result kind: {result.kind}")


def agent_explain_payload(
    view: ExplainView,
    *,
    options: FervisRenderOptions,
) -> dict[str, object]:
    if options.inputs_only:
        input_view = input_lineage_view(
            view.lineage,
            answer_output=options.answer_output,
        )
        return {
            "view_kind": FervisViewKind.INPUT_LINEAGE.value,
            "detail": options.detail.value,
            "filters": _render_filters(options),
            "input_lineage": agent_input_lineage_view(
                input_view, detail=options.detail
            ),
        }
    lineage_view = agent_lineage_view(view.timeline, detail=options.detail)
    return {
        **lineage_view,
        "detail": options.detail.value,
        "filters": _render_filters(options),
    }


def render_artifact_content(artifact: ObservabilityArtifactContent) -> str:
    header = (
        f"Artifact {artifact.artifact_id} "
        f"({artifact.artifact_kind.value}, {artifact.content_type}, "
        f"{artifact.size_bytes} bytes)"
    )
    if artifact.content is not None:
        return f"{header}\n{artifact.content}"
    return f"{header}\nstorage_ref: {artifact.storage_ref or '<none>'}"


def artifact_content_json(artifact: ObservabilityArtifactContent) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact.artifact_kind.value,
        "content_hash": artifact.content_hash,
        "content_type": artifact.content_type,
        "size_bytes": artifact.size_bytes,
        "content": artifact.content,
        "storage_ref": artifact.storage_ref,
    }


def prompt_viewer_result_json(result: PromptViewerResult) -> dict[str, object]:
    return {
        "run_count": result.run_count,
        "index_path": str(result.index_path),
        "output_format": result.output_format.value,
    }


def usage_report_json(report: RuntimeUsageReport) -> dict[str, object]:
    return {
        "scope": report.scope.value,
        "scope_id": report.scope_id,
        "calls": report.calls,
        "duration_ms_total": report.duration_ms_total,
        "duration_ms_by_step": {
            step_key.value: duration
            for step_key, duration in sorted(
                report.duration_ms_by_step.items(),
                key=lambda item: item[0].value,
            )
        },
        "usage_totals": [
            {
                "usage_kind": usage_kind.value,
                "unit": unit.value,
                "quantity": quantity,
            }
            for (usage_kind, unit), quantity in sorted(
                report.usage_totals.items(),
                key=lambda item: (item[0][0].value, item[0][1].value),
            )
        ],
        "cost_micros_by_currency": report.cost_micros_by_currency,
        "unpriced_usage_count": report.unpriced_usage_count,
        "missing_cost_count": report.missing_cost_count,
        "pricing_versions": report.pricing_versions,
    }


def _render_filters(options: FervisRenderOptions) -> dict[str, object]:
    return {
        "answer_output": options.answer_output,
        "fact": options.fact_filter,
        "step": options.step,
        "errors_only": options.errors_only,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, RuntimeUsageReport):
        return _jsonable(usage_report_json(value))
    if isinstance(value, ObservabilityArtifactContent):
        return artifact_content_json(value)
    if isinstance(value, ObservabilityArtifact):
        return _artifact_json(value)
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _artifact_json(artifact: ObservabilityArtifact) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact.artifact_kind.value,
        "content_hash": artifact.content_hash,
        "content_type": artifact.content_type,
        "size_bytes": artifact.size_bytes,
        "has_content": artifact.has_content,
        "storage_ref": artifact.storage_ref,
    }
