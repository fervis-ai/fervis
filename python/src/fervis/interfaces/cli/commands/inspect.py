"""`fervis inspect` command adapters."""

from __future__ import annotations

import argparse

from fervis.interfaces.agent.actions import inspect_prompt_index_action
from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
    FervisOutputFormat,
    FervisRenderOptions,
    FervisViewKind,
)
from fervis.interfaces.cli.parsers import output_format
from fervis.interfaces.cli.rendering import (
    artifact_content_json,
    prompt_viewer_result_json,
)
from fervis.observability.usage import ObservabilityRootNotFound


def inspect_prompts_result(
    args: argparse.Namespace, *, ports: FervisCliPorts
) -> FervisCommandResult:
    from fervis.observability.prompt_viewer.render_prompts import (
        PromptInspectionFormat,
        PromptViewerRequest,
        render_prompt_viewer,
    )

    prompt_format = PromptInspectionFormat(args.viewer_format)
    if args.open and prompt_format is not PromptInspectionFormat.HTML:
        raise ValueError("--open requires --viewer-format html")
    result = render_prompt_viewer(
        PromptViewerRequest(
            run_id=args.run_id,
            output_dir=args.output_dir,
            title=args.title,
            output_format=prompt_format,
            open_browser=args.open,
        ),
        prompt_capture_query=ports.prompt_capture_query,
    )
    selected_format = output_format(args)
    if selected_format is FervisOutputFormat.TEXT:
        return FervisCommandResult(
            kind=FervisCommandKind.INSPECT_PROMPTS,
            payload=result,
            view_kind=FervisViewKind.COMMAND,
            render_options=FervisRenderOptions(output_format=selected_format),
        )
    return command_envelope_result(
        kind=FervisCommandKind.INSPECT_PROMPTS,
        command="inspect.prompts",
        project=ports.project,
        payload_schema="fervis-prompt-inspection-result.v0.1",
        payload=prompt_viewer_result_json(result),
        view_kind=FervisViewKind.COMMAND,
        next_actions=[inspect_prompt_index_action(str(result.index_path))],
    )


def inspect_artifact_result(
    args: argparse.Namespace, *, ports: FervisCliPorts
) -> FervisCommandResult:
    artifact = ports.observability_query.artifact_content(args.artifact_id)
    if artifact is None:
        raise ObservabilityRootNotFound(f"artifact {args.artifact_id!r} was not found")
    selected_format = output_format(args)
    if selected_format is FervisOutputFormat.AGENT:
        return command_envelope_result(
            kind=FervisCommandKind.INSPECT_ARTIFACT,
            command="inspect.artifact",
            project=ports.project,
            payload_schema="fervis-artifact-content-result.v0.1",
            payload=artifact_content_json(artifact),
            view_kind=FervisViewKind.COMMAND,
        )
    return FervisCommandResult(
        kind=FervisCommandKind.INSPECT_ARTIFACT,
        payload=artifact,
        view_kind=FervisViewKind.COMMAND,
        render_options=FervisRenderOptions(output_format=selected_format),
    )
