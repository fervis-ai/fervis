"""`fervis explain` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.commands.lineage import lineage_view
from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
    FervisOutputFormat,
    FervisRenderOptions,
    FervisViewKind,
)
from fervis.interfaces.cli.parsers import lineage_detail, output_format
from fervis.interfaces.cli.rendering import agent_lineage_payload


def explain_result(
    args: argparse.Namespace, *, ports: FervisCliPorts
) -> FervisCommandResult:
    view = lineage_view(args, ports=ports)
    options = FervisRenderOptions(
        answer_output=args.answer_output,
        fact_filter=args.fact,
        step=args.step,
        errors_only=args.errors or args.error,
        inputs_only=args.inputs,
        detail=lineage_detail(args),
        output_format=output_format(args),
    )
    view_kind = FervisViewKind.INPUT_LINEAGE if args.inputs else FervisViewKind.LINEAGE
    if options.output_format is FervisOutputFormat.AGENT:
        return command_envelope_result(
            kind=FervisCommandKind.EXPLAIN,
            command="explain",
            project=ports.project,
            payload_schema="fervis-explain-result.v0.1",
            payload=agent_lineage_payload(view, options=options),
            view_kind=view_kind,
        )
    return FervisCommandResult(
        kind=FervisCommandKind.EXPLAIN,
        payload=view,
        view_kind=view_kind,
        render_options=options,
    )
