"""`fervis usage` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.commands.scopes import (
    root_from_args,
    step_key,
    usage_kind,
)
from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
    FervisOutputFormat,
    FervisRenderOptions,
    FervisRoot,
    FervisRootKind,
    FervisViewKind,
)
from fervis.interfaces.cli.parsers import lineage_detail, output_format
from fervis.interfaces.cli.rendering import usage_report_json
from fervis.observability.usage import (
    RuntimeUsageFilter,
    RuntimeUsageService,
)


def usage_result(
    args: argparse.Namespace, *, ports: FervisCliPorts
) -> FervisCommandResult:
    service = RuntimeUsageService(ports.observability_query)
    root = root_from_args(args)
    filters = RuntimeUsageFilter(
        step_key=step_key(args.step),
        provider=args.provider,
        model_key=args.model,
        usage_kind=usage_kind(args.usage_kind),
    )
    report = _usage_report_for_root(service, root, filters=filters)
    options = FervisRenderOptions(
        detail=lineage_detail(args),
        output_format=output_format(args),
    )
    if options.output_format is FervisOutputFormat.AGENT:
        return command_envelope_result(
            kind=FervisCommandKind.USAGE,
            command="usage",
            project=ports.project,
            payload_schema="fervis-usage-result.v0.1",
            payload=usage_report_json(report),
            view_kind=FervisViewKind.USAGE,
        )
    return FervisCommandResult(
        kind=FervisCommandKind.USAGE,
        payload=report,
        view_kind=FervisViewKind.USAGE,
        render_options=options,
    )


def _usage_report_for_root(
    service: RuntimeUsageService,
    root: FervisRoot,
    *,
    filters: RuntimeUsageFilter,
):
    if root.kind is FervisRootKind.ANSWER:
        return service.for_answer(root.root_id, filters=filters)
    if root.kind is FervisRootKind.QUESTION:
        return service.for_question(root.root_id, filters=filters)
    if root.kind is FervisRootKind.RUN:
        return service.for_run(root.root_id, filters=filters)
    if root.kind is FervisRootKind.CONVERSATION:
        return service.for_conversation(root.root_id, filters=filters)
    raise ValueError(f"unsupported root kind: {root.kind}")
