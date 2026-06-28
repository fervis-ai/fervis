"""`fervis explain` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.commands.scopes import root_from_args, step_key
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
from fervis.interfaces.cli.rendering import agent_explain_payload
from fervis.lineage.enums import RunStepKey
from fervis.lineage.views.explain import (
    ExplainView,
    ExplainViewService,
    LineageSlice,
)


def explain_result(
    args: argparse.Namespace, *, ports: FervisCliPorts
) -> FervisCommandResult:
    view = _explain_view(args, ports=ports)
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
            payload=agent_explain_payload(view, options=options),
            view_kind=view_kind,
        )
    return FervisCommandResult(
        kind=FervisCommandKind.EXPLAIN,
        payload=view,
        view_kind=view_kind,
        render_options=options,
    )


def _explain_view(args: argparse.Namespace, *, ports: FervisCliPorts) -> ExplainView:
    root = root_from_args(args)
    service = ExplainViewService(
        lineage_query=ports.lineage_query,
        observability_query=ports.observability_query,
    )
    parsed_step_key = step_key(args.step)
    lineage_slice = LineageSlice(
        answer_output=args.answer_output,
        fact_filter=args.fact,
        step_key=parsed_step_key,
        errors_only=bool(args.errors or args.error),
    )
    return _explain_root_view(
        service,
        root,
        question_scope=args.question,
        run_scope=args.run,
        step_key=parsed_step_key,
        lineage_slice=lineage_slice,
    )


def _explain_root_view(
    service: ExplainViewService,
    root: FervisRoot,
    *,
    question_scope: str | None,
    run_scope: str | None,
    step_key: RunStepKey | None,
    lineage_slice: LineageSlice,
) -> ExplainView:
    if root.kind is FervisRootKind.ANSWER:
        return service.for_answer(
            root.root_id,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    if root.kind is FervisRootKind.QUESTION:
        return _explain_question_root(
            service,
            root.root_id,
            run_scope=run_scope,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    if root.kind is FervisRootKind.RUN:
        return service.for_run(
            root.root_id,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    if root.kind is FervisRootKind.CONVERSATION:
        return _explain_conversation_root(
            service,
            root.root_id,
            question_scope=question_scope,
            run_scope=run_scope,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    raise ValueError("explain requires an answer id or explicit root flag")


def _explain_question_root(
    service: ExplainViewService,
    question_id: str,
    *,
    run_scope: str | None,
    step_key: RunStepKey | None,
    lineage_slice: LineageSlice,
) -> ExplainView:
    if run_scope:
        return service.for_question_run(
            question_id,
            run_scope,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    return service.for_question(
        question_id,
        step_key=step_key,
        lineage_slice=lineage_slice,
    )


def _explain_conversation_root(
    service: ExplainViewService,
    conversation_id: str,
    *,
    question_scope: str | None,
    run_scope: str | None,
    step_key: RunStepKey | None,
    lineage_slice: LineageSlice,
) -> ExplainView:
    if question_scope:
        return service.for_conversation_question(
            conversation_id,
            question_scope,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    if run_scope:
        return service.for_conversation_run(
            conversation_id,
            run_scope,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    return service.for_conversation(
        conversation_id,
        step_key=step_key,
        lineage_slice=lineage_slice,
    )
