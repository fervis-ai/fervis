"""Shared lineage-view composition for CLI presentation adapters."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.scopes import root_from_args, step_key
from fervis.interfaces.cli.contracts import FervisCliPorts, FervisRoot, FervisRootKind
from fervis.lineage.enums import RunStepKey
from fervis.lineage.views.explain import ExplainView, ExplainViewService, LineageSlice


def lineage_view(args: argparse.Namespace, *, ports: FervisCliPorts) -> ExplainView:
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
    return _root_view(
        service,
        root,
        question_scope=args.question,
        run_scope=args.run,
        step_key=parsed_step_key,
        lineage_slice=lineage_slice,
    )


def _root_view(
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
        if run_scope:
            return service.for_question_run(
                root.root_id,
                run_scope,
                step_key=step_key,
                lineage_slice=lineage_slice,
            )
        return service.for_question(
            root.root_id,
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
        if question_scope:
            return service.for_conversation_question(
                root.root_id,
                question_scope,
                step_key=step_key,
                lineage_slice=lineage_slice,
            )
        if run_scope:
            return service.for_conversation_run(
                root.root_id,
                run_scope,
                step_key=step_key,
                lineage_slice=lineage_slice,
            )
        return service.for_conversation(
            root.root_id,
            step_key=step_key,
            lineage_slice=lineage_slice,
        )
    raise ValueError("lineage view requires an answer id or explicit root flag")
