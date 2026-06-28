"""Shared root and filter parsing for lineage-oriented CLI commands."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.contracts import FervisRoot, FervisRootKind
from fervis.lineage.enums import ModelUsageKind, RunStepKey


def root_from_args(args: argparse.Namespace) -> FervisRoot:
    roots = _provided_roots(args)
    if len(roots) != 1:
        raise ValueError(
            "provide exactly one root: answer id, --question-id, --run-id, or --conversation-id"
        )
    root = roots[0]
    _validate_root_scopes(root, args)
    return root


def step_key(value: str | None) -> RunStepKey | None:
    if not value:
        return None
    try:
        return RunStepKey(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in RunStepKey)
        raise ValueError(
            f"unknown step {value!r}; expected one of: {allowed}"
        ) from error


def usage_kind(value: str | None) -> ModelUsageKind | None:
    if not value:
        return None
    try:
        return ModelUsageKind(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in ModelUsageKind)
        raise ValueError(
            f"unknown usage kind {value!r}; expected one of: {allowed}"
        ) from error


def _provided_roots(args: argparse.Namespace) -> list[FervisRoot]:
    roots: list[FervisRoot] = []
    if args.answer_id:
        roots.append(FervisRoot(FervisRootKind.ANSWER, args.answer_id))
    if args.question_id:
        roots.append(FervisRoot(FervisRootKind.QUESTION, args.question_id))
    if args.run_id:
        roots.append(FervisRoot(FervisRootKind.RUN, args.run_id))
    if args.conversation_id:
        roots.append(FervisRoot(FervisRootKind.CONVERSATION, args.conversation_id))
    return roots


def _validate_root_scopes(root: FervisRoot, args: argparse.Namespace) -> None:
    question_scope = getattr(args, "question", None)
    run_scope = getattr(args, "run", None)
    if question_scope and run_scope and root.kind is FervisRootKind.CONVERSATION:
        raise ValueError("use either --question or --run, not both")
    if question_scope:
        _validate_question_scope(root)
    if run_scope:
        _validate_run_scope(root)


def _validate_question_scope(root: FervisRoot) -> None:
    if root.kind is not FervisRootKind.CONVERSATION:
        raise ValueError("--question can only scope --conversation-id")


def _validate_run_scope(root: FervisRoot) -> None:
    if root.kind is FervisRootKind.ANSWER:
        raise ValueError("--run can only scope --question-id or --conversation-id")
    if root.kind is FervisRootKind.RUN:
        raise ValueError("--run cannot scope --run-id")
