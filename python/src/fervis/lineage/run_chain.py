"""Shared run-chain traversal for lineage-backed views."""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar


class RunChainNode(Protocol):
    run_id: str
    previous_run_id: str | None
    trigger_clarification_response_run_id: str | None


RunNodeT = TypeVar("RunNodeT", bound=RunChainNode)


def run_chain_ids(
    run_id: str,
    *,
    get_run: Callable[[str], RunNodeT | None],
    missing: Callable[[str], Exception],
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    _append_run_chain(
        run_id, get_run=get_run, missing=missing, output=output, seen=seen
    )
    return tuple(output)


def _append_run_chain(
    run_id: str,
    *,
    get_run: Callable[[str], RunNodeT | None],
    missing: Callable[[str], Exception],
    output: list[str],
    seen: set[str],
) -> None:
    if run_id in seen:
        return
    run = get_run(run_id)
    if run is None:
        raise missing(run_id)
    seen.add(run_id)
    for prerequisite in _run_prerequisites(run):
        _append_run_chain(
            prerequisite,
            get_run=get_run,
            missing=missing,
            output=output,
            seen=seen,
        )
    output.append(run_id)


def _run_prerequisites(run: RunChainNode) -> tuple[str, ...]:
    return tuple(
        run_id
        for run_id in (run.previous_run_id, run.trigger_clarification_response_run_id)
        if run_id
    )
