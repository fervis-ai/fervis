"""Shared run-chain traversal for lineage-backed views."""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar


class RunChainNode(Protocol):
    @property
    def run_id(self) -> str: ...

    @property
    def base_run_id(self) -> str | None: ...


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
    return (run.base_run_id,) if run.base_run_id else ()
