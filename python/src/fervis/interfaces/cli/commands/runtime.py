"""`fervis runtime` command adapters."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
    FervisOutputFormat,
    FervisRenderOptions,
    FervisViewKind,
)
from fervis.interfaces.cli.runtime_ask import (
    RuntimeAskJsonlSink,
    RuntimeAskPorts,
    run_runtime_ask,
)


def runtime_ask_result(
    args: argparse.Namespace,
    *,
    ports: FervisCliPorts,
    event_sink: RuntimeAskJsonlSink | None = None,
) -> FervisCommandResult:
    stream = run_runtime_ask(
        args,
        ports=RuntimeAskPorts(
            questions=ports.questions,
            question_run_limits=ports.question_run_limits,
            project=ports.project,
            question_run_follower=ports.question_run_follower,
            model_policy=ports.model_policy,
        ),
        event_sink=event_sink,
    )
    return FervisCommandResult(
        kind=FervisCommandKind.RUNTIME_ASK,
        payload=stream,
        view_kind=FervisViewKind.QUESTION_RUN,
        render_options=FervisRenderOptions(output_format=FervisOutputFormat.AGENT),
    )
