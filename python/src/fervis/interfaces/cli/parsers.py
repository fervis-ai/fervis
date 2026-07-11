"""Argparse construction for the Fervis CLI surface."""

from __future__ import annotations

import argparse
from pathlib import Path

from fervis.interfaces.cli.contracts import FervisOutputFormat
from fervis.lineage.views.detail import LineageRenderDetail


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_init_parser(subparsers)
    _add_catalog_parser(subparsers)
    _add_config_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_migrate_parser(subparsers)
    _add_auth_parser(subparsers)
    _add_models_parser(subparsers)
    _add_explain_parser(subparsers)
    _add_goldset_parser(subparsers)
    _add_inspect_parser(subparsers)
    _add_project_parser(subparsers)
    _add_runtime_parser(subparsers)
    _add_sources_parser(subparsers)
    _add_usage_parser(subparsers)
    _add_worker_parser(subparsers)
    return command_parser


def project_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_project_parser(subparsers)
    return command_parser


def init_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_init_parser(subparsers)
    return command_parser


def catalog_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_catalog_parser(subparsers)
    return command_parser


def doctor_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_doctor_parser(subparsers)
    return command_parser


def migrate_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_migrate_parser(subparsers)
    return command_parser


def auth_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_auth_parser(subparsers)
    return command_parser


def models_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_models_parser(subparsers)
    return command_parser


def config_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_config_parser(subparsers)
    return command_parser


def sources_parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="fervis")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    _add_sources_parser(subparsers)
    return command_parser


def command_name(argv: tuple[str, ...]) -> str:
    if not argv:
        return "fervis"
    first = argv[0]
    if (
        first
        in {
            "config",
            "auth",
            "goldset",
            "inspect",
            "model",
            "project",
            "runtime",
            "sources",
        }
        and len(argv) > 1
        and not argv[1].startswith("-")
    ):
        return f"{first}.{argv[1]}"
    return first


def is_runtime_ask_argv(argv: tuple[str, ...]) -> bool:
    return len(argv) >= 2 and argv[0] == "runtime" and argv[1] == "ask"


def is_worker_argv(argv: tuple[str, ...]) -> bool:
    return len(argv) >= 1 and argv[0] == "worker"


def output_format(args: argparse.Namespace) -> FervisOutputFormat:
    return FervisOutputFormat(args.format)


def lineage_detail(args: argparse.Namespace) -> LineageRenderDetail:
    value = getattr(args, "detail", None)
    if value:
        return LineageRenderDetail(value)
    return LineageRenderDetail.COMPACT


def comma_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _add_init_parser(subparsers) -> None:
    init = subparsers.add_parser("init", help="Generate Fervis project config")
    init.add_argument("--framework", choices=("django", "fastapi", "flask"))
    init.add_argument(
        "--app",
        help="Flask app object or factory import path, for example app:create_app.",
    )
    init.add_argument(
        "--source-prefix",
        action="append",
        default=[],
        help="Flask source path prefix exposed to Fervis. Repeat for multiple prefixes.",
    )
    init.add_argument(
        "--blueprint",
        action="append",
        default=[],
        help="Optional Flask blueprint name to expose. Repeat for multiple blueprints.",
    )
    init.add_argument(
        "--app-factory",
        help="FastAPI app factory import path, for example app.main:create_app.",
    )
    init.add_argument(
        "--path-prefixes",
        help="Comma-separated FastAPI source path prefixes when inference is not possible.",
    )
    init.add_argument(
        "--yes",
        action="store_true",
        help="Write generated files without prompting.",
    )


def _add_catalog_parser(subparsers) -> None:
    subparsers.add_parser(
        "catalog",
        help="Show configured host API sources exposed to Fervis",
    )


def _add_config_parser(subparsers) -> None:
    config = subparsers.add_parser("config", help="Inspect or edit Fervis config")
    nested = config.add_subparsers(dest="config_command", required=True)
    show = nested.add_parser("show", help="Show the loaded Fervis config")
    show.add_argument("--env", dest="env")
    get = nested.add_parser("get", help="Read one supported config path")
    get.add_argument("path")
    get.add_argument("--env", dest="env")
    set_parser = nested.add_parser("set", help="Patch one supported config path")
    set_parser.add_argument("path")
    set_parser.add_argument("value")
    set_parser.add_argument("--env", dest="env")
    nested.add_parser("upgrade", help="Upgrade Fervis JSON config files")


def _add_doctor_parser(subparsers) -> None:
    doctor = subparsers.add_parser(
        "doctor",
        help="Validate Fervis project wiring",
        description=(
            "Validate Fervis config, framework mounting, source catalog, host auth, "
            "persistence, and runtime readiness."
        ),
    )
    doctor.add_argument(
        "--probe-read-context-key",
        help=(
            "Optional real host read context key used to verify one configured "
            "read through the host API adapter."
        ),
    )


def _add_migrate_parser(subparsers) -> None:
    subparsers.add_parser("migrate", help="Apply Fervis-owned persistence migrations")


def _add_worker_parser(subparsers) -> None:
    worker = subparsers.add_parser("worker", help="Process queued Fervis runs")
    worker.add_argument("--once", action="store_true", default=False)
    worker.add_argument("--worker-id", default="fervis-worker")
    worker.add_argument("--batch-size", type=int, default=1)
    worker.add_argument("--lease-seconds", type=int, default=300)
    worker.add_argument("--sleep-seconds", type=float, default=1.0)


def _add_auth_parser(subparsers) -> None:
    auth = subparsers.add_parser("auth", help="Configure host auth execution")
    nested = auth.add_subparsers(dest="auth_command", required=True)
    configure = nested.add_parser(
        "configure",
        help="Generate schema-backed Fervis host auth config",
    )
    configure.add_argument(
        "--framework",
        choices=("django-drf", "django", "fastapi", "flask"),
    )
    configure.add_argument(
        "--security-mode",
        choices=("principal_reauthorization",),
        default="principal_reauthorization",
    )
    configure.add_argument(
        "--transport-mode",
        choices=("in_process", "http"),
        required=True,
    )
    configure.add_argument("--principal-dependency")
    configure.add_argument(
        "--principal-source",
        choices=("flask-login", "flask-g", "callable"),
    )
    configure.add_argument("--principal-id-attr")
    configure.add_argument("--principal-resolver")
    configure.add_argument("--base-url-env")
    configure.add_argument("--request-overlay-source")
    configure.add_argument("--auth-query-param", action="append", default=[])
    configure.add_argument(
        "--capture-credential-header",
        action="append",
        default=[],
        help="Capture and replay this request auth header for host reads.",
    )
    configure.add_argument(
        "--credential-key-env",
        default="FERVIS_READ_CREDENTIAL_KEY",
        help="Environment variable containing the credential encryption key.",
    )
    configure.add_argument("--credential-ttl-seconds", type=int, default=900)
    configure.add_argument("--env", dest="env")


def _add_models_parser(subparsers) -> None:
    subparsers.add_parser("models", help="Show supported Fervis model providers")
    model = subparsers.add_parser("model", help="Manage the active Fervis model")
    nested = model.add_subparsers(dest="model_command", required=True)
    allow = nested.add_parser("allow", help="Allow a provider:model_key in config")
    allow.add_argument("model_ref")
    use = nested.add_parser("use", help="Set model.default")
    use.add_argument("model_ref")
    use.add_argument("--env", dest="env")


def _add_explain_parser(subparsers) -> None:
    explain = subparsers.add_parser("explain", help="Show answer lineage")
    explain.add_argument("answer_id", nargs="?")
    explain.add_argument("--question-id")
    explain.add_argument("--run-id")
    explain.add_argument("--conversation-id")
    explain.add_argument("--question", help="Scope a conversation view to one question")
    explain.add_argument("--answer-output")
    explain.add_argument("--fact")
    explain.add_argument("--step")
    explain.add_argument("--run", help="Scope a question/conversation view to one run")
    explain.add_argument("--errors", action="store_true")
    explain.add_argument("--error", action="store_true")
    explain.add_argument(
        "--inputs",
        action="store_true",
        help="Show end-user input lineage for the selected answer lineage",
    )
    _add_detail_arguments(explain)
    explain.add_argument(
        "--format",
        choices=(FervisOutputFormat.AGENT.value, FervisOutputFormat.TEXT.value),
        default=FervisOutputFormat.AGENT.value,
        help=(
            "Output format. agent emits structured JSON; text emits "
            "human-readable prose."
        ),
    )


def _add_usage_parser(subparsers) -> None:
    usage = subparsers.add_parser("usage", help="Show model-call usage lineage")
    usage.add_argument("answer_id", nargs="?")
    usage.add_argument("--question-id")
    usage.add_argument("--run-id")
    usage.add_argument("--conversation-id")
    usage.add_argument("--step")
    usage.add_argument("--provider")
    usage.add_argument("--model")
    usage.add_argument("--usage-kind")
    _add_detail_arguments(usage)
    usage.add_argument(
        "--format",
        choices=(FervisOutputFormat.AGENT.value, FervisOutputFormat.TEXT.value),
        default=FervisOutputFormat.AGENT.value,
        help=(
            "Output format. agent emits structured JSON; text emits "
            "human-readable prose."
        ),
    )


def _add_goldset_parser(subparsers) -> None:
    goldset = subparsers.add_parser("goldset", help="Run a Fervis goldset suite")
    nested = goldset.add_subparsers(dest="goldset_command", required=True)
    run = nested.add_parser("run", help="Run host-owned goldset cases")
    run.add_argument("--suite-path", "--suite", dest="suite_path")
    run.add_argument("--case-ids")
    run.add_argument("--limit", type=int)
    run.add_argument("--tenant-id")
    run.add_argument("--principal-id")
    run.add_argument("--model", dest="model_key")
    run.add_argument("--ledger-file")
    run.add_argument("--wait-seconds", type=float, default=60.0)
    run.add_argument("--determinism-runs", type=int, default=1)
    run.add_argument(
        "--enforce-structured-determinism",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument("--attempts", type=int, default=1)
    run.add_argument("--retry-sleep-seconds", type=float, default=300.0)
    run.add_argument(
        "--retry-provider-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


def _add_inspect_parser(subparsers) -> None:
    inspect = subparsers.add_parser("inspect", help="Inspect runtime artifacts")
    nested = inspect.add_subparsers(dest="inspect_command", required=True)
    prompts = nested.add_parser("prompts", help="Inspect captured model prompts")
    prompts.add_argument("--run-id", required=True)
    prompts.add_argument(
        "--format",
        choices=(FervisOutputFormat.AGENT.value, FervisOutputFormat.TEXT.value),
        default=FervisOutputFormat.AGENT.value,
        help=(
            "Output format. agent emits structured JSON; text emits "
            "human-readable prose."
        ),
    )
    prompts.add_argument(
        "--viewer-format",
        choices=("raw", "html"),
        default="raw",
        help="Generated prompt-viewer artifact format.",
    )
    prompts.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".goldset-runs/prompt-viewer/latest"),
    )
    prompts.add_argument("--title", default="Fervis Prompt Viewer")
    prompts.add_argument("--open", action="store_true")
    artifact = nested.add_parser("artifact", help="Print one persisted artifact")
    artifact.add_argument("artifact_id")
    artifact.add_argument(
        "--format",
        choices=(FervisOutputFormat.AGENT.value, FervisOutputFormat.TEXT.value),
        default=FervisOutputFormat.AGENT.value,
        help=(
            "Output format. agent emits structured JSON; text emits "
            "human-readable prose."
        ),
    )


def _add_project_parser(subparsers) -> None:
    project = subparsers.add_parser("project", help="Inspect Fervis project wiring")
    nested = project.add_subparsers(dest="project_command", required=True)
    nested.add_parser("inspect", help="Show detected project metadata")


def _add_sources_parser(subparsers) -> None:
    sources = subparsers.add_parser("sources", help="Manage exposed Fervis sources")
    nested = sources.add_subparsers(dest="sources_command", required=True)
    add = nested.add_parser("add", help="Add an explicit source declaration")
    source_kind = add.add_subparsers(dest="source_kind", required=True)
    django = source_kind.add_parser("django-app", help="Expose Django app APIs")
    django.add_argument("name")
    django.add_argument("--app-modules", required=True)
    django.add_argument("--path-prefixes", required=True)
    fastapi = source_kind.add_parser("fastapi-app", help="Expose a FastAPI app")
    fastapi.add_argument("name")
    fastapi.add_argument("--import-paths", required=True)
    fastapi.add_argument("--path-prefixes", required=True)
    flask = source_kind.add_parser("flask-app", help="Expose a Flask app")
    flask.add_argument("name")
    flask.add_argument(
        "--app",
        required=True,
        help="Flask app object or factory import path, for example app:create_app.",
    )
    flask.add_argument(
        "--source-prefix",
        action="append",
        default=[],
        required=True,
        help="Flask source path prefix exposed to Fervis. Repeat for multiple prefixes.",
    )
    flask.add_argument(
        "--blueprint",
        action="append",
        default=[],
        help="Optional Flask blueprint name to expose. Repeat for multiple blueprints.",
    )


def _add_runtime_parser(subparsers) -> None:
    runtime = subparsers.add_parser("runtime", help="Run Fervis runtime operations")
    nested = runtime.add_subparsers(dest="runtime_command", required=True)
    ask = nested.add_parser("ask", help="Ask a factual question")
    ask.add_argument("question")
    ask.add_argument("--tenant-id", required=True)
    ask.add_argument("--principal-id", required=True)
    ask.add_argument("--conversation-id")
    ask.add_argument("--question-id")
    ask.add_argument("--base-run-id")
    ask.add_argument("--clarification-id")
    ask.add_argument("--model", dest="model_key")
    ask.add_argument("--idempotency-key")
    ask.add_argument("--max-budget-usd")
    ask.add_argument("--max-thinking-tokens", type=int)
    ask.add_argument(
        "--wait",
        nargs="?",
        const=60.0,
        default=0.0,
        type=float,
        metavar="SECONDS",
        help="Follow queued progress for up to SECONDS; --wait uses 60.",
    )


def _add_detail_arguments(command_parser) -> None:
    detail = command_parser.add_mutually_exclusive_group()
    detail.add_argument(
        "--compact",
        action="store_const",
        const=LineageRenderDetail.COMPACT.value,
        dest="detail",
        default=LineageRenderDetail.COMPACT.value,
    )
    detail.add_argument(
        "--verbose",
        action="store_const",
        const=LineageRenderDetail.VERBOSE.value,
        dest="detail",
    )
    detail.add_argument(
        "--debug",
        action="store_const",
        const=LineageRenderDetail.DEBUG.value,
        dest="detail",
    )
