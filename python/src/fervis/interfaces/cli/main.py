"""Framework-neutral Fervis CLI entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fervis.interfaces.cli.contracts import FervisCliPorts
from fervis.interfaces.cli.dispatch import (
    run_catalog_command,
    run_auth_command,
    run_blocked_command,
    run_config_command,
    run_doctor_command,
    run_fervis,
    run_help,
    run_init_command,
    run_migrate_command,
    run_models_command,
    run_project_command,
    run_sources_command,
)
from fervis.project import discover_project
from fervis.project.configuration import (
    ConfigProblem,
    load_fervis_project_config,
)
from fervis.project.django_runtime import django_project_runtime


PROJECT_COMMAND = "project"
CATALOG_COMMAND = "catalog"
CONFIG_COMMAND = "config"
AUTH_COMMAND = "auth"
INIT_COMMAND = "init"
DOCTOR_COMMAND = "doctor"
MIGRATE_COMMAND = "migrate"
SOURCES_COMMAND = "sources"
MODELS_COMMAND = "models"
MODEL_COMMAND = "model"
WORKER_COMMAND = "worker"


def main(argv: tuple[str, ...] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)
    project = discover_project(_invocation_cwd())
    if args[:1] == (AUTH_COMMAND,):
        return run_auth_command(args, project=project)
    if args[:1] == (CATALOG_COMMAND,):
        return run_catalog_command(args, project=project)
    if args[:1] == (CONFIG_COMMAND,):
        return run_config_command(args, project=project)
    if args[:1] == (INIT_COMMAND,):
        return run_init_command(args, project=project)
    if args[:1] == (DOCTOR_COMMAND,):
        return run_doctor_command(args, project=project)
    if args[:1] == (MIGRATE_COMMAND,):
        return run_migrate_command(args, project=project)
    if args[:1] in {(MODELS_COMMAND,), (MODEL_COMMAND,)}:
        return run_models_command(args, project=project)
    if args[:1] == (PROJECT_COMMAND,):
        return run_project_command(args, project=project)
    if args[:1] == (SOURCES_COMMAND,):
        return run_sources_command(args, project=project)
    if _is_help_request(args):
        return run_help(args)
    if _uses_sql_runtime_storage(args):
        return _sql_runtime_main(args, project=project)
    if project.framework != "django":
        return run_blocked_command(
            args,
            project=project,
            reason=(
                f"Fervis CLI runtime commands are not wired for "
                f"{project.framework} projects yet."
            ),
        )

    from fervis.interfaces.cli.django import main as django_main

    return django_main(args, project=project)


def _sql_runtime_main(args: tuple[str, ...], *, project) -> int:
    try:
        with django_project_runtime(project):
            loaded = load_fervis_project_config(project)
            if isinstance(loaded, ConfigProblem):
                return run_blocked_command(
                    args,
                    project=project,
                    reason=loaded.message,
                )
            from fervis.storage.sql.bundle import sql_storage_bundle
            from fervis.storage.sql.work_items import SQLWorkItemQueue
            from fervis.interfaces.common.admission import ConfiguredModelPolicy
            from fervis.run_work.queued_execution import LocalQueuedRunFollower
            from fervis.run_work.worker import (
                RunWorkBatchProcessor,
                RunWorkServiceWorker,
            )

            bundle = sql_storage_bundle(project=project, loaded_config=loaded)
            return run_fervis(
                args,
                ports=FervisCliPorts(
                    lineage_query=bundle.lineage_query,
                    observability_query=bundle.observability_query,
                    prompt_capture_query=bundle.prompt_capture_query,
                    questions=bundle.questions,
                    project=project,
                    question_run_follower=LocalQueuedRunFollower(
                        run_work=bundle.run_work,
                        work_queue=SQLWorkItemQueue(bundle.engine),
                    ),
                    run_worker=RunWorkBatchProcessor(
                        worker=RunWorkServiceWorker(bundle.run_work),
                        work_queue=SQLWorkItemQueue(bundle.engine),
                    ),
                    model_policy=ConfiguredModelPolicy.from_config(
                        loaded.config.model
                    ),
                ),
            )
    except RuntimeError as error:
        return run_blocked_command(args, project=project, reason=str(error))


def _uses_sql_runtime_storage(args: tuple[str, ...]) -> bool:
    if not args:
        return False
    if args[:1] in {("explain",), ("inspect",), ("usage",)}:
        return True
    if args[:1] == ("runtime",):
        return len(args) > 1 and args[1] == "ask"
    if args[:1] == (WORKER_COMMAND,):
        return True
    if args[:1] == ("goldset",):
        return True
    if args[:1] == ("inspect",):
        return len(args) > 1 and args[1] in {"artifact", "prompts"}
    return False


def _invocation_cwd() -> Path:
    return Path(os.environ.get("FERVIS_INVOCATION_CWD") or Path.cwd())


def _is_help_request(args: tuple[str, ...]) -> bool:
    return any(arg in {"-h", "--help"} for arg in args)


if __name__ == "__main__":
    raise SystemExit(main())
