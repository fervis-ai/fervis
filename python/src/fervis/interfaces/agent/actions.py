"""Agent-facing follow-up action payloads."""

from __future__ import annotations

from fervis.interfaces.agent.commands import Placeholder
from fervis.interfaces.agent.commands import commands, render_command


def external_command_action(command: str, *, description: str) -> dict[str, object]:
    return {
        "kind": "command",
        "command": command,
        "description": description,
    }


def fervis_command_action(command, *, description: str) -> dict[str, object]:
    return external_command_action(
        render_command(command),
        description=description,
    )


def run_doctor_action() -> dict[str, object]:
    return fervis_command_action(
        commands.doctor(),
        description=(
            "Runs Fervis diagnostics again to verify configuration, persistence, "
            "source catalog, auth, and runtime readiness."
        ),
    )


def run_doctor_probe_action() -> dict[str, object]:
    return fervis_command_action(
        commands.doctor(probe_read_context_key=Placeholder("read-context-key")),
        description=(
            "Runs Fervis diagnostics with a real host read context key so Fervis "
            "can verify one configured host read. Replace <read-context-key> with "
            "an existing principal or delegated capability key from the host API."
        ),
    )


def run_migrate_action() -> dict[str, object]:
    return fervis_command_action(
        commands.migrate(),
        description=(
            "Applies Fervis-owned persistence migrations, creating the local "
            "SQLite store when needed."
        ),
    )


def run_init_action(framework: str) -> dict[str, object]:
    return fervis_command_action(
        commands.init(framework=framework, yes=True),
        description=(
            "Generates the versioned Fervis config and installs the framework "
            "mount hook for the detected project."
        ),
    )


def choose_framework_init_action() -> dict[str, object]:
    django = render_command(commands.init(framework="django", yes=True))
    fastapi = render_command(
        commands.init(
            framework="fastapi",
            app_factory=Placeholder("module:function"),
            yes=True,
        )
    )
    return {
        "kind": "choose_framework",
        "description": (
            "Run Fervis init with the host framework explicitly, for example "
            f"`{django}` or `{fastapi}`."
        ),
    }


def resolve_blocked_edits_action() -> dict[str, object]:
    return {
        "kind": "resolve_blocked_edits",
        "description": (
            "Read payload.blocked_edits, make the requested project change, then "
            "rerun the command."
        ),
    }


def edit_config_action() -> dict[str, object]:
    return {
        "kind": "edit",
        "file": "config/fervis.json",
        "description": (
            "Edit the Fervis JSON config schema to fix the configuration "
            "problem reported in the diagnostic payload."
        ),
    }


def add_schema_metadata_action(
    endpoint_name: str,
    *,
    framework_kind: str,
) -> dict[str, object]:
    return {
        "kind": "add_schema_metadata",
        "endpoint": endpoint_name,
        "description": _schema_metadata_description(framework_kind),
    }


def _schema_metadata_description(framework_kind: str) -> str:
    if framework_kind == "fastapi":
        return (
            "Declare this FastAPI route's response through response_model or an "
            "equivalent precise return annotation, then rerun fervis catalog."
        )
    if framework_kind == "django":
        return (
            "Declare this Django REST Framework endpoint's response through its "
            "serializer contract, then rerun fervis catalog."
        )
    if framework_kind == "flask":
        return (
            "Expose this endpoint's response/query contract through a supported "
            "Flask surface: OpenAPI/Swagger, Marshmallow metadata, JSON:API "
            "resource/schema metadata, or Flask-AppBuilder metadata. For plain "
            "Flask routes, follow github.com/fervis-ai/fervis/python/flask/AGENTS.md."
        )
    return "Declare this host endpoint's response contract, then rerun fervis catalog."


def fix_schema_cardinality_action(endpoint_name: str) -> dict[str, object]:
    return {
        "kind": "fix_schema_cardinality",
        "endpoint": endpoint_name,
        "description": (
            "Update this route's response contract through a supported Flask "
            "surface: OpenAPI/Swagger, Marshmallow metadata, JSON:API "
            "resource/schema metadata, or Flask-AppBuilder metadata. The "
            "declared object-vs-array shape must match the JSON returned by "
            "the host API."
        ),
    }


def install_dependencies_action(module: str) -> dict[str, object]:
    return {
        "kind": "install_dependencies",
        "module": module,
        "command": "uv sync",
        "description": (
            f"Install the host project dependencies so Python can import {module!r} "
            "while building the Fervis source catalog."
        ),
    }


def set_env_action(name: str) -> dict[str, object]:
    return {
        "kind": "set_env",
        "name": name,
        "description": (
            f"Set the {name} environment variable, then rerun the Fervis command."
        ),
    }


def chmod_action(path: str) -> dict[str, object]:
    return {
        "kind": "chmod",
        "path": path,
        "description": (
            "Make this filesystem path writable and searchable by the current "
            "process, then rerun the Fervis command."
        ),
    }


def configure_auth_action(*, framework: str) -> dict[str, object]:
    if framework == "django":
        return fervis_command_action(
            commands.auth_configure(
                framework="django-drf",
                transport_mode="in_process",
            ),
            description=(
                "Generates config/fervis_auth.json for Django + DRF using the "
                "request user as the host principal."
            ),
        )
    if framework == "fastapi":
        return fervis_command_action(
            commands.auth_configure(
                principal_dependency=Placeholder("module:function"),
                principal_id_attr="id",
                principal_resolver=Placeholder("module:function"),
                transport_mode="in_process",
            ),
            description=(
                "Generates config/fervis_auth.json for FastAPI by recording the host "
                "dependency that returns the current principal and the resolver that "
                "reconstructs that principal later."
            ),
        )
    if framework == "flask":
        return fervis_command_action(
            commands.auth_configure(
                framework="flask",
                principal_source="flask-g",
                principal_id_attr="id",
                principal_resolver=Placeholder("module:function"),
                transport_mode="in_process",
            ),
            description=(
                "Generates config/fervis_auth.json for Flask by recording how "
                "Fervis captures the current request principal and the resolver "
                "that reconstructs that principal later."
            ),
        )
    raise ValueError(f"unsupported auth action framework: {framework}")


def inspect_question_action(
    question_id: str, *, debug: bool = False
) -> dict[str, object]:
    return {
        "kind": "inspect_question",
        "question_id": question_id,
        "command": render_command(
            commands.debug_question(question_id)
            if debug
            else commands.explain_question(question_id)
        ),
        "description": (
            "Inspects the question, including its current answer state and all "
            "runs attempted for that question."
        ),
    }


def inspect_run_action(run_id: str) -> dict[str, object]:
    return {
        "kind": "inspect_run",
        "run_id": run_id,
        "command": render_command(commands.debug_run(run_id)),
        "description": "Inspects one execution run for diagnostics.",
    }


def inspect_prompt_index_action(path: str) -> dict[str, object]:
    return {
        "kind": "inspect_prompt_index",
        "path": path,
        "description": (
            "Open the generated prompt-inspection index file for the captured "
            "model-turn artifacts."
        ),
    }


def provide_clarification_action(
    conversation_id: str,
    *,
    question_id: str | None = None,
    run_id: str | None = None,
    clarification_id: str | None = None,
    tenant_id: str | None = None,
    principal_id: str | None = None,
) -> dict[str, object]:
    clarification_value = str(clarification_id or "").strip()
    if not clarification_value:
        raise ValueError("clarification_id is required")
    tenant = tenant_id or "<tenant_id>"
    principal = principal_id or "<principal_id>"
    question = question_id or "<question_id>"
    run = run_id or "<run_id>"
    required_inputs = ["answer"]
    if question_id is None:
        required_inputs.append("question_id")
    if run_id is None:
        required_inputs.append("run_id")
    if tenant_id is None:
        required_inputs.append("tenant_id")
    if principal_id is None:
        required_inputs.append("principal_id")
    return {
        "kind": "provide_clarification",
        "conversation_id": conversation_id,
        "question_id": question_id or "",
        "run_id": run_id or "",
        "clarification_id": clarification_value,
        "command": render_command(
            commands.runtime_ask(
                Placeholder("answer", quoted=True),
                question_id=question,
                run_id=run,
                clarification_id=clarification_value,
                conversation_id=conversation_id,
                tenant_id=tenant,
                principal_id=principal,
            )
        ),
        "description": (
            "Continues the same question by submitting the clarification answer "
            "requested by Fervis."
        ),
        "required_inputs": required_inputs,
    }
