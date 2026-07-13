from __future__ import annotations

import pytest

from fervis.interfaces.agent.actions import (
    add_schema_metadata_action,
    choose_framework_init_action,
    chmod_action,
    configure_auth_action,
    edit_config_action,
    fix_schema_cardinality_action,
    inspect_prompt_index_action,
    inspect_question_action,
    inspect_run_action,
    install_dependencies_action,
    provide_clarification_action,
    resolve_blocked_edits_action,
    run_doctor_action,
    run_doctor_probe_action,
    run_init_action,
    run_migrate_action,
    set_env_action,
)


def test_agent_actions_are_self_describing() -> None:
    actions = [
        run_doctor_action(),
        run_doctor_probe_action(),
        run_migrate_action(),
        run_init_action("fastapi"),
        choose_framework_init_action(),
        resolve_blocked_edits_action(),
        edit_config_action(),
        install_dependencies_action("sentry_sdk"),
        set_env_action("OPENAI_API_KEY"),
        chmod_action("/tmp/fervis"),
        configure_auth_action(framework="fastapi"),
        configure_auth_action(framework="django"),
        fix_schema_cardinality_action("list_orders"),
        inspect_question_action("question-1"),
        inspect_run_action("run-1"),
        inspect_prompt_index_action("/tmp/prompts/index.html"),
        provide_clarification_action(
            "conversation-1",
            question_id="question-1",
            run_id="run-1",
            clarification_id="clarification-1",
            tenant_id="tenant-1",
            principal_id="principal-1",
        ),
    ]

    for action in actions:
        assert action["kind"]
        assert action["description"]


def test_probe_doctor_action_names_the_required_key() -> None:
    action = run_doctor_probe_action()

    assert action == {
        "kind": "command",
        "command": "fervis doctor --probe-read-context-key <read-context-key>",
        "description": (
            "Runs Fervis diagnostics with a real host read context key so Fervis can "
            "verify one configured host read. Replace <read-context-key> with an "
            "existing principal or delegated capability key from the host API."
        ),
    }


def test_fervis_actions_use_canonical_installed_command() -> None:
    assert run_doctor_action()["command"] == "fervis doctor"
    assert run_migrate_action()["command"] == "fervis migrate"


def test_schema_cardinality_action_names_all_supported_flask_schema_surfaces() -> None:
    action = fix_schema_cardinality_action("list_orders")

    description = str(action["description"])
    assert "OpenAPI/Swagger" in description
    assert "Marshmallow" in description
    assert "JSON:API" in description
    assert "Flask-AppBuilder" in description


def test_fastapi_schema_action_uses_framework_native_response_contracts() -> None:
    action = add_schema_metadata_action(
        "list_orders",
        framework_kind="fastapi",
    )

    assert action == {
        "kind": "add_schema_metadata",
        "endpoint": "list_orders",
        "description": (
            "Declare this FastAPI route's response through response_model or an "
            "equivalent precise return annotation, then rerun fervis catalog."
        ),
    }


def test_provide_clarification_action_requires_real_clarification_id() -> None:
    with pytest.raises(ValueError, match="clarification_id is required"):
        provide_clarification_action(
            "conversation-1",
            question_id="question-1",
            run_id="run-1",
            clarification_id="",
            tenant_id="tenant-1",
            principal_id="principal-1",
        )

    with pytest.raises(ValueError, match="clarification_id is required"):
        provide_clarification_action(
            "conversation-1",
            question_id="question-1",
            run_id="run-1",
            clarification_id=None,
            tenant_id="tenant-1",
            principal_id="principal-1",
        )
