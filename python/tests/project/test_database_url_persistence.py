from alembic import command
from datetime import datetime, timezone

from sqlalchemy import inspect, select

from fervis.project.integration import DatabaseUrlPersistence
from fervis.project.persistence.alembic_runner import alembic_config
from fervis.project.persistence.database_url import DatabaseUrlPersistenceBackend
from fervis.project.persistence.contracts import MigrationStatus
from fervis.project.persistence.schema import metadata as current_metadata
from fervis.project.persistence.sqlite_engine import create_sqlite_engine
from fervis.project.persistence.schema_snapshots.v0001 import metadata as v1_metadata


def test_database_url_inspection_is_failure_total_for_malformed_url(monkeypatch):
    monkeypatch.setenv("FERVIS_DATABASE_URL", "not a url")
    backend = DatabaseUrlPersistenceBackend(config=DatabaseUrlPersistence())

    checks = backend.inspect()

    assert [(check.id, check.passed) for check in checks] == [
        ("persistence.target", False)
    ]


def test_database_url_persistence_rejects_ephemeral_in_memory_sqlite(monkeypatch):
    monkeypatch.setenv("FERVIS_DATABASE_URL", "sqlite:///:memory:")
    backend = DatabaseUrlPersistenceBackend(config=DatabaseUrlPersistence())

    checks = backend.inspect()
    migration = backend.migrate()

    assert [(check.id, check.passed) for check in checks] == [
        ("persistence.target", False)
    ]
    assert migration.status is MigrationStatus.BLOCKED


def test_database_url_migration_upgrades_revision_one_database(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "fervis.sqlite3"
    database_url = f"sqlite:///{database_path}"
    engine = create_sqlite_engine(database_url)
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "0001_initial")
    monkeypatch.setenv("FERVIS_DATABASE_URL", database_url)
    backend = DatabaseUrlPersistenceBackend(config=DatabaseUrlPersistence())

    migration = backend.migrate()
    columns = {
        item["name"] for item in inspect(engine).get_columns("fervis_run_work_item")
    }

    assert migration.status is MigrationStatus.APPLIED
    assert migration.pending_revisions == ["fervis.0002"]
    assert {"idempotency_authority_ref", "idempotency_scope"} <= columns
    assert all(check.passed for check in backend.inspect())


def test_revision_two_preserves_clarification_lineage(tmp_path, monkeypatch):
    database_path = tmp_path / "fervis.sqlite3"
    database_url = f"sqlite:///{database_path}"
    engine = create_sqlite_engine(database_url)
    now = datetime.now(timezone.utc)
    tables = v1_metadata.tables
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "0001_initial")
        connection.execute(
            tables["fervis_conversation"].insert(),
            {
                "conversation_id": "conversation-1",
                "tenant_id": "tenant-1",
                "read_context_ref": {},
                "origin_kind": "initial",
                "origin_ref": "",
                "created_at": now,
            },
        )
        connection.execute(
            tables["fervis_question"].insert(),
            {
                "question_id": "question-1",
                "conversation_id": "conversation-1",
                "conversation_sequence": 1,
                "origin_message_ref": "message-1",
                "original_question": "Which warehouse?",
                "created_at": now,
            },
        )
        connection.execute(
            tables["fervis_question_run"].insert(),
            {
                "run_id": "run-1",
                "question_id": "question-1",
                "run_number": 1,
                "kind": "model_assisted",
                "trigger_kind": "initial",
                "trigger_clarification_response_id": "",
                "adapter_ref": "test",
                "runtime_version": "test",
                "created_at": now,
            },
        )
        connection.execute(
            tables["fervis_run_step"].insert(),
            {
                "step_id": "step-1",
                "run_id": "run-1",
                "sequence": 1,
                "step_key": "question_contract",
                "attempt": 1,
                "scope_type": "run",
                "scope_id": "run-1",
                "kind": "model",
                "input_summary_json": {},
                "output_summary_json": {},
                "error_json": {},
                "created_at": now,
            },
        )
        connection.execute(
            tables["fervis_requested_fact"].insert(),
            {
                "requested_fact_id": "fact-1",
                "run_id": "run-1",
                "produced_by_step_id": "step-1",
                "fact_key": "fact-1",
                "description": "warehouse",
                "answer_expression_family": "direct",
                "requested_fact_json": {},
                "answer_requests_json": [],
                "created_at": now,
            },
        )
        connection.execute(
            tables["fervis_fact_result"].insert(),
            {
                "fact_result_id": "result-1",
                "run_id": "run-1",
                "requested_fact_id": "fact-1",
                "produced_by_step_id": "step-1",
                "result_kind": "needs_clarification",
                "evidence_refs_json": [],
                "payload_schema": "clarification",
                "payload_schema_rev": 1,
                "payload_json": {},
                "created_at": now,
            },
        )
        connection.execute(
            tables["fervis_clarification_request"].insert(),
            {
                "clarification_id": "clarification-1",
                "run_id": "run-1",
                "fact_result_id": "result-1",
                "step_id": None,
                "need": "target_reference",
                "reason": "multiple_matching_entities",
                "payload_json": {},
                "created_at": now,
            },
        )
        connection.execute(
            tables["fervis_clarification_response"].insert(),
            {
                "response_id": "response-1",
                "run_id": "run-1",
                "clarification_id": "clarification-1",
                "source_message_ref": "message-2",
                "selected_option_id": "warehouse-1",
                "response_text": "Warehouse 1",
                "evidence_ref": "user:message-2",
                "created_at": now,
            },
        )
    monkeypatch.setenv("FERVIS_DATABASE_URL", database_url)

    migration = DatabaseUrlPersistenceBackend(
        config=DatabaseUrlPersistence()
    ).migrate()

    request_table = current_metadata.tables["fervis_clarification_request"]
    response_table = current_metadata.tables["fervis_clarification_response"]
    with engine.connect() as connection:
        request = connection.execute(select(request_table)).mappings().one()
        response = connection.execute(select(response_table)).mappings().one()
    assert migration.status is MigrationStatus.APPLIED
    assert request["clarification_id"] == "clarification-1"
    assert request["step_id"] == "step-1"
    assert response["response_id"] == "response-1"
