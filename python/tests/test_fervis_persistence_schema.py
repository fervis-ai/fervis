from __future__ import annotations

from importlib import import_module

from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import sqlite

from fervis.project.persistence import schema

MODEL_REFS = (
    "fervis.lineage.models.Conversation",
    "fervis.lineage.models.Question",
    "fervis.lineage.models.QuestionRun",
    "fervis.lineage.models.AnswerProgram",
    "fervis.lineage.models.ProgramInvocation",
    "fervis.lineage.models.ProgramRevision",
    "fervis.lineage.models.RunStep",
    "fervis.lineage.models.ModelCall",
    "fervis.lineage.models.ModelCallUsage",
    "fervis.lineage.models.RunArtifact",
    "fervis.lineage.models.CatalogEndpoint",
    "fervis.lineage.models.SourceRead",
    "fervis.lineage.models.RunResult",
    "fervis.lineage.models.RuntimeErrorDetail",
    "fervis.lineage.models.RequestedFact",
    "fervis.lineage.models.FactResult",
    "fervis.lineage.models.MemoryArtifact",
    "fervis.lineage.models.ClarificationRequest",
    "fervis.lineage.models.ClarificationResponse",
    "fervis.lineage.models.Answer",
    "fervis.lineage.models.AnswerOutput",
    "fervis.lineage.models.AnswerPresentation",
    "fervis.lineage.models.ExecutionProofGraph",
    "fervis.run_work.queue.django.models.RunWorkItem",
)

TABLE_NAME_OVERRIDES = {
    "fervis_run_work_item": "fervis_run_work_item",
}


def test_persistence_metadata_uses_current_fervis_model_columns() -> None:
    for model_ref in MODEL_REFS:
        model = _model(model_ref)
        table_name = TABLE_NAME_OVERRIDES.get(
            model._meta.db_table,
            model._meta.db_table,
        )
        table = schema.metadata.tables[table_name]

        assert set(table.columns.keys()) == {
            field.column for field in model._meta.local_fields
        }


def test_persistence_metadata_uses_fervis_work_item_table() -> None:
    assert "fervis_run_work_item" in schema.metadata.tables


def test_fresh_schema_matches_program_and_run_contract() -> None:
    expected_columns = {
        "fervis_question_run": {
            "run_id",
            "question_id",
            "run_number",
            "kind",
            "trigger_kind",
            "base_run_id",
            "trigger_clarification_response_id",
            "adapter_ref",
            "runtime_version",
            "created_at",
        },
        "fervis_answer_program": {
            "program_id",
            "schema_revision",
            "canonical_json",
            "created_at",
        },
        "fervis_program_invocation": {
            "invocation_id",
            "run_id",
                "program_id",
                "bindings_json",
                "kind",
                "base_invocation_id",
                "patch_id",
            "binding_patch_json",
            "revision_id",
            "created_at",
        },
        "fervis_program_revision": {
            "revision_id",
            "base_program_id",
            "revised_program_id",
            "capability_id",
            "application_json",
            "created_at",
        },
        "fervis_run_work_item": {
            "id",
            "run_id",
            "conversation_id",
            "tenant_id",
            "user_id",
            "read_context_ref",
            "status",
            "spec_kind",
            "execution_spec",
            "idempotency_key",
            "attempt_count",
            "active_attempt",
            "max_attempts",
            "lease_owner",
            "lease_expires_at",
            "next_attempt_at",
            "last_error",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
        },
    }

    assert {
        table_name: set(schema.metadata.tables[table_name].columns.keys())
        for table_name in expected_columns
    } == expected_columns


def test_initial_revision_fingerprint_matches_current_metadata() -> None:
    assert (
        schema.metadata_fingerprint()
        == "0f709421843a2bd09b9e40cbdce435ed781a0ec637fdeea01e91833786e462a5"
    )
    schema.assert_head_schema_fingerprint_is_current()


def test_persistence_metadata_preserves_runtime_constraint_names() -> None:
    work_item = schema.metadata.tables["fervis_run_work_item"]
    artifact = schema.metadata.tables["fervis_run_artifact"]

    assert "fervis_work_idempotency_unique" in {
        index.name for index in work_item.indexes if index.unique
    }
    assert "fervis_work_active_conversation_unique" in {
        index.name for index in work_item.indexes if index.unique
    }
    assert "fervis_artifact_one_body" in {
        constraint.name for constraint in artifact.constraints
    }


def test_persistence_metadata_preserves_indexed_runtime_fields() -> None:
    work_item = schema.metadata.tables["fervis_run_work_item"]
    index_columns = {
        tuple(column.name for column in index.columns)
        for index in work_item.indexes
    }

    assert ("status", "next_attempt_at", "created_at") in index_columns
    assert ("conversation_id",) in index_columns
    assert ("lease_expires_at",) in index_columns


def test_sqlite_auto_primary_keys_compile_as_integer() -> None:
    work_item = schema.metadata.tables["fervis_run_work_item"]
    ddl = str(CreateTable(work_item).compile(dialect=sqlite.dialect()))

    assert "id INTEGER NOT NULL" in ddl


def test_persistence_metadata_enforces_json_and_nonnegative_checks() -> None:
    run_step = schema.metadata.tables["fervis_run_step"]
    checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in run_step.constraints
        if constraint.__class__.__name__ == "CheckConstraint"
    }

    assert checks["fervis_run_step_sequence_nonnegative_ck"] == "sequence >= 0"
    assert (
        checks["fervis_run_step_input_summary_json_json_valid_ck"]
        == "JSON_VALID(input_summary_json)"
    )


def _model(model_ref: str):
    module_name, class_name = model_ref.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)
