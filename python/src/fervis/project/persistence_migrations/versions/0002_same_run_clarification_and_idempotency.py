"""Advance run waiting and idempotency to their current first-class contracts."""

from __future__ import annotations

import hashlib
import json
import sqlalchemy as sa
from alembic import op


revision = "0002_same_run_clarification_and_idempotency"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            UPDATE fervis_clarification_request
            SET step_id = (
                SELECT produced_by_step_id
                FROM fervis_fact_result
                WHERE fervis_fact_result.fact_result_id =
                    fervis_clarification_request.fact_result_id
            )
            WHERE step_id IS NULL AND fact_result_id IS NOT NULL
            """
        )
    )
    unresolved_clarifications = connection.scalar(
        sa.text(
            "SELECT COUNT(*) FROM fervis_clarification_request "
            "WHERE step_id IS NULL"
        )
    )
    if unresolved_clarifications:
        raise RuntimeError(
            "cannot migrate clarification without owning step lineage"
        )

    op.drop_column("fervis_question_run", "trigger_clarification_response_id")

    op.drop_index(
        "fervis_clarification_request_fact_result_id_idx",
        table_name="fervis_clarification_request",
    )
    with op.batch_alter_table("fervis_clarification_request") as batch:
        batch.drop_column("fact_result_id")
        batch.alter_column("step_id", existing_type=sa.String(128), nullable=False)

    with op.batch_alter_table("fervis_run_work_item") as batch:
        batch.add_column(
            sa.Column(
                "idempotency_authority_ref",
                sa.String(96),
                nullable=False,
                server_default="",
            )
        )
        batch.add_column(
            sa.Column(
                "idempotency_scope",
                sa.String(160),
                nullable=False,
                server_default="",
            )
        )

    work_items = sa.table(
        "fervis_run_work_item",
        sa.column("id"),
        sa.column("tenant_id"),
        sa.column("user_id"),
        sa.column("conversation_id"),
        sa.column("read_context_ref"),
        sa.column("idempotency_authority_ref"),
        sa.column("idempotency_scope"),
    )
    for row in connection.execute(sa.select(work_items)).mappings():
        authority_payload = json.dumps(
            {
                "tenant_id": row["tenant_id"],
                "principal_id": row["user_id"],
                "read_context_ref": row["read_context_ref"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        authority_ref = (
            "idempotency-authority:sha256:"
            + hashlib.sha256(authority_payload.encode()).hexdigest()
        )
        connection.execute(
            work_items.update()
            .where(work_items.c.id == row["id"])
            .values(
                idempotency_authority_ref=authority_ref,
                idempotency_scope=f"conversation:{row['conversation_id']}",
            )
        )

    with op.batch_alter_table("fervis_run_work_item") as batch:
        batch.drop_index("fervis_work_idempotency_unique")
        batch.drop_index("fervis_work_active_conversation_unique")
        batch.create_index(
            "fervis_work_idempotency_unique",
            ["idempotency_authority_ref", "idempotency_scope", "idempotency_key"],
            unique=True,
            sqlite_where=sa.text("idempotency_key IS NOT NULL"),
        )
        batch.create_index(
            "fervis_work_active_conversation_unique",
            ["tenant_id", "conversation_id"],
            unique=True,
            sqlite_where=sa.text(
                "status IN ('QUEUED', 'RUNNING', 'WAITING_FOR_CLARIFICATION')"
            ),
        )


def downgrade() -> None:
    raise NotImplementedError("Fervis persistence migrations are forward-only")
