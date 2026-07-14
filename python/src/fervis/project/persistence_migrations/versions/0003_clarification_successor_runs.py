"""Persist clarification-triggered successor runs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0003_clarification_successor_runs"
down_revision = "0002_same_run_clarification_and_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fervis_question_run",
        sa.Column(
            "trigger_clarification_response_id",
            sa.String(128),
            nullable=True,
        ),
    )
    op.get_bind().execute(
        sa.text(
            "UPDATE fervis_question_run "
            "SET trigger_clarification_response_id = '' "
            "WHERE trigger_clarification_response_id IS NULL"
        )
    )
    with op.batch_alter_table("fervis_question_run") as batch:
        batch.alter_column(
            "trigger_clarification_response_id",
            existing_type=sa.String(128),
            nullable=False,
        )


def downgrade() -> None:
    raise NotImplementedError("Fervis persistence migrations are forward-only")
