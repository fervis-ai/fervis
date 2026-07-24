"""Persist semantic requested facts without legacy family projections."""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op


revision = "0004_semantic_requested_facts"
down_revision = "0003_clarification_successor_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("fervis_requested_fact") as batch:
        batch.add_column(
            sa.Column("requested_fact_fingerprint", sa.String(64), nullable=True)
        )
        batch.add_column(sa.Column("inputs_json", sa.JSON(), nullable=True))

    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE fervis_requested_fact "
            "SET requested_fact_fingerprint = '', inputs_json = :empty_inputs"
        ),
        {"empty_inputs": json.dumps({})},
    )

    with op.batch_alter_table("fervis_requested_fact") as batch:
        batch.drop_constraint(
            "fervis_requested_fact_answer_requests_json_json_valid_ck",
            type_="check",
        )
        batch.create_check_constraint(
            "fervis_requested_fact_inputs_json_json_valid_ck",
            "JSON_VALID(inputs_json)",
        )
        batch.alter_column(
            "requested_fact_fingerprint",
            existing_type=sa.String(64),
            nullable=False,
        )
        batch.alter_column(
            "inputs_json",
            existing_type=sa.JSON(),
            nullable=False,
        )
        batch.drop_column("answer_expression_family")
        batch.drop_column("answer_requests_json")


def downgrade() -> None:
    raise NotImplementedError("Fervis persistence migrations are forward-only")
