"""SQL authority predicates for Fervis-owned question state."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fervis.host_api.contracts.authority import (
    ReadAuthority,
    read_context_ref_matches,
)
from fervis.project.persistence.schema import metadata

from .transaction import sql_connection


def conversation_is_authorized(
    engine: Engine,
    *,
    conversation_id: str,
    authority: ReadAuthority,
) -> bool:
    conversation = metadata.tables["fervis_conversation"]
    with sql_connection(engine) as connection:
        row = connection.execute(
            sa.select(conversation.c.read_context_ref)
            .select_from(conversation)
            .where(
                conversation.c.conversation_id == conversation_id,
                conversation.c.tenant_id == authority.tenant_id,
            )
            .limit(1)
        ).first()
        if row is None:
            return True
        return read_context_ref_matches(
            row.read_context_ref or {},
            authority.read_context_ref,
        )


def question_is_authorized(
    engine: Engine,
    *,
    question_id: str,
    authority: ReadAuthority,
) -> bool:
    question = metadata.tables["fervis_question"]
    conversation = metadata.tables["fervis_conversation"]
    statement = (
        sa.select(conversation.c.read_context_ref)
        .select_from(
            question.join(
                conversation,
                question.c.conversation_id == conversation.c.conversation_id,
            )
        )
        .where(
            question.c.question_id == question_id,
            conversation.c.tenant_id == authority.tenant_id,
        )
    )
    with sql_connection(engine) as connection:
        row = connection.execute(statement).first()
        return bool(
            row is not None
            and read_context_ref_matches(
                row.read_context_ref or {},
                authority.read_context_ref,
            )
        )
