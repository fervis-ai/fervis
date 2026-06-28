"""Fervis persistence schema after read-context run work."""

from __future__ import annotations

import sqlalchemy as sa

from .v0001 import metadata as v0001_metadata

metadata = sa.MetaData()

for table in v0001_metadata.tables.values():
    table.to_metadata(metadata)

fervis_conversation = metadata.tables["fervis_conversation"]
fervis_conversation.append_column(
    sa.Column("read_context_ref", sa.JSON(), nullable=False)
)

fervis_run_work_item = metadata.tables["fervis_run_work_item"]
fervis_run_work_item.append_column(
    sa.Column("read_context_ref", sa.JSON(), nullable=False)
)
