"""Planner row-source handles compiled from catalog and memory relations."""

from .builder import (
    api_row_source_id,
    build_row_source_catalog,
    memory_row_source_id,
    row_source_ids_for_read_ids,
    row_sources_for_read_id,
)
from .evidence import (
    read_evidence_ref,
    read_field_evidence_ref,
    required_input_evidence_ref,
    row_source_description_evidence_ref,
    row_source_evidence_ref,
    row_source_evidence_refs,
    row_source_field_evidence_ref,
    row_source_param_evidence_ref,
)
from .field_paths import executable_field_ids_for_row_path
from .lookup import row_source_for_relation
from .model import (
    CALENDAR_DATE_FIELD_ID,
    CALENDAR_END_PARAM_ID,
    CALENDAR_END_PARAM_REF,
    CALENDAR_MAX_ROWS,
    CALENDAR_ROW_SOURCE_ID,
    CALENDAR_START_PARAM_ID,
    CALENDAR_START_PARAM_REF,
    RowSource,
    RowSourceBlockedFact,
    RowSourceCatalog,
    RowSourceField,
    RowSourceKind,
    RowSourceParam,
    RowSourceParamSemantics,
)
from fervis.lookup.relation_catalog.model import RowCardinality
from .payload import (
    memory_row_source_prompt_payload,
    row_source_param_prompt_payload,
    row_source_prompt_payload,
)
from .source_groups import api_read_source_groups, read_row_source_counts

__all__ = (
    "CALENDAR_DATE_FIELD_ID",
    "CALENDAR_END_PARAM_ID",
    "CALENDAR_END_PARAM_REF",
    "CALENDAR_MAX_ROWS",
    "CALENDAR_ROW_SOURCE_ID",
    "CALENDAR_START_PARAM_ID",
    "CALENDAR_START_PARAM_REF",
    "RowSource",
    "RowSourceBlockedFact",
    "RowSourceCatalog",
    "RowSourceField",
    "RowSourceKind",
    "RowSourceParam",
    "RowSourceParamSemantics",
    "RowCardinality",
    "api_row_source_id",
    "api_read_source_groups",
    "build_row_source_catalog",
    "executable_field_ids_for_row_path",
    "memory_row_source_id",
    "memory_row_source_prompt_payload",
    "read_evidence_ref",
    "read_row_source_counts",
    "read_field_evidence_ref",
    "required_input_evidence_ref",
    "row_source_description_evidence_ref",
    "row_source_evidence_ref",
    "row_source_evidence_refs",
    "row_source_field_evidence_ref",
    "row_source_for_relation",
    "row_source_ids_for_read_ids",
    "row_source_param_evidence_ref",
    "row_source_param_prompt_payload",
    "row_source_prompt_payload",
    "row_sources_for_read_id",
)
