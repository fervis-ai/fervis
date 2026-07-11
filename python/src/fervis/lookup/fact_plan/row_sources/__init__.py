"""Planner row-source handles compiled from catalog and memory relations."""

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
    RowSourceValueType,
)
from fervis.lookup.relation_catalog.model import RowCardinality


def __getattr__(name: str):
    if name in {
        "api_row_source_id",
        "build_row_source_catalog",
        "memory_row_source_id",
        "row_source_ids_for_read_ids",
        "row_sources_for_read_id",
    }:
        from . import builder

        return getattr(builder, name)
    if name in {
        "read_evidence_ref",
        "read_field_evidence_ref",
        "required_input_evidence_ref",
        "row_source_description_evidence_ref",
        "row_source_evidence_ref",
        "row_source_evidence_refs",
        "row_source_field_evidence_ref",
        "row_source_param_evidence_ref",
    }:
        from . import evidence

        return getattr(evidence, name)
    if name == "executable_field_ids_for_row_path":
        from . import field_paths

        return getattr(field_paths, name)
    if name == "row_source_for_relation":
        from . import lookup

        return getattr(lookup, name)
    if name in {
        "memory_row_source_prompt_payload",
        "row_source_param_prompt_payload",
        "row_source_prompt_payload",
    }:
        from . import payload

        return getattr(payload, name)
    if name in {"api_read_source_groups", "read_row_source_counts"}:
        from . import source_groups

        return getattr(source_groups, name)
    raise AttributeError(name)

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
    "RowSourceValueType",
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
