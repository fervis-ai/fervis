"""Row-source proof and evidence references."""

from __future__ import annotations

from .model import RowSourceCatalog
from fervis.lookup.fact_planning.required_inputs import required_input_id


def row_source_evidence_refs(row_sources: RowSourceCatalog) -> frozenset[str]:
    refs: set[str] = set()
    for source in row_sources.sources:
        refs.add(row_source_evidence_ref(source.id))
        if source.description:
            refs.add(row_source_description_evidence_ref(source.id))
        refs.update(
            row_source_field_evidence_ref(
                row_source_id=source.id,
                field_id=field.id,
            )
            for field in source.fields
        )
        refs.update(
            row_source_param_evidence_ref(
                row_source_id=source.id,
                param_id=param.id,
            )
            for param in source.params
        )
        refs.update(
            required_input_evidence_ref(
                required_input_id=required_input_id(
                    row_source_id=source.id,
                    param_id=param.id,
                )
            )
            for param in source.params
            if param.required and param.default is None
        )
        for fact in source.blocked_facts:
            refs.update(fact.proof_refs)
    return frozenset(refs)


def row_source_evidence_ref(row_source_id: str) -> str:
    return f"row_source:{row_source_id}"


def read_evidence_ref(read_id: str) -> str:
    return read_id


def read_field_evidence_ref(*, read_id: str, field_id: str) -> str:
    return f"{read_id}.{field_id}"


def row_source_description_evidence_ref(row_source_id: str) -> str:
    return f"row_source:{row_source_id}:description"


def row_source_field_evidence_ref(*, row_source_id: str, field_id: str) -> str:
    return f"row_source:{row_source_id}:field:{field_id}"


def row_source_param_evidence_ref(*, row_source_id: str, param_id: str) -> str:
    return f"row_source:{row_source_id}:param:{param_id}"


def required_input_evidence_ref(*, required_input_id: str) -> str:
    return f"required_input:{required_input_id}"
