"""Row-source proof and evidence references."""

from __future__ import annotations

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
