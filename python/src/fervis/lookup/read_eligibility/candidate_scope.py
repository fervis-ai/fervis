"""Canonical read-scoped surfaces for read eligibility."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.read_eligibility.candidate_identity import (
    read_candidate_signature,
)
from fervis.lookup.read_eligibility.model import ReadEligibilityRequest


@dataclass(frozen=True)
class ReadEligibilityCandidateScope:
    requested_fact_id: str
    source_candidate_id: str
    source_candidate_signature: str
    read_id: str
    field_refs_by_path: dict[str, str]
    field_refs_by_evidence_token: dict[str, str]
    row_path_ids_by_evidence_token: dict[str, str]
    row_path_ids: tuple[str, ...]


def read_eligibility_candidate_scopes_from_cards(
    request: ReadEligibilityRequest,
    *,
    card_payload: dict[str, object],
) -> tuple[ReadEligibilityCandidateScope, ...]:
    reads_by_id = {
        read.id: read for read in request.catalog_selection.relation_catalog.reads
    }
    output: list[ReadEligibilityCandidateScope] = []
    for group in _array(card_payload.get("requested_fact_read_candidates")):
        if not isinstance(group, dict):
            continue
        requested_fact_id = str(group.get("requested_fact_id") or "")
        for card in _array(group.get("read_candidates")):
            if not isinstance(card, dict):
                continue
            read_id = str(card.get("read_id") or "")
            read = reads_by_id[read_id]
            field_refs_by_path_from_catalog = {
                field.path: field.ref for field in read.fields if field.path
            }
            field_tokens = tuple(_response_field_tokens(card))
            field_refs_by_path = {
                field_path: field_refs_by_path_from_catalog[field_path]
                for field_path in (
                    _field_path_from_token(token) for token in field_tokens
                )
                if field_path in field_refs_by_path_from_catalog
            }
            field_refs_by_evidence_token = {
                token: field_refs_by_path_from_catalog[field_path]
                for token in field_tokens
                for field_path in (_field_path_from_token(token),)
                if field_path in field_refs_by_path_from_catalog
            }
            row_tokens = tuple(_response_row_tokens(card))
            row_path_ids_by_evidence_token = {
                token: row_path_id
                for token in row_tokens
                for row_path_id in (_row_path_id_from_token(token),)
                if row_path_id
            }
            row_path_ids = tuple(
                row_path_id
                for row_path_id in (
                    _row_path_id_from_token(token) for token in row_tokens
                )
                if row_path_id
            )
            output.append(
                ReadEligibilityCandidateScope(
                    requested_fact_id=requested_fact_id,
                    source_candidate_id=str(card.get("source_candidate_id") or ""),
                    source_candidate_signature=read_candidate_signature(
                        card,
                        requested_fact_id=requested_fact_id,
                    ),
                    read_id=read_id,
                    field_refs_by_path=field_refs_by_path,
                    field_refs_by_evidence_token=field_refs_by_evidence_token,
                    row_path_ids_by_evidence_token=row_path_ids_by_evidence_token,
                    row_path_ids=row_path_ids,
                )
            )
    return tuple(output)


def _response_field_tokens(card: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        token
        for row in _array(card.get("response_rows"))
        if isinstance(row, dict)
        for field in _array(row.get("fields"))
        if isinstance(field, dict)
        for token in (str(field.get("evidence_token") or ""),)
        if token
    )


def _response_row_tokens(card: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        token
        for row in _array(card.get("response_rows"))
        if isinstance(row, dict)
        for token in (str(row.get("evidence_token") or ""),)
        if token
    )


def _array(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _field_path_from_token(token: str) -> str:
    marker = ".field."
    index = token.find(marker)
    if index < 0:
        return ""
    return token[index + len(marker) :]


def _row_path_id_from_token(token: str) -> str:
    marker = ".row."
    index = token.find(marker)
    if index < 0:
        return ""
    return token[index + len(marker) :]
