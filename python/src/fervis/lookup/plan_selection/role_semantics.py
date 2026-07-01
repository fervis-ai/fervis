"""Model-facing support-role semantics for plan selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceRoleSemantics:
    role: str
    answers_as: str
    output_kind: str
    usage: str

    def payload(self, *, field_ids: tuple[str, ...]) -> dict[str, object]:
        return {
            "role": self.role,
            "answers_as": self.answers_as,
            "output_kind": self.output_kind,
            "usage": self.usage,
            "field_ids": list(field_ids),
        }


SUPPORT_KEY_ROLE_SEMANTICS: dict[str, EvidenceRoleSemantics] = {
    "row_count_basis_evidence": EvidenceRoleSemantics(
        role="row_count_basis",
        answers_as="record_count",
        output_kind="count",
        usage=(
            "Produces a count of records or business instances from the selected "
            "row population."
        ),
    ),
    "metric_measure_evidence": EvidenceRoleSemantics(
        role="metric_measure",
        answers_as="measured_value",
        output_kind="numeric_measure",
        usage=(
            "Produces a numeric result from field values that measure the "
            "requested quantity, such as a sum, total, average, minimum, "
            "maximum, or ranking metric."
        ),
    ),
    "group_key_evidence": EvidenceRoleSemantics(
        role="group_key",
        answers_as="bucket_or_entity_identity",
        output_kind="identity",
        usage=(
            "Identifies the entity, category, or bucket used to partition rows "
            "before computing or selecting the answer."
        ),
    ),
    "scope_evidence": EvidenceRoleSemantics(
        role="scope",
        answers_as="scope_constraint",
        output_kind="constraint",
        usage=(
            "Limits the selected records to the scope requested by the "
            "question, such as time, status, type, identity, category, or "
            "location."
        ),
    ),
}

SUPPORT_KEY_ROLE_ORDER = (
    "row_count_basis_evidence",
    "metric_measure_evidence",
    "group_key_evidence",
    "scope_evidence",
)


def evidence_role_payload(
    *,
    support_key: str,
    field_ids: tuple[str, ...],
) -> dict[str, object] | None:
    semantics = SUPPORT_KEY_ROLE_SEMANTICS.get(support_key)
    if semantics is None or not field_ids:
        return None
    return semantics.payload(field_ids=field_ids)
