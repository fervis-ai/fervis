"""The sole ordinary-business-instance classification vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.types.enums import StrEnum


class NormalInstanceProfileId(StrEnum):
    ORDINARY_BUSINESS_INSTANCE_V1 = "ORDINARY_BUSINESS_INSTANCE_V1"


class NormalInstanceExcludedStateRole(StrEnum):
    NOT_REALIZED = "NOT_REALIZED"
    CANCELED_OR_VOIDED = "CANCELED_OR_VOIDED"
    FAILED_OR_REJECTED_BEFORE_EFFECT = "FAILED_OR_REJECTED_BEFORE_EFFECT"
    REVERSED_OR_CORRECTION_ARTIFACT = "REVERSED_OR_CORRECTION_ARTIFACT"
    TEST_PLACEHOLDER_OR_DEMO = "TEST_PLACEHOLDER_OR_DEMO"
    SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT = (
        "SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT"
    )


@dataclass(frozen=True)
class NormalInstanceExcludedStateRoleDefinition:
    role: NormalInstanceExcludedStateRole
    definition: str

    def to_answer_request_dict(self) -> dict[str, object]:
        return {"role": self.role.value, "definition": self.definition}


@dataclass(frozen=True)
class SubjectChoiceMembership:
    included: bool
    explicit_user_override_applies: bool


def derive_subject_choice_membership(
    *,
    choice_included: bool,
    matched_excluded_role: str | None,
    explicit_user_override_applies: bool,
) -> SubjectChoiceMembership:
    return SubjectChoiceMembership(
        included=(
            explicit_user_override_applies
            if matched_excluded_role is not None
            else choice_included
        ),
        explicit_user_override_applies=explicit_user_override_applies,
    )


NORMAL_INSTANCE_EXCLUDED_STATE_ROLES: tuple[
    NormalInstanceExcludedStateRoleDefinition, ...
] = (
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.NOT_REALIZED,
        definition=(
            "A tentative, draft, planned-only, or provisional representation "
            "that has not become an effective instance of the requested subject."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.CANCELED_OR_VOIDED,
        definition=(
            "The entity was canceled, voided, aborted, or otherwise explicitly "
            "made not count as an ordinary business instance."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.FAILED_OR_REJECTED_BEFORE_EFFECT,
        definition=(
            "The entity failed, was rejected, was declined, or did not become "
            "effective in normal business operations."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.REVERSED_OR_CORRECTION_ARTIFACT,
        definition=(
            "The entity is a reversal, refund, correction, adjustment, or "
            "counter-entry rather than the ordinary business instance itself."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=NormalInstanceExcludedStateRole.TEST_PLACEHOLDER_OR_DEMO,
        definition=(
            "The entity is test, demo, sample, placeholder, sandbox, or seed "
            "data rather than an ordinary business instance."
        ),
    ),
    NormalInstanceExcludedStateRoleDefinition(
        role=(
            NormalInstanceExcludedStateRole.SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT
        ),
        definition=(
            "The entity is superseded, deleted, archived as non-current, or "
            "kept only as a historical artifact rather than a current ordinary "
            "business instance."
        ),
    ),
)


__all__ = [
    "NORMAL_INSTANCE_EXCLUDED_STATE_ROLES",
    "NormalInstanceExcludedStateRole",
    "NormalInstanceExcludedStateRoleDefinition",
    "NormalInstanceProfileId",
    "SubjectChoiceMembership",
    "derive_subject_choice_membership",
]
