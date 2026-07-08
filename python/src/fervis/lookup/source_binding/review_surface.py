"""Typed source-binding review surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.normal_instance_roles import (
    NORMAL_INSTANCE_ROLE_PROFILES_KEY,
)
from fervis.lookup.source_binding.param_surface import (
    param_requires_finite_choice_review,
)
from fervis.lookup.source_binding.param_values import canonical_param_value


class SourceBindingReviewAxisKind(StrEnum):
    FINITE_CHOICE_PARAM = "finite_choice_param"
    ROW_PREDICATE = "row_predicate"


@dataclass(frozen=True)
class FiniteChoiceOmissionBehavior:
    kind: str = ""
    default_value: str = ""


@dataclass(frozen=True)
class NormalInstanceReviewProfile:
    test_id: str
    excluded_role_ids: tuple[str, ...]


@dataclass(frozen=True)
class FiniteChoiceReviewAxis:
    axis_id: str
    choices: tuple[str, ...]
    required: bool = False
    omission_behavior: FiniteChoiceOmissionBehavior = FiniteChoiceOmissionBehavior()
    normal_instance_profiles: tuple[NormalInstanceReviewProfile, ...] = ()
    owned_membership_test_ids: tuple[str, ...] = ()

    def can_be_omitted(self, *, include_values: tuple[str, ...]) -> bool:
        if self.required:
            return False
        include_set = set(include_values)
        if self.omission_behavior.kind == "all_values":
            return include_set == set(self.choices)
        if self.omission_behavior.kind == "uses_default":
            return bool(self.omission_behavior.default_value) and include_set == {
                self.omission_behavior.default_value
            }
        return False

    def normal_instance_profile(
        self,
        test_id: str,
    ) -> NormalInstanceReviewProfile | None:
        return next(
            (
                profile
                for profile in self.normal_instance_profiles
                if profile.test_id == test_id
            ),
            None,
        )


@dataclass(frozen=True)
class RowPredicateReviewAxis:
    axis_id: str
    field_id: str
    field_type: str
    operator: str
    allowed_values: tuple[str, ...]
    owned_membership_test_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceBindingPopulationRole:
    role_id: str


@dataclass(frozen=True)
class SourceBindingReviewSurface:
    finite_choice_params: dict[str, FiniteChoiceReviewAxis]
    row_predicates: dict[str, RowPredicateReviewAxis]
    population_roles: tuple[SourceBindingPopulationRole, ...]


def source_binding_review_surface(
    candidate: SourceCandidate,
) -> SourceBindingReviewSurface:
    return SourceBindingReviewSurface(
        finite_choice_params=_finite_choice_params(candidate),
        row_predicates=_row_predicates(candidate),
        population_roles=_population_roles(candidate),
    )


def _finite_choice_params(
    candidate: SourceCandidate,
) -> dict[str, FiniteChoiceReviewAxis]:
    output: dict[str, FiniteChoiceReviewAxis] = {}
    for param in candidate.params:
        if not isinstance(param, dict) or not param_requires_finite_choice_review(param):
            continue
        param_id = str(param.get("param_id") or "")
        choices = tuple(
            canonical_param_value(choice)
            for choice in param.get("choices") or ()
            if canonical_param_value(choice)
        )
        if not param_id or not choices:
            continue
        output[param_id] = FiniteChoiceReviewAxis(
            axis_id=param_id,
            choices=choices,
            required=param.get("required") is True,
            omission_behavior=_omission_behavior(param.get("population_contract")),
            normal_instance_profiles=_normal_instance_profiles(
                param.get(NORMAL_INSTANCE_ROLE_PROFILES_KEY),
            ),
            owned_membership_test_ids=_owned_membership_test_ids(
                param.get("population_contract"),
            ),
        )
    return output


def _row_predicates(
    candidate: SourceCandidate,
) -> dict[str, RowPredicateReviewAxis]:
    payload = candidate.payload or {}
    output: dict[str, RowPredicateReviewAxis] = {}
    for item in payload.get("row_predicates") or ():
        if not isinstance(item, dict):
            continue
        predicate_id = str(item.get("predicate_id") or "")
        field_id = str(item.get("field_id") or "")
        allowed_values = tuple(
            str(value) for value in item.get("allowed_values") or () if str(value)
        )
        if not predicate_id or not field_id or not allowed_values:
            continue
        output[predicate_id] = RowPredicateReviewAxis(
            axis_id=predicate_id,
            field_id=field_id,
            field_type=str(item.get("type") or ""),
            operator=str(item.get("operator") or "in"),
            allowed_values=allowed_values,
            owned_membership_test_ids=_owned_membership_test_ids(item),
        )
    return output


def _population_roles(
    candidate: SourceCandidate,
) -> tuple[SourceBindingPopulationRole, ...]:
    payload = candidate.payload or {}
    return tuple(
        SourceBindingPopulationRole(role_id=role_id)
        for item in payload.get("population_roles") or ()
        if isinstance(item, dict)
        for role_id in (str(item.get("role_id") or ""),)
        if role_id
    )


def _omission_behavior(raw: object) -> FiniteChoiceOmissionBehavior:
    if not isinstance(raw, dict):
        return FiniteChoiceOmissionBehavior()
    omission = raw.get("omission_behavior")
    if not isinstance(omission, dict):
        return FiniteChoiceOmissionBehavior()
    return FiniteChoiceOmissionBehavior(
        kind=str(omission.get("kind") or ""),
        default_value=canonical_param_value(omission.get("default_value")),
    )


def _normal_instance_profiles(raw: object) -> tuple[NormalInstanceReviewProfile, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(
        profile
        for item in raw
        if isinstance(item, dict)
        for profile in (_normal_instance_profile(item),)
        if profile is not None
    )


def _normal_instance_profile(
    item: dict[object, object],
) -> NormalInstanceReviewProfile | None:
    test_id = str(item.get("test_id") or "")
    if not test_id:
        return None
    return NormalInstanceReviewProfile(
        test_id=test_id,
        excluded_role_ids=_normal_instance_role_ids(item.get("excluded_state_roles")),
    )


def _normal_instance_role_ids(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(
        role_id
        for item in raw
        if isinstance(item, dict)
        for role_id in (str(item.get("role") or ""),)
        if role_id
    )


def _owned_membership_test_ids(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, dict):
        return ()
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in raw.get("owned_membership_test_ids") or ()
            if str(item).strip()
        )
    )
