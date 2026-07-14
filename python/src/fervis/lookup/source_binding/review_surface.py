"""Typed source-binding review surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum

from fervis.lookup.source_binding.candidates import SourceCandidate


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
        if not param.finite_choice_review:
            continue
        param_id = param.id
        choices = param.choices
        if not param_id or not choices:
            continue
        output[param_id] = FiniteChoiceReviewAxis(
            axis_id=param_id,
            choices=choices,
            required=param.required,
            omission_behavior=FiniteChoiceOmissionBehavior(
                kind=param.omission_kind,
                default_value=param.omission_default_value,
            ),
            normal_instance_profiles=tuple(
                NormalInstanceReviewProfile(
                    test_id=profile.test_id,
                    excluded_role_ids=profile.excluded_role_ids,
                )
                for profile in param.normal_instance_profiles
            ),
            owned_membership_test_ids=param.owned_membership_test_ids,
        )
    return output


def _row_predicates(
    candidate: SourceCandidate,
) -> dict[str, RowPredicateReviewAxis]:
    output: dict[str, RowPredicateReviewAxis] = {}
    for item in candidate.row_predicates:
        predicate_id = item.id
        field_id = item.field_id
        allowed_values = item.allowed_values
        if not predicate_id or not field_id or not allowed_values:
            continue
        output[predicate_id] = RowPredicateReviewAxis(
            axis_id=predicate_id,
            field_id=field_id,
            field_type=item.field_type,
            operator=item.operator,
            allowed_values=allowed_values,
            owned_membership_test_ids=item.owned_membership_test_ids,
        )
    return output


def _population_roles(
    candidate: SourceCandidate,
) -> tuple[SourceBindingPopulationRole, ...]:
    return tuple(
        SourceBindingPopulationRole(role_id=role_id)
        for role_id in candidate.population_role_ids
        if role_id
    )
