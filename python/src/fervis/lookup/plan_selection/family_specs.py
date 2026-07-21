"""Plan-selection family contribution contract."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import Protocol

from fervis.lookup.plan_selection.support_options import PlanSelectionSupportOption
from fervis.lookup.source_binding.candidates.contracts import (
    CountBasisEvidence,
    FulfillmentSupportSet,
    RowPopulationEvidence,
)


class SupportSetGrouper(Protocol):
    def __call__(
        self,
        support_sets: tuple[FulfillmentSupportSet, ...],
        *,
        requirement_id: str,
        required_answer_output_ids: tuple[str, ...],
        source_candidate_id: str,
    ) -> tuple[tuple[FulfillmentSupportSet, ...], ...]: ...


_MEMBER_REQUIREMENT_SUPPORT_ROLES: dict[str, frozenset[str]] = {
    "metric": frozenset(("MEASURED_VALUE", "ROW_COUNT")),
    "group_identity": frozenset(("GROUP_KEY",)),
    "operation": frozenset(("MEASURED_VALUE", "ROW_COUNT", "GROUP_KEY")),
    "value_1": frozenset(("VALUE_SOURCE", "MEASURED_VALUE", "ROW_COUNT")),
    "value_2": frozenset(("VALUE_SOURCE", "MEASURED_VALUE", "ROW_COUNT")),
}


_MEMBER_REQUIREMENT_BINDING_ROLES: dict[str, frozenset[str]] = {
    "group_identity": frozenset(("GROUP_KEY",)),
    "value_1": frozenset(("VALUE_SOURCE", "MEASURED_VALUE", "ROW_COUNT")),
    "value_2": frozenset(("VALUE_SOURCE", "MEASURED_VALUE", "ROW_COUNT")),
}


class SourceMemberConstraint(StrEnum):
    ANY = "ANY"
    SAME_SOURCE_CANDIDATE = "SAME_SOURCE_CANDIDATE"
    DISTINCT_SOURCE_CANDIDATES = "DISTINCT_SOURCE_CANDIDATES"


@dataclass(frozen=True)
class PlanSelectionShapeSpec:
    plan_shape: str
    member_requirements: tuple[str, ...]
    single_source: bool = False
    distinct_members: bool = False
    row_population_grain_requirements: frozenset[str] = frozenset()
    intrinsic_source_requirements: frozenset[str] = frozenset()
    answer_fulfillment_requirements: frozenset[str] | None = None
    complete_answer_fulfillment_requirements: frozenset[str] | None = None
    support_set_grouper: SupportSetGrouper | None = None

    @property
    def member_constraint(self) -> SourceMemberConstraint:
        if self.distinct_members:
            return SourceMemberConstraint.DISTINCT_SOURCE_CANDIDATES
        if self.single_source:
            return SourceMemberConstraint.SAME_SOURCE_CANDIDATE
        return SourceMemberConstraint.ANY

    def validation_roles_for_requirement(
        self,
        requirement_id: str,
    ) -> frozenset[str] | None:
        return _MEMBER_REQUIREMENT_SUPPORT_ROLES.get(requirement_id)

    def binding_roles_for_requirement(
        self,
        requirement_id: str,
        *,
        support_options: tuple[PlanSelectionSupportOption, ...],
    ) -> frozenset[str] | None:
        if requirement_id == "population":
            return frozenset()
        if requirement_id == "metric":
            has_measured_value = any(
                "MEASURED_VALUE" in option.support_roles for option in support_options
            )
            if (
                self.plan_shape == "aggregate_by_group"
                and has_measured_value
            ):
                return frozenset(("MEASURED_VALUE",))
            return frozenset(("MEASURED_VALUE", "ROW_COUNT"))
        return _MEMBER_REQUIREMENT_BINDING_ROLES.get(requirement_id)

    def support_set_groups_for_requirement(
        self,
        support_sets: tuple[FulfillmentSupportSet, ...],
        *,
        requirement_id: str,
        required_answer_output_ids: tuple[str, ...],
        source_candidate_id: str,
    ) -> tuple[tuple[FulfillmentSupportSet, ...], ...]:
        if self.support_set_grouper is not None:
            return self.support_set_grouper(
                support_sets,
                requirement_id=requirement_id,
                required_answer_output_ids=required_answer_output_ids,
                source_candidate_id=source_candidate_id,
            )
        if requirement_id in self.row_population_grain_requirements:
            return _row_population_grain_groups(
                support_sets,
                required_answer_output_ids=required_answer_output_ids,
            )
        return (support_sets,) if support_sets else ()

    def allows_intrinsic_support_for_requirement(self, requirement_id: str) -> bool:
        return (
            requirement_id in self.intrinsic_source_requirements
            or requirement_id
            in {
                "value_1",
                "value_2",
            }
        )

    def requires_answer_fulfillment_for_requirement(
        self,
        requirement_id: str,
    ) -> bool:
        if self.answer_fulfillment_requirements is None:
            return requirement_id in self.member_requirements
        return requirement_id in self.answer_fulfillment_requirements

    def requires_complete_answer_fulfillment_for_requirement(
        self,
        requirement_id: str,
    ) -> bool:
        if not self.requires_answer_fulfillment_for_requirement(requirement_id):
            return False
        if self.complete_answer_fulfillment_requirements is None:
            return True
        return requirement_id in self.complete_answer_fulfillment_requirements

    def supports_member_combo(
        self,
        *,
        source_candidate_ids: tuple[str, ...],
    ) -> bool:
        if self.distinct_members and len(set(source_candidate_ids)) != len(
            source_candidate_ids
        ):
            return False
        if not self.single_source:
            return True
        return len(set(source_candidate_ids)) == 1


def _row_population_grain_groups(
    support_sets: tuple[FulfillmentSupportSet, ...],
    *,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[tuple[FulfillmentSupportSet, ...], ...]:
    """Validate row-population grain support without selecting operation parts."""

    support_sets_by_output: dict[str, list[FulfillmentSupportSet]] = {
        answer_output_id: [] for answer_output_id in required_answer_output_ids
    }
    has_support_by_output = {
        answer_output_id: False for answer_output_id in required_answer_output_ids
    }
    for support_set in support_sets:
        answer_output_id = support_set.answer_output_id
        if answer_output_id not in support_sets_by_output:
            continue
        support_sets_by_output[answer_output_id].append(support_set)
        if _support_set_has_row_count_basis(support_set):
            if _support_set_has_executable_row_count_basis(support_set):
                has_support_by_output[answer_output_id] = True
        else:
            has_support_by_output[answer_output_id] = True
    if any(not has_support for has_support in has_support_by_output.values()):
        return ()
    return (
        tuple(
            support_set
            for answer_output_id in required_answer_output_ids
            for support_set in support_sets_by_output[answer_output_id]
        ),
    )


def _support_set_has_row_count_basis(support_set: FulfillmentSupportSet) -> bool:
    return any(slot.row_count_basis_evidence for slot in support_set.fulfillment_slots)


def _support_set_has_executable_row_count_basis(
    support_set: FulfillmentSupportSet,
) -> bool:
    return any(
        any(
            _row_count_basis_evidence_is_executable(evidence)
            for evidence in slot.row_count_basis_evidence
        )
        for slot in support_set.fulfillment_slots
    )


def _row_count_basis_evidence_is_executable(
    evidence: CountBasisEvidence,
) -> bool:
    if isinstance(evidence, RowPopulationEvidence):
        return (
            evidence.row_cardinality == "many"
            and bool(evidence.row_source_id)
            and bool(evidence.row_path_id)
        )
    return bool(evidence.field_id)
