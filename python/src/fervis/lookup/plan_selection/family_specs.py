"""Plan-selection family contribution contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SupportSetGrouper(Protocol):
    def __call__(
        self,
        support_sets: tuple[dict[str, Any], ...],
        *,
        requirement_id: str,
        required_answer_output_ids: tuple[str, ...],
        source_candidate_id: str,
    ) -> tuple[tuple[dict[str, Any], ...], ...]: ...


_MEMBER_REQUIREMENT_SUPPORT_ROLES: dict[str, frozenset[str]] = {
    "metric": frozenset(("MEASURED_VALUE", "ROW_POPULATION")),
    "group_identity": frozenset(("GROUP_KEY",)),
    "operation": frozenset(("MEASURED_VALUE", "ROW_POPULATION", "GROUP_KEY")),
    "value_1": frozenset(("VALUE_SOURCE",)),
    "value_2": frozenset(("VALUE_SOURCE",)),
}


_MEMBER_REQUIREMENT_BINDING_ROLES: dict[str, frozenset[str]] = {
    "group_identity": frozenset(("GROUP_KEY",)),
    "value_1": frozenset(("VALUE_SOURCE",)),
    "value_2": frozenset(("VALUE_SOURCE",)),
}


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

    def validation_roles_for_requirement(
        self,
        requirement_id: str,
    ) -> frozenset[str] | None:
        return _MEMBER_REQUIREMENT_SUPPORT_ROLES.get(requirement_id)

    def binding_roles_for_requirement(
        self,
        requirement_id: str,
        *,
        support_options: tuple[dict[str, object], ...],
    ) -> frozenset[str] | None:
        if requirement_id == "population":
            return frozenset()
        if requirement_id == "metric":
            has_measured_value = any(
                "MEASURED_VALUE" in set(option.get("support_roles") or ())
                for option in support_options
            )
            if (
                self.plan_shape in {"aggregate_by_group", "ranked_aggregate"}
                and has_measured_value
            ):
                return frozenset(("MEASURED_VALUE",))
            return frozenset(("MEASURED_VALUE", "ROW_POPULATION"))
        return _MEMBER_REQUIREMENT_BINDING_ROLES.get(requirement_id)

    def support_set_groups_for_requirement(
        self,
        support_sets: tuple[dict[str, Any], ...],
        *,
        requirement_id: str,
        required_answer_output_ids: tuple[str, ...],
        source_candidate_id: str,
    ) -> tuple[tuple[dict[str, Any], ...], ...]:
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
    support_sets: tuple[dict[str, Any], ...],
    *,
    required_answer_output_ids: tuple[str, ...],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Validate row-population grain support without selecting operation parts."""

    support_sets_by_output: dict[str, list[dict[str, Any]]] = {
        answer_output_id: [] for answer_output_id in required_answer_output_ids
    }
    has_support_by_output = {
        answer_output_id: False for answer_output_id in required_answer_output_ids
    }
    for support_set in support_sets:
        answer_output_id = str(support_set.get("answer_output_id") or "")
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


def _support_set_has_row_count_basis(support_set: dict[str, Any]) -> bool:
    return any(
        isinstance(slot, dict) and bool(slot.get("row_count_basis_evidence"))
        for slot in support_set.get("fulfillment_slots") or ()
    )


def _support_set_has_executable_row_count_basis(
    support_set: dict[str, Any],
) -> bool:
    return any(
        isinstance(slot, dict)
        and any(
            _row_count_basis_evidence_is_executable(evidence)
            for evidence in slot.get("row_count_basis_evidence") or ()
            if isinstance(evidence, dict)
        )
        for slot in support_set.get("fulfillment_slots") or ()
    )


def _row_count_basis_evidence_is_executable(evidence: dict[str, Any]) -> bool:
    if str(evidence.get("type") or "") == "row_population":
        return (
            str(evidence.get("row_cardinality") or "") == "many"
            and bool(str(evidence.get("row_source_id") or ""))
            and bool(
                str(evidence.get("row_path_id") or "")
                or str(evidence.get("field_id") or "")
            )
        )
    field_id = str(evidence.get("field_id") or "")
    if not field_id:
        return False
    return True
