"""Typed fact-local source-strategy selection contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fervis.lookup.conversation_resolution.overlay import (
    ConversationResolutionOverlay,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.fact_plan.fact_plan import PlanImpossible
from fervis.lookup.fact_planning.plan_shapes import ALL_PLAN_SHAPES
from fervis.lookup.question_contract import QuestionContract, RequestedFact
from fervis.lookup.turn_prompts.context import HostPromptContext


class SourceAlignment(StrEnum):
    DIRECT = "DIRECT"
    PARTIAL = "PARTIAL"
    NOT_ALIGNED = "NOT_ALIGNED"


@dataclass(frozen=True)
class PlanSelectionRequest:
    question: str
    question_contract: QuestionContract
    requested_facts: tuple[RequestedFact, ...]
    relation_catalog: RelationCatalog
    source_candidate_payload: dict[str, Any]
    conversation_context: dict[str, Any]
    conversation_resolution_overlay: ConversationResolutionOverlay | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)


@dataclass(frozen=True)
class SourceStrategyMember:
    source_candidate_id: str
    requirement_ids: tuple[str, ...] = ()
    fulfillment_support_set_ids: tuple[str, ...] = ()
    kind: str = ""
    read_id: str = ""
    value_id: str = ""
    memory_relation_id: str = ""
    source_relation_id: str = ""
    calendar_id: str = ""
    field_ids: tuple[str, ...] = ()
    operation_evidence: tuple[dict[str, Any], ...] = ()
    source_interface: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.source_candidate_id:
            raise ValueError("source strategy member requires source candidate")
        if any(not requirement_id.strip() for requirement_id in self.requirement_ids):
            raise ValueError("source strategy member requirement ids must be non-empty")
        if len(set(self.requirement_ids)) != len(self.requirement_ids):
            raise ValueError("source strategy member requirement ids must be unique")
        if any(
            not support_set_id.strip()
            for support_set_id in self.fulfillment_support_set_ids
        ):
            raise ValueError("source strategy member support set ids must be non-empty")
        if len(set(self.fulfillment_support_set_ids)) != len(
            self.fulfillment_support_set_ids
        ):
            raise ValueError("source strategy member support set ids must be unique")


@dataclass(frozen=True)
class SourceStrategy:
    source_strategy_id: str
    plan_shape: str
    required_answer_output_ids: tuple[str, ...]
    source_members: tuple[SourceStrategyMember, ...]

    def __post_init__(self) -> None:
        if not self.source_strategy_id:
            raise ValueError("source strategy requires id")
        if self.plan_shape not in ALL_PLAN_SHAPES:
            raise ValueError("source strategy references unknown plan shape")
        if not self.required_answer_output_ids:
            raise ValueError("source strategy requires answer outputs")
        if not self.source_members:
            raise ValueError("source strategy requires source members")


@dataclass(frozen=True)
class SelectedSourceStrategy:
    plan_selection_id: str
    requested_fact_id: str
    source_strategy_id: str
    plan_shape: str
    required_answer_output_ids: tuple[str, ...]
    source_members: tuple[SourceStrategyMember, ...]
    basis: str

    def __post_init__(self) -> None:
        if not self.plan_selection_id.strip():
            raise ValueError("plan selection requires id")
        if not self.requested_fact_id.strip():
            raise ValueError("plan selection requires requested fact")
        if not self.source_strategy_id.strip():
            raise ValueError("plan selection requires source strategy")
        if self.plan_shape not in ALL_PLAN_SHAPES:
            raise ValueError("plan selection references unknown plan shape")
        if not self.required_answer_output_ids:
            raise ValueError("plan selection requires answer outputs")
        if not self.source_members:
            raise ValueError("plan selection requires source members")
        if not self.basis.strip():
            raise ValueError("plan selection requires basis")


@dataclass(frozen=True)
class PlanSelectionSet:
    plan_selections: tuple[SelectedSourceStrategy, ...]

    def __post_init__(self) -> None:
        if not self.plan_selections:
            raise ValueError("plan selection requires at least one plan")

    def plan_selections_for(
        self,
        requested_fact_id: str,
    ) -> tuple[SelectedSourceStrategy, ...]:
        return tuple(
            plan
            for plan in self.plan_selections
            if plan.requested_fact_id == requested_fact_id
        )

    def plan_selection_for(self, requested_fact_id: str) -> SelectedSourceStrategy:
        matches = list(self.plan_selections_for(requested_fact_id))
        if len(matches) != 1:
            raise ValueError("expected exactly one plan selection for requested fact")
        return matches[0]


@dataclass(frozen=True)
class BoundRoleTarget:
    requirement_id: str
    source_candidate_id: str
    source_binding_ids: tuple[str, ...]
    fulfillment_support_set_ids: tuple[str, ...] = ()
    answer_output_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.requirement_id.strip():
            raise ValueError("bound role target requires requirement id")
        if not self.source_candidate_id.strip():
            raise ValueError("bound role target requires source candidate")
        if not self.source_binding_ids:
            raise ValueError("bound role target requires source bindings")
        if any(not source_id.strip() for source_id in self.source_binding_ids):
            raise ValueError("bound role target source bindings require ids")


@dataclass(frozen=True)
class BoundSourceStrategyMember:
    source_candidate_id: str
    role_targets: tuple[BoundRoleTarget, ...]
    field_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_candidate_id.strip():
            raise ValueError("bound source strategy member requires source candidate")
        if not self.role_targets:
            raise ValueError("bound source strategy member requires role targets")
        if any(
            target.source_candidate_id != self.source_candidate_id
            for target in self.role_targets
        ):
            raise ValueError("bound source strategy member role target mismatch")

    @property
    def source_binding_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                source_id
                for target in self.role_targets
                for source_id in target.source_binding_ids
            )
        )

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(target.requirement_id for target in self.role_targets)
        )


@dataclass(frozen=True)
class BoundSelectedSourceStrategy:
    plan_selection_id: str
    requested_fact_id: str
    source_strategy_id: str
    plan_shape: str
    required_answer_output_ids: tuple[str, ...]
    source_members: tuple[BoundSourceStrategyMember, ...]

    def __post_init__(self) -> None:
        if not self.plan_selection_id.strip():
            raise ValueError("bound plan selection requires plan selection id")
        if not self.requested_fact_id.strip():
            raise ValueError("bound plan selection requires requested fact")
        if not self.source_strategy_id.strip():
            raise ValueError("bound plan selection requires source strategy")
        if self.plan_shape not in ALL_PLAN_SHAPES:
            raise ValueError("bound plan selection references unknown plan shape")
        if not self.required_answer_output_ids:
            raise ValueError("bound plan selection requires answer outputs")
        if not self.source_members:
            raise ValueError("bound plan selection requires source members")

    @property
    def source_binding_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                source_id
                for member in self.source_members
                for source_id in member.source_binding_ids
            )
        )


@dataclass(frozen=True)
class BoundPlanSelectionSet:
    plan_selections: tuple[BoundSelectedSourceStrategy, ...]

    def __post_init__(self) -> None:
        if not self.plan_selections:
            raise ValueError("bound plan selection set requires at least one plan")

    def plan_shape_for(self, requested_fact_id: str) -> str:
        matches = self.plan_shapes_for(requested_fact_id)
        if len(matches) != 1:
            raise ValueError(
                "expected exactly one aligned plan shape for requested fact"
            )
        return matches[0]

    def plan_shapes_for(self, requested_fact_id: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                plan.plan_shape
                for plan in self.plan_selections
                if plan.requested_fact_id == requested_fact_id
            )
        )

    def plan_shapes_by_requested_fact_id(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, list[str]] = {}
        for plan in self.plan_selections:
            shapes = output.setdefault(plan.requested_fact_id, [])
            if plan.plan_shape not in shapes:
                shapes.append(plan.plan_shape)
        return {key: tuple(value) for key, value in output.items()}

    def source_binding_ids_for(self, requested_fact_id: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                source_binding_id
                for plan in self.plan_selections
                if plan.requested_fact_id == requested_fact_id
                for source_binding_id in plan.source_binding_ids
            )
        )

    def single_source_binding_id_for(self, requested_fact_id: str) -> str | None:
        source_binding_ids = self.source_binding_ids_for(requested_fact_id)
        if len(source_binding_ids) != 1:
            return None
        return source_binding_ids[0]

    def required_answer_output_ids_for(self, requested_fact_id: str) -> tuple[str, ...]:
        output = tuple(
            dict.fromkeys(
                answer_output_id
                for plan in self.plan_selections
                if plan.requested_fact_id == requested_fact_id
                for answer_output_id in plan.required_answer_output_ids
            )
        )
        if not output:
            raise ValueError("expected aligned answer outputs for requested fact")
        return output

    def answer_output_ids_by_requested_fact_id(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, list[str]] = {}
        for plan in self.plan_selections:
            ids = output.setdefault(plan.requested_fact_id, [])
            for answer_output_id in plan.required_answer_output_ids:
                if answer_output_id not in ids:
                    ids.append(answer_output_id)
        return {key: tuple(value) for key, value in output.items()}

    def source_binding_ids_by_requirement_by_requested_fact_id(
        self,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        output: dict[str, dict[str, tuple[str, ...]]] = {}
        for plan in self.plan_selections:
            roles = output.setdefault(plan.requested_fact_id, {})
            for member in plan.source_members:
                for target in member.role_targets:
                    current = list(roles.get(target.requirement_id, ()))
                    for source_binding_id in target.source_binding_ids:
                        if source_binding_id not in current:
                            current.append(source_binding_id)
                    roles[target.requirement_id] = tuple(current)
        return output

    def pattern_names(self) -> tuple[str, ...]:
        output: list[str] = []
        for plan in self.plan_selections:
            if plan.plan_shape not in output:
                output.append(plan.plan_shape)
        return tuple(output)


PlanSelectionOutcome = PlanSelectionSet | PlanImpossible


@dataclass(frozen=True)
class PlanSelectionResult:
    outcome: PlanSelectionOutcome
