"""Assertions for the semantic Source Binding boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_semantic_source_binding_boundary import build_request
from fervis.lookup.source_binding.model import (
    AssociationRealizationKind,
    FactRealizationKind,
    SourceMechanicKind,
)
from fervis.lookup.source_binding import (
    VerifiedSourceStrategy,
    compile_source_binding_plan,
    verify_source_strategy,
)
from fervis.lookup.answer_program.values import ValueProjectionKind


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    request = build_request(context)
    try:
        plan = compile_source_binding_plan(arguments, request=request)
    except ValueError as exc:
        return [f"semantic source binding parser rejected output: {exc}"]
    if context.get("kind") == "explicit_choice_override":
        expected_choice = str(context["requested_choice"])
        expected_override = bool(context["expected_override"])
        selected_values = tuple(
            request.source_catalog.choice_value(application.value_ref).value
            for application in plan.invocation_applications
            if application.owner_ref
            in {
                requirement.requirement_ref
                for requirement in request.index.boolean_requirements
            }
            and application.value_ref.startswith("source_choice:")
        )
        if selected_values != (expected_choice,):
            return [
                f"Boolean requirement selected {selected_values!r}, "
                f"expected {(expected_choice,)!r}"
            ]
        reviews = {
            request.source_catalog.choice_value(review.choice_ref).value: review
            for realization in plan.subject_binding.branch_realizations
            for surface in realization.surface_reviews
            for review in surface.choice_reviews
        }
        requested_review = reviews.get(expected_choice)
        if requested_review is None or not requested_review.included:
            return [f"{expected_choice} is not included"]
        actual_override = requested_review.explicit_user_override_applies
        if actual_override is not expected_override:
            return [f"{expected_choice} explicit override is {actual_override!r}"]
        if expected_override and (
            requested_review.matched_excluded_role != "CANCELED_OR_VOIDED"
        ):
            return [
                f"{expected_choice} excluded role is "
                f"{requested_review.matched_excluded_role!r}"
            ]
        if not isinstance(
            verify_source_strategy(plan, request=request),
            VerifiedSourceStrategy,
        ):
            return ["explicit choice binding did not verify"]
        return []
    if context.get("kind") in {"time_binding", "required_choice_binding"}:
        mechanics = tuple(
            mechanic
            for values in plan.boolean_bindings.values()
            for realization in values
            for mechanic in realization.mechanics
        )
        application_refs = {
            application_ref
            for mechanic in mechanics
            if mechanic.kind is SourceMechanicKind.INVOCATION_PREDICATE
            for application_ref in mechanic.application_refs
        }
        applications = tuple(
            target
            for application in plan.invocation_applications
            if application.application_ref in application_refs
            for target in application.target_applications
        )
        actual = {
            (application.target_ref, application.projection)
            for application in applications
        }
        expected = {
            ("source_events.start_date", ValueProjectionKind.TEMPORAL_START),
            ("source_events.end_date", ValueProjectionKind.TEMPORAL_END),
        }
        if actual != expected:
            return [f"time applications are {sorted(actual)!r}"]
        if surface_suffix := context.get("unrestricted_choice_surface"):
            surfaces = {
                surface_ref: surface
                for branch in arguments["subject_binding"]["branch_realizations"]
                for surface_ref, surface in branch[
                    "finite_choice_reviews"
                ].items()
            }
            surface = next(
                (
                    item
                    for surface_ref, item in surfaces.items()
                    if surface_ref.endswith(f".{surface_suffix}")
                ),
                None,
            )
            if surface is None:
                return [f"{surface_suffix} surface review is missing"]
            for value, review in surface["choice_reviews"].items():
                if (
                    review["matched_excluded_role"] != "NONE"
                    or review["choice_inclusion"] != "INCLUDE"
                ):
                    return [f"{surface_suffix}={value} is restricted"]
            applied_surfaces = {
                application["surface_ref"]
                for owners in arguments["finite_choice_applications"].values()
                for application in owners.values()
            }
            if any(
                surface_ref.endswith(f".{surface_suffix}")
                for surface_ref in applied_surfaces
            ):
                return [f"{surface_suffix} claims a requirement application"]
        if context.get("kind") == "required_choice_binding":
            selected_values = {
                (
                    target.target_ref,
                    str(
                        request.source_catalog.choice_value(application.value_ref).value
                    ),
                )
                for application in plan.invocation_applications
                if application.value_ref.startswith("source_choice:")
                for target in application.target_applications
            }
            if {target for target, _value in selected_values} != {
                "source_events.group_by",
                "source_events.granularity",
            }:
                return [f"required choices are {sorted(selected_values)!r}"]
        if not isinstance(
            verify_source_strategy(plan, request=request), VerifiedSourceStrategy
        ):
            return ["time binding did not verify"]
        return []
    if context.get("kind") == "scalar_binding":
        application_refs = {
            application_ref
            for values in plan.boolean_bindings.values()
            for realization in values
            for mechanic in realization.mechanics
            if mechanic.kind is SourceMechanicKind.INVOCATION_PREDICATE
            for application_ref in mechanic.application_refs
        }
        applications = tuple(
            target
            for application in plan.invocation_applications
            if application.application_ref in application_refs
            for target in application.target_applications
        )
        actual = {
            (application.target_ref, application.projection)
            for application in applications
        }
        expected = {
            (
                "source_measurements.minimum_amount",
                ValueProjectionKind.WHOLE_VALUE,
            )
        }
        if actual != expected:
            return [f"scalar applications are {sorted(actual)!r}"]
        if not isinstance(
            verify_source_strategy(plan, request=request), VerifiedSourceStrategy
        ):
            return ["scalar binding did not verify"]
        return []
    if context.get("kind") == "finite_choice_binding":
        choice_applications = tuple(
            application
            for application in plan.invocation_applications
            if application.value_ref.startswith("source_choice:")
        )
        if any(
            not application.value_ref.endswith(":priority:0")
            or {target.target_ref for target in application.target_applications}
            != {"source_observations.priority"}
            for application in choice_applications
        ):
            return ["finite-choice binding selected the wrong choice or target"]
        if not isinstance(
            verify_source_strategy(plan, request=request), VerifiedSourceStrategy
        ):
            return ["finite-choice binding did not verify"]
        return []
    if context.get("kind") == "association_binding":
        realizations = tuple(
            item for values in plan.association_bindings.values() for item in values
        )
        if len(realizations) != 1:
            return ["association binding does not have one realization"]
        [realization] = realizations
        expected_kind = AssociationRealizationKind(
            context.get("expected_association_kind", "DECLARED_RELATION")
        )
        if realization.kind is not expected_kind:
            return [
                f"association kind is {realization.kind.value}, expected {expected_kind.value}"
            ]
        if expected_kind is AssociationRealizationKind.DECLARED_RELATION:
            if realization.relation_evidence_ref is None or set(
                realization.source_refs
            ) != {"source_projects", "source_assignments"}:
                return ["association did not use the declared cross-source relation"]
        elif realization.source_refs != ("source_assignments",):
            return ["association did not use the co-resident source"]
        if not isinstance(
            verify_source_strategy(plan, request=request), VerifiedSourceStrategy
        ):
            return ["association binding did not verify"]
        return []
    if context.get("kind") == "identity_set_binding":
        applications = tuple(
            target
            for application in plan.invocation_applications
            for target in application.target_applications
        )
        if {
            (item.target_ref, item.projection, item.component_ref)
            for item in applications
        } != {
            (
                "source_observations_by_member.member_id",
                ValueProjectionKind.IDENTITY_COMPONENT,
                "member_id",
            )
        }:
            return ["identity collection was not applied to the member parameter"]
        if not isinstance(
            verify_source_strategy(plan, request=request), VerifiedSourceStrategy
        ):
            return ["identity collection binding did not verify"]
        return []
    fact_realizations = tuple(
        item for values in plan.fact_bindings.values() for item in values
    )
    if not any(
        item.kind is FactRealizationKind.ENTITY_KEY for item in fact_realizations
    ):
        return ["member identity was not bound to the declared entity key"]
    mechanics = tuple(
        mechanic
        for values in plan.boolean_bindings.values()
        for realization in values
        for mechanic in realization.mechanics
    )
    if not any(
        item.kind is SourceMechanicKind.RETURNED_ROW_PREDICATE for item in mechanics
    ):
        return ["identity qualification lacks a returned-row predicate"]
    expected_subject_values = tuple(context.get("expected_subject_values", ()))
    if expected_subject_values:
        reviews = tuple(
            review
            for realization in plan.subject_binding.branch_realizations
            for review in realization.surface_reviews
        )
        if len(reviews) != 1:
            return ["ordinary subject requires exactly one surface review"]
        [review] = reviews
        actual_values = tuple(
            str(request.source_catalog.choice_value(choice_ref).value)
            for choice_ref in review.included_choice_refs
        )
        if set(actual_values) != set(expected_subject_values):
            return [
                f"ordinary-instance values are {sorted(actual_values)}, "
                f"expected {sorted(expected_subject_values)}"
            ]
    return []


__all__ = ["validate"]
