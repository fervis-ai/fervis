"""One executable invocation-alternatives plan for verification and compilation."""

from __future__ import annotations

from fervis.lookup.source_binding.model import (
    SemanticSourceBindingRequest,
    SourceBindingPlan,
    InvocationTargetApplication,
)
from fervis.lookup.source_binding.occurrences import OccurrenceScope, ReadOccurrence
from fervis.lookup.answer_program.values import FactValue

from fervis.lookup.available_sources import source_choice_literal
from fervis.lookup.answer_program.values import ValueProjectionKind
from fervis.lookup.source_binding.param_values import fact_value_parameter_projection
from fervis.lookup.source_binding.param_binding_sets import (
    ParamBindingSetAlternatives,
    RelationInputOrigin,
    parameter_binding_sets,
    alternate_param_binding_sets,
    intersect_param_binding_sets,
    combine_param_binding_sets,
)


def invocation_value(
    request: SemanticSourceBindingRequest, value_ref: str
) -> FactValue:
    matches = [
        value.typed_value
        for value in request.canonical_values
        if value.canonical_value_id == value_ref
    ]
    matches += [
        value.typed_value
        for value in request.catalog_values
        if value.value_id == value_ref
    ]
    matches += [
        source_choice_literal(
            value, snapshot_ref=request.source_catalog.contract_snapshot.ref
        )
        for value in request.source_catalog.choice_values
        if value.value_ref == value_ref
    ]
    from fervis.lookup.source_binding.population_values import population_control_values
    control = population_control_values(request).get(value_ref)
    if control is not None:
        matches.append(control)
    if len(matches) != 1:
        raise ValueError(f"invocation value {value_ref} lacks one typed binding")
    return matches[0]


def invocation_binding_sets(
    *,
    request: SemanticSourceBindingRequest,
    plan: SourceBindingPlan,
    scope: OccurrenceScope,
    occurrence: ReadOccurrence,
    branch_id: str,
) -> ParamBindingSetAlternatives:
    source = request.source_catalog.source(occurrence.source_ref)
    groups_by_target: dict[str, list[tuple[str, ParamBindingSetAlternatives]]] = {}
    for application in scope.applications_for(
        request,
        plan,
        branch_id=branch_id,
        occurrence=occurrence,
    ):
        for target in application.target_applications:
            param = next(
                item for item in source.params if item.param_ref == target.target_ref
            )
            projected = invocation_target_value(request, source_ref=source.id, target=target)
            groups_by_target.setdefault(target.target_ref, []).append(
                (
                    application.owner_ref or application.application_ref,
                    parameter_binding_sets(
                        param_id=param.id,
                        value=projected,
                        parameter_type=param.type.value,
                        origin_kind=(
                            RelationInputOrigin.QUESTION_INPUT
                            if any(
                                item.canonical_value_id == target.value_ref
                                for item in request.canonical_values
                            )
                            else RelationInputOrigin.PLAN_CONTROL
                        ),
                        value_id=target.value_ref,
                        value_component=projection_component(target),
                        proof_refs=(application.application_ref, target.target_ref),
                    ),
                )
            )
    independent_groups: list[ParamBindingSetAlternatives] = []
    for target_ref, entries in groups_by_target.items():
        # Values selected by one requirement are alternatives. Independent
        # requirements constrain the same parameter by intersection.
        by_owner: dict[str, list[ParamBindingSetAlternatives]] = {}
        for owner_ref, group in entries:
            by_owner.setdefault(owner_ref, []).append(group)
        constraints = [
            alternate_param_binding_sets(groups) for groups in by_owner.values()
        ]
        intersection = intersect_param_binding_sets(constraints)
        if not intersection:
            raise ValueError(f"invocation constraints conflict on target {target_ref}")
        independent_groups.append(intersection)
    return combine_param_binding_sets(independent_groups)


def invocation_target_value(request, *, source_ref: str, target: InvocationTargetApplication):
    """Use the same typed projection for proof checks and executable bindings."""
    source = request.source_catalog.source(source_ref)
    param = next(item for item in source.params if item.param_ref == target.target_ref)
    return fact_value_parameter_projection(
        invocation_value(request, target.value_ref),
        projection=target.projection,
        component_id=target.component_ref,
        type_name=param.type.value,
        choices=tuple(str(item) for item in param.choices),
    )


def projection_component(target: InvocationTargetApplication) -> str:
    if target.projection is ValueProjectionKind.WHOLE_VALUE:
        return "value"
    if target.projection is ValueProjectionKind.TEMPORAL_START:
        return "start"
    if target.projection is ValueProjectionKind.TEMPORAL_END:
        return "end"
    if target.projection is ValueProjectionKind.IDENTITY_COMPONENT:
        if target.component_ref is None:
            raise ValueError("identity application lacks a key component")
        return f"key_component:{target.component_ref}"
    raise TypeError("unsupported invocation value projection")
