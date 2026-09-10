"""Prove declared filter coverage for every allocated logical REST invocation."""

from fervis.host_api.contracts import ParameterSemantics
from fervis.lookup.source_binding.occurrences import occurrence_scope
from fervis.lookup.source_binding.population_values import (
    covers_population,
    argument_text,
)


def uncovered_parameter_scopes(plan, *, request) -> tuple[str, ...]:
    boolean_owners = {
        item.requirement_ref for item in request.index.boolean_requirements
    }
    failed = []
    for branch in request.strategy.branches:
        scope = occurrence_scope(request, plan, branch.branch_id)
        for occurrence in scope.occurrences:
            applications = scope.applications_for(
                request, plan, branch_id=branch.branch_id, occurrence=occurrence
            )
            source = request.source_catalog.source(occurrence.source_ref)
            for declared_param in source.params:
                from dataclasses import replace
                param = replace(declared_param, population=request.parameter_population(source.id, declared_param.param_ref))
                if param.semantics is not ParameterSemantics.OPAQUE_QUERY_PARAM:
                    continue
                if not param.required and param.default is None and param.default_is_known:
                    continue
                selected = tuple(
                    application
                    for application in applications
                    if any(
                        target.target_ref == param.param_ref
                        for target in application.target_applications
                    )
                )
                if not selected and request.access_supplies(source.id,param.param_ref):
                    continue
                # An exact invocation predicate may narrow its own population.
                if any(
                    application.owner_ref in boolean_owners
                    and request.invocation_preserves_population(
                        application.owner_ref, branch_id=branch.branch_id
                    )
                    for application in selected
                ):
                    continue
                supplied_values = {
                    value.value_id
                    for value in request.catalog_values
                    if value.target_ref == param.param_ref
                }
                if any(
                    application.value_ref in supplied_values for application in selected
                ):
                    continue
                arguments = (
                    tuple(
                        _argument_value(request, application.value_ref)
                        for application in selected
                        if application.owner_ref
                        and application.owner_ref.startswith("read_population:")
                    )
                    if selected
                    else (
                        (argument_text(param.default),)
                        if param.default is not None and not param.required
                        else ()
                    )
                )
                if covers_population(source, param, arguments):
                    continue
                failed.append(f"population_coverage:{occurrence.id}:{param.param_ref}")
    return tuple(dict.fromkeys(failed))


def _argument_value(request, value_ref):
    from fervis.lookup.source_binding.invocation_bindings import invocation_value
    from fervis.lookup.answer_program.values import LiteralValuePayload
    payload = invocation_value(request, value_ref).payload
    if not isinstance(payload, LiteralValuePayload):
        raise ValueError("population invocation controls must be scalar literals")
    return payload.value
