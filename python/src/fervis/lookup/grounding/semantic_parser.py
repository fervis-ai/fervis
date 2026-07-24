"""Parse semantic identity-grounding reviews mechanically."""

from __future__ import annotations

from fervis.lookup.relation_catalog.model import requires_caller_supplied_input

from fervis.lookup.grounding import semantic_provider_contract as output
from fervis.lookup.grounding.identity import (
    IdentifierKind,
    InputBindingPurpose,
    InputBindingOption,
    LookupTextResolutionDecision,
    ResourceTypeMatch,
)
from fervis.lookup.grounding.semantic import (
    CanonicalInputValue,
    CompatibleIdentityRoute,
    IdentityGroundingTask,
    SemanticGroundingResult,
    SemanticGroundingRequest,
    identity_resolution_tasks,
)
from fervis.lookup.grounding.surface import resolver_option_surface_from_catalog
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.grounding.time_intents import normalize_grounding_date_intent
from fervis.lookup.grounding.time_resolution import Field, Status, resolve_time


def parse_semantic_grounding(
    payload: dict[str, object],
    *,
    request: SemanticGroundingRequest,
) -> SemanticGroundingResult:
    parsed = output.SemanticGroundingOutput.parse(payload)
    canonical_values = list(_time_values(parsed, request=request))
    tasks = {task.task_ref: task for task in request.tasks}
    if set(parsed.reference_reviews) != set(tasks):
        raise ValueError("semantic grounding must review every reference task once")
    compatible_by_task: dict[str, tuple[CompatibleIdentityRoute, ...]] = {}
    identity_tasks: list[IdentityGroundingTask] = []
    for task_ref, raw_review in parsed.reference_reviews.items():
        task = tasks[task_ref]
        review = raw_review.parse_as(output.IdentityGroundingReviewOutput)
        _required_text(review.identifier_kind_basis)
        identifier_kind = IdentifierKind(review.identifier_kind)
        purpose = InputBindingPurpose(review.purpose)
        purpose_options = {
            option.id: option
            for option in task.options
            if option.purpose is purpose
        }
        if not purpose_options:
            raise ValueError("semantic grounding selected an unavailable purpose")
        resource_types = {
            option.candidate.entity_kind for option in purpose_options.values()
        }
        if set(review.resource_type_reviews) != resource_types:
            raise ValueError("resource-type review does not cover every shown type")
        type_matches = {
            resource_type: ResourceTypeMatch(resource_review.compatibility)
            for resource_type, resource_review in review.resource_type_reviews.items()
        }
        options = purpose_options
        compatible: list[CompatibleIdentityRoute] = []
        reviewed_routes: set[str] = set()
        for resource_type, resource_review in review.resource_type_reviews.items():
            _required_text(resource_review.compatibility_basis)
            expected_routes = {
                option.id
                for option in purpose_options.values()
                if option.candidate.entity_kind == resource_type
            }
            if set(resource_review.route_reviews) != expected_routes:
                raise ValueError("resource-type review does not cover its routes")
            reviewed_routes.update(resource_review.route_reviews)
            for route_ref, route_review in resource_review.route_reviews.items():
                option = options[route_ref]
                _required_text(route_review.assessment_basis)
                resolution = route_review.resolution
                decision = LookupTextResolutionDecision(resolution.decision)
                if decision is LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT:
                    if (
                        type_matches[resource_type]
                        is not ResourceTypeMatch.POSSIBLE_DENOTED_KIND
                    ):
                        raise ValueError(
                            "compatible route must return a compatible resource type"
                        )
                    compatible.append(
                        _compatible_route(
                            resolution,
                            request=request,
                            option=option,
                            identifier_kind=identifier_kind,
                            operand=request.input(task.input_ref).operand,
                        )
                    )
                elif (
                    resolution.lookup_request_params
                    or resolution.returned_identity_verification_fields
                ):
                    raise ValueError("incompatible route selected lookup mechanics")
        if reviewed_routes != set(options):
            raise ValueError("route review does not cover every shown route")
        compatible_by_task[task_ref] = tuple(compatible)
        identity_tasks.append(
            IdentityGroundingTask(
                task_ref=task.task_ref,
                input_ref=task.input_ref,
                use_refs=task.use_refs,
                expected_set_ref=task.expected_set_ref,
                options=tuple(purpose_options.values()),
            )
        )
    return SemanticGroundingResult(
        identity_tasks=identity_resolution_tasks(
            tuple(identity_tasks),
            compatible_bindings_by_task_ref=compatible_by_task,
        ),
        canonical_values=tuple(canonical_values),
    )


def _compatible_route(
    resolution: output.ResolverMechanicsOutput,
    *,
    request: SemanticGroundingRequest,
    option: InputBindingOption,
    identifier_kind: IdentifierKind,
    operand: str | tuple[str, ...],
) -> CompatibleIdentityRoute:
    surface = resolver_option_surface_from_catalog(request.resolver_catalog, option)
    parameter_refs = tuple(resolution.lookup_request_params)
    if len(parameter_refs) != len(set(parameter_refs)):
        raise ValueError("semantic grounding repeats a request parameter")
    for param_ref in parameter_refs:
        surface.parameter(param_ref)
    missing_required = {
        parameter.param_ref
        for parameter in surface.request_parameters
        if requires_caller_supplied_input(parameter)
    } - set(parameter_refs)
    if missing_required:
        raise ValueError("semantic grounding omits a required request parameter")
    field_paths = tuple(resolution.returned_identity_verification_fields)
    if len(field_paths) != len(set(field_paths)):
        raise ValueError("semantic grounding repeats a verification field")
    for field_path in field_paths:
        surface.match_field(field_path)
    operands = (operand,) if isinstance(operand, str) else operand
    for lookup_text in operands:
        compiled = [
            surface.compiled_request_value(param_ref, lookup_text=lookup_text)[1]
            for param_ref in parameter_refs
        ]
        compiled.extend(
            surface.compiled_match_value(field_path, lookup_text=lookup_text)
            for field_path in field_paths
        )
        if any(
            type(value) is not type(compiled[0]) or value != compiled[0]
            for value in compiled[1:]
        ):
            raise ValueError(
                "grounding request and verification fields parse the input differently"
            )
    return CompatibleIdentityRoute(
        option_id=option.id,
        identifier_kind=identifier_kind,
        lookup_request_param_refs=parameter_refs,
        returned_identity_verification_field_paths=field_paths,
    )


def _time_values(
    parsed: output.SemanticGroundingOutput,
    *,
    request: SemanticGroundingRequest,
) -> tuple[CanonicalInputValue, ...]:
    tasks = {task.task_ref: task for task in request.time_tasks}
    if set(parsed.time_resolutions) != set(tasks):
        raise ValueError("semantic grounding must resolve every time task once")
    values: list[CanonicalInputValue] = []
    for position, (task_ref, resolution) in enumerate(
        parsed.time_resolutions.items(), start=1
    ):
        task = tasks[task_ref]
        date_intent = resolution.date_intent
        if date_intent.expression != task.expression:
            raise ValueError("semantic time resolution changed the expression")
        intent = normalize_grounding_date_intent(
            date_intent.expression,
            date_intent.intent,
            path=f"time_resolutions.{task_ref}.date_intent",
        )
        resolved = resolve_time(
            task.expression,
            intent=intent,
            anchor_date=request.runtime_date,
            timezone=request.timezone,
        )
        if resolved.get(Field.STATUS) != Status.RESOLVED:
            raise ValueError("semantic time input could not be resolved")
        granularity = _time_granularity(resolved)
        fact_value = FactValue.time(
            id=f"canonical_time_value_{position}",
            known_input_id=task.input_ref,
            expression=task.expression,
            intent=dict(resolved.get(Field.INTENT) or {}),
            resolved_start=str(resolved.get(Field.START) or ""),
            resolved_end=str(resolved.get(Field.END) or ""),
            granularity=granularity,
            proof_refs=(f"question_input:{task.input_ref}",),
        )
        values.append(
            CanonicalInputValue(
                canonical_value_id=fact_value.id,
                input_ref=task.input_ref,
                use_refs=task.use_refs,
                typed_value=fact_value,
                certification_refs=(f"time_resolution:{task_ref}",),
            )
        )
    return tuple(values)


def _time_granularity(resolved: dict[str, object]) -> str:
    intent = resolved.get(Field.INTENT)
    if not isinstance(intent, dict):
        return ""
    anchor = intent.get("anchor_period")
    return str(
        intent.get("precision")
        or intent.get("unit")
        or (anchor.get("unit") if isinstance(anchor, dict) else "")
        or ""
    )


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("semantic grounding requires a non-empty assessment basis")
    return text


__all__ = ["parse_semantic_grounding"]
