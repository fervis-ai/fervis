"""Parse and validate grounding compatibility reviews."""

from __future__ import annotations

from fervis.lookup.grounding import provider_contract as provider_output
from fervis.lookup.grounding.model import (
    CompatibleInputBinding,
    InputBindingCompatibility,
    InputBindingOption,
    GroundingRequest,
    GroundingCompatibilityResult,
    IdentifierKind,
    KnownInputBindingTask,
    LookupTextResolutionDecision,
    NO_SHOWN_RESOURCE_TYPE,
    ResourceTypeMatch,
    LookupRequestParameter,
    TimeResolutionIntent,
    resolver_fit_question_for_option,
)
from fervis.lookup.grounding.surface import resolver_option_surface
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogParameterValueError,
    CatalogScalarParameterValue,
    parse_catalog_parameter_value,
)
from fervis.lookup.grounding.time_intents import normalize_grounding_date_intent


def parse_grounding_compatibility(
    payload: dict[str, object],
    *,
    request: GroundingRequest,
) -> GroundingCompatibilityResult:
    output = provider_output.GroundingOutput.parse(payload)
    time_resolutions = _time_resolutions_from_payload(
        output.known_time_resolutions,
        request=request,
    )
    reviews = output.known_input_binding_reviews
    tasks_by_id = {task.known_input_id: task for task in request.tasks}
    options_by_task = {
        task.known_input_id: {option.id: option for option in task.options}
        for task in request.tasks
    }
    compatibilities: list[InputBindingCompatibility] = []
    seen: set[str] = set()
    for known_input_id, review in reviews.items():
        if known_input_id in seen:
            raise ValueError("duplicate grounding review")
        if known_input_id not in tasks_by_id:
            raise ValueError("review references unknown known input")
        task = tasks_by_id[known_input_id]
        _required_text(review.resource_type_basis)
        resource_type_x = _required_text(review.resource_type_x)
        if resource_type_x not in {
            *task.shown_resource_types,
            NO_SHOWN_RESOURCE_TYPE,
        }:
            raise ValueError("grounding resource_type_x was not shown")
        _required_text(review.identifier_kind_basis)
        identifier_kind = IdentifierKind(review.identifier_kind)
        compatible_bindings = _compatible_bindings(
            review.option_reviews,
            request=request,
            task=task,
            options_by_id=options_by_task[known_input_id],
            resource_type_x=resource_type_x,
            identifier_kind=identifier_kind,
        )
        seen.add(known_input_id)
        compatibilities.append(
            InputBindingCompatibility(
                known_input_id=known_input_id,
                bindings=compatible_bindings,
            )
        )
    missing = set(tasks_by_id) - seen
    if missing:
        raise ValueError("grounding compatibility review missing known input")
    return GroundingCompatibilityResult(
        compatibilities=tuple(compatibilities),
        time_resolutions=time_resolutions,
    )


def _time_resolutions_from_payload(
    resolutions: dict[str, provider_output.KnownTimeResolutionOutput],
    *,
    request: GroundingRequest,
) -> tuple[TimeResolutionIntent, ...]:
    tasks_by_id = {task.known_input_id: task for task in request.time_tasks}
    if set(resolutions) != set(tasks_by_id):
        raise ValueError("grounding time resolutions must cover every time input")
    output: list[TimeResolutionIntent] = []
    for known_input_id, resolution in resolutions.items():
        task = tasks_by_id[known_input_id]
        date_intent = resolution.date_intent
        expression = _required_text(date_intent.expression)
        normalized = normalize_grounding_date_intent(
            expression,
            date_intent.intent,
            path=f"known_time_resolutions.{task.known_input_id}.date_intent",
        )
        if expression != task.time_expression:
            raise ValueError("grounding date_intent expression mismatch")
        output.append(
            TimeResolutionIntent(
                known_input_id=task.known_input_id,
                date_intent=normalized,
            )
        )
    return tuple(output)


def _compatible_bindings(
    reviews: dict[str, provider_output.OptionReviewOutput],
    *,
    request: GroundingRequest,
    task: KnownInputBindingTask,
    options_by_id: dict[str, InputBindingOption],
    resource_type_x: str,
    identifier_kind: IdentifierKind,
) -> tuple[CompatibleInputBinding, ...]:
    if set(reviews) != set(options_by_id):
        raise ValueError("grounding option reviews must cover every binding option")
    compatible: list[CompatibleInputBinding] = []
    for option_id, option in options_by_id.items():
        review = reviews[option_id]
        resource_type = _required_text(review.resource_type)
        if resource_type != option.candidate.entity_kind:
            raise ValueError("grounding option resource_type mismatch")
        resource_type_match = ResourceTypeMatch(review.resource_type_match)
        expected_match = (
            ResourceTypeMatch.SAME_RESOURCE_TYPE
            if resource_type == resource_type_x
            else ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE
        )
        if resource_type_match is not expected_match:
            raise ValueError("grounding resource_type_match contradicts resource types")
        expected_question = resolver_fit_question_for_option(
            task=task,
            option=option,
        )
        if _required_text(review.resolver_fit_question) != expected_question:
            raise ValueError("grounding resolver_fit_question mismatch")
        _required_text(review.because)
        resolution = review.resolution
        decision = LookupTextResolutionDecision(resolution.decision)
        if decision is LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT:
            if resource_type_match is not ResourceTypeMatch.SAME_RESOURCE_TYPE:
                raise ValueError(
                    "positive grounding review requires the same resource type"
                )
            compatible.append(
                _compatible_binding(
                    resolution,
                    request=request,
                    lookup_text=task.lookup_text,
                    option=option,
                    identifier_kind=identifier_kind,
                )
            )
        elif (
            resolution.lookup_request_params
            or resolution.returned_identity_verification_fields
        ):
            raise ValueError("negative grounding review must not select read inputs")
    return tuple(compatible)


def _compatible_binding(
    resolution: provider_output.ResolverResolutionOutput,
    *,
    request: GroundingRequest,
    lookup_text: str,
    option: InputBindingOption,
    identifier_kind: IdentifierKind,
) -> CompatibleInputBinding:
    surface = resolver_option_surface(request, option)
    lookup_request_parameters: list[LookupRequestParameter] = []
    compiled_lookup_values: list[CatalogScalarParameterValue] = []
    selected_param_refs = tuple(
        request_param.param_ref for request_param in resolution.lookup_request_params
    )
    if len(selected_param_refs) != len(set(selected_param_refs)):
        raise ValueError("grounding review repeats a request parameter")
    for request_param in resolution.lookup_request_params:
        param_ref = request_param.param_ref
        supplied_value = request_param.value
        parameter, expected_value = surface.compiled_request_value(
            param_ref,
            lookup_text=lookup_text,
        )
        try:
            parsed_value = parse_catalog_parameter_value(
                supplied_value,
                type_name=parameter.type.value,
                choices=parameter.choices,
            )
        except CatalogParameterValueError as exc:
            raise ValueError(
                "grounding review supplies an invalid request value"
            ) from exc
        if not isinstance(parsed_value, (str, int, float, bool)):
            raise ValueError("grounding request parameter must be scalar")
        if not _same_scalar_value(parsed_value, expected_value):
            raise ValueError("grounding request value must equal the lookup value")
        compiled_lookup_values.append(expected_value)
        lookup_request_parameters.append(
            LookupRequestParameter(param_ref=param_ref, value=parsed_value)
        )
    missing_required_params = {
        parameter.param_ref
        for parameter in surface.request_parameters
        if parameter.required and parameter.default is None
    } - set(selected_param_refs)
    if missing_required_params:
        raise ValueError("grounding review omits a required request parameter")
    match_paths = tuple(resolution.returned_identity_verification_fields)
    if not match_paths:
        raise ValueError("positive grounding review requires a response match field")
    if len(match_paths) != len(set(match_paths)):
        raise ValueError("grounding review repeats a response match field")
    for field_path in match_paths:
        compiled_lookup_values.append(
            surface.compiled_match_value(field_path, lookup_text=lookup_text)
        )
    lookup_value = compiled_lookup_values[0]
    if any(
        not _same_scalar_value(value, lookup_value)
        for value in compiled_lookup_values[1:]
    ):
        raise ValueError(
            "grounding request and match fields parse the lookup differently"
        )
    return CompatibleInputBinding(
        option_id=option.id,
        lookup_value=lookup_value,
        identifier_kind=identifier_kind,
        lookup_request_parameters=tuple(lookup_request_parameters),
        returned_identity_verification_field_paths=match_paths,
    )


def _same_scalar_value(
    left: CatalogScalarParameterValue,
    right: CatalogScalarParameterValue,
) -> bool:
    return type(left) is type(right) and left == right


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("grounding compatibility review requires non-empty text")
    return text
