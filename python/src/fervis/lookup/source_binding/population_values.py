"""Finite retrieval coverage follows row predicates, not parameter enums."""


def supports_population_argument(param) -> bool:
    from fervis.lookup.plan_execution.declared_values import declared_kind, DeclaredValueKind
    return declared_kind(param.type.value) in {
        DeclaredValueKind.STRING, DeclaredValueKind.INTEGER,
        DeclaredValueKind.DECIMAL, DeclaredValueKind.BOOLEAN,
    }


def population_values(source, param) -> tuple[str, ...]:
    contract = param.population
    if contract is None:
        return ()
    legal = set(param.finite_choices)
    declared = {
        *contract.unfiltered_values,
        *(item.argument for item in contract.value_mapping),
    }
    for value in declared:
        validate_population_argument(param, value)
    from fervis.lookup.plan_execution.declared_values import declared_key
    field = population_field(source, contract)
    admissions: dict[tuple[str, str], frozenset[tuple[str, str]]] = {}
    for item in contract.value_mapping:
        key = declared_key(item.argument, param.type.value)
        rows = frozenset(declared_key(value, field.type.value if field is not None else None)
                         for value in item.row_values)
        if key in admissions and admissions[key] != rows:
            raise ValueError("the same typed argument cannot admit different row populations")
        admissions[key] = rows
    if contract.unfiltered_values:
        return (contract.unfiltered_values[0],)
    if contract.preserves_default and param.default is not None and (not legal or argument_text(param.default) in legal):
        return (argument_text(param.default),)
    field = population_field(source, contract)
    if field is None or field.nullable or not field.declared_value_domain:
        return ()
    domain = set(field.finite_choices)
    if not domain:
        return ()
    admitted = {value for item in contract.value_mapping for value in item.row_values}
    if not admitted <= domain:
        raise ValueError("population contract references an undeclared returned value")
    if admitted != domain:
        return ()
    complete_arguments = tuple(item.argument for item in contract.value_mapping if set(item.row_values) == domain)
    if complete_arguments:
        return (complete_arguments[0],)
    return tuple(item.argument for item in contract.value_mapping if item.row_values)


def validate_population_argument(param, value):
    from fervis.lookup.plan_execution.declared_values import parse_declared_value
    from fervis.lookup.plan_execution.errors import RelationEngineError
    if param.finite_choices and value not in param.finite_choices:
        raise ValueError("population contract references an undeclared argument")
    if not supports_population_argument(param):
        raise ValueError("population controls require a supported scalar parameter")
    try:
        parse_declared_value(value, param.type.value)
    except RelationEngineError as exc:
        raise ValueError("population argument does not match its parameter type") from exc


def covers_population(source, param, argument_values) -> bool:
    contract = param.population
    if contract is None:
        return False
    if contract.preserves_population:
        return True
    if contract.preserves_default and param.default is not None and set(argument_values) == {argument_text(param.default)}:
        return True
    # Validate the request-value evidence before checking this selection.
    complete = population_values(source, param)
    if not complete:
        return False
    values = set(argument_values)
    if values & set(contract.unfiltered_values):
        return True
    field = population_field(source, contract)
    if field is None or field.nullable or not field.declared_value_domain:
        return False
    admitted = {
        value
        for item in contract.value_mapping
        if item.argument in values
        for value in item.row_values
    }
    return admitted == set(field.finite_choices)


def population_owner(occurrence_id, parameter_ref):
    return f"read_population:{occurrence_id}:{parameter_ref}"


def argument_text(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def complete_population_applications(plan, *, request):
    from fervis.lookup.source_binding.occurrences import occurrence_scope
    from fervis.lookup.source_binding.model import (
        InvocationValueApplication,
        InvocationTargetApplication,
    )
    from fervis.lookup.answer_program.values import ValueProjectionKind
    from fervis.host_api.contracts import ParameterSemantics

    added = []
    for branch in request.strategy.branches:
        scope = occurrence_scope(request, plan, branch.branch_id)
        for occurrence in scope.occurrences:
            applied = scope.applications_for(
                request, plan, branch_id=branch.branch_id, occurrence=occurrence
            )
            targets = {
                target.target_ref
                for application in applied
                for target in application.target_applications
            }
            source = request.source_catalog.source(occurrence.source_ref)
            for declared_param in source.params:
                from dataclasses import replace
                param = replace(declared_param, population=request.parameter_population(source.id, declared_param.param_ref))
                if param.semantics is not ParameterSemantics.OPAQUE_QUERY_PARAM:
                    continue
                if param.param_ref in targets or (
                    not param.required and param.default is None and param.default_is_known
                ):
                    continue
                if (
                    not param.required
                    and param.default is not None
                    and covers_population(
                        source, param, (argument_text(param.default),)
                    )
                ):
                    continue
                values = population_values(source, param)
                if not values:
                    continue
                owner = population_owner(occurrence.id, param.param_ref)
                for ordinal, value in enumerate(values):
                    value_ref = population_control_ref(source.id, param.param_ref, ordinal)
                    added.append(
                        InvocationValueApplication(
                            application_ref=f"{branch.branch_id}:{owner}:{ordinal}",
                            branch_id=branch.branch_id,
                            source_ref=source.id,
                            value_ref=value_ref,
                            owner_ref=owner,
                            target_applications=(
                                InvocationTargetApplication(
                                    mapping_basis="The source row-admission evidence covers the declared row domain.",
                                    target_ref=param.param_ref,
                                    value_ref=value_ref,
                                    projection=ValueProjectionKind.WHOLE_VALUE,
                                    component_ref=None,
                                ),
                            ),
                        )
                    )
    return tuple(added)


def population_control_ref(source_ref, parameter_ref, ordinal):
    return f"source_population:{source_ref}:{parameter_ref}:{ordinal}"


def population_control_values(request):
    from dataclasses import replace
    from fervis.lookup.available_sources import source_value_literal
    result = {}
    for source in request.source_catalog.sources:
        for declared in source.params:
            param = replace(declared, population=request.parameter_population(source.id, declared.param_ref))
            for ordinal, argument in enumerate(population_values(source, param)):
                ref = population_control_ref(source.id, param.param_ref, ordinal)
                result[ref] = source_value_literal(
                    value_ref=ref, value=argument, declared_type=param.type,
                    label=param.name, source_ref=source.id,
                    proof_refs=(request.source_catalog.contract_snapshot.ref, source.id, param.param_ref, ref),
                )
    return result


def population_field(source, contract):
    if contract is None or not contract.field_path:
        return None
    return next((field for field in source.fields
                 if contract.field_path in (field.path, field.response_path, field.field_ref)), None)


def contradictory_request_predicates(plan, *, request):
    """One predicate must have the same truth set in returned and request values."""
    from fervis.lookup.available_sources import SourceChoiceSurfaceKind
    owners = {item.requirement_ref for item in request.index.boolean_requirements}
    from fervis.lookup.source_binding.invocation_bindings import invocation_target_value
    from fervis.lookup.plan_execution.declared_values import parse_declared_value
    selections = {}
    for application in plan.invocation_applications:
        if application.owner_ref not in owners:
            continue
        for target in application.target_applications:
            population = request.parameter_population(application.source_ref, target.target_ref)
            if population is None or not population.value_mapping:
                continue
            value = invocation_target_value(request, source_ref=application.source_ref, target=target)
            values = value if isinstance(value, tuple) else (value,)
            key = (application.branch_id, application.source_ref, target.target_ref, application.owner_ref)
            selections.setdefault(key, set()).update(values)
    failed = []
    for (branch_id, source_ref, target_ref, owner), arguments in selections.items():
        source = request.source_catalog.source(source_ref)
        population = request.parameter_population(source_ref, target_ref)
        field = population_field(source, population)
        if field is None or population is None or not population.value_mapping:
            continue
        reviews = [review
                   for branch in plan.subject_binding.branch_realizations if branch.branch_id == branch_id
                   for review in branch.surface_reviews
                   if (surface := request.source_catalog.choice_surface(review.surface_ref)).kind is SourceChoiceSurfaceKind.RETURNED_FIELD
                   and surface.source_ref == source_ref and surface.target_ref == field.field_ref]
        if not reviews:
            continue
        required_values = {request.source_catalog.choice_value(choice.choice_ref).value
                           for review in reviews for choice in review.choice_reviews
                           if owner in choice.selection_requirement_refs}
        param = next(item for item in source.params if item.param_ref == target_ref)
        arguments = {parse_declared_value(value, param.type.value) for value in arguments}
        mapped_arguments = {parse_declared_value(item.argument, param.type.value) for item in population.value_mapping}
        admitted = {value for item in population.value_mapping
                    if parse_declared_value(item.argument, param.type.value) in arguments
                    for value in item.row_values}
        if not arguments <= mapped_arguments or admitted != required_values:
            failed.append(owner)
    return tuple(dict.fromkeys(failed))


def parameter_comparison_field(source, param, population):
    """Validate one declared scalar admission relation at the catalog boundary."""
    from fervis.lookup.expression_operators import ExpressionBinaryOperator, infer_operator_result
    from fervis.lookup.relation_catalog.row_sources import semantic_type_for_row_source_type
    field = population_field(source, population)
    if field is None or not population.comparison_operator:
        raise ValueError("parameter comparison lacks its declared field")
    infer_operator_result(ExpressionBinaryOperator(population.comparison_operator), (
        semantic_type_for_row_source_type(field.type),
        semantic_type_for_row_source_type(param.type),
    ))
    return field
