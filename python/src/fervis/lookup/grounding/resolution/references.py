"""Catalog-backed named-reference grounding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.relation_catalog import (
    RelationCatalog,
    source_field_has_primary_stable_identity,
)
from fervis.lookup.relation_catalog.selection import (
    EntityTargetResolverSelection,
)
from fervis.lookup.source_reads.response import (
    EndpointResponseError,
    SourceReadFailedError,
    extract_source_read_rows,
    observe_source_read_response,
    path_value,
    relative_response_path,
    source_read_completeness,
)
from fervis.lookup.lineage.source_reads import (
    SourceReadLineageScope,
    record_source_read_observation,
    record_source_read_error,
    require_catalog_endpoint_for_lineage,
)
from fervis.lookup.grounding.model import (
    CanonicalInputLedger,
    GroundedInputUse,
    GroundingCandidate,
    GroundingIssue,
    GroundingTerminalKind,
    GroundingRequestedFactCard,
    InputBindingOption,
    InputBindingRoute,
    InputBindingCompatibility,
    KnownInputBindingTask,
    ResolverOutputFieldCard,
    ResolverQueryParamCard,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceField,
    RowSourceKind,
    RowSourceParam,
    RowSourceParamSemantics,
)
from fervis.lookup.fact_plan.values import FactValue, IdentityValuePayload
from fervis.lookup.question_contract import (
    KnownInputKind,
    QuestionContract,
    RequestedFact,
    RequestedFactKnownInput,
)
from fervis.memory.identities import MemoryIdentityValue

from .values import _grounded_value_id, _normalize_lookup_text, _symbol

_MISSING = object()


@dataclass(frozen=True)
class _ResolvedReferenceTasks:
    values: tuple[FactValue, ...] = ()
    issues: tuple[GroundingIssue, ...] = ()
    model_tasks: tuple[KnownInputBindingTask, ...] = ()


@dataclass(frozen=True)
class _ResolvedLookupRow:
    row: dict[str, Any]
    matched_field_ref: str = ""
    matched_field_path: str = ""


@dataclass(frozen=True)
class _LookupField:
    ref: str
    path: str
    relative_path: str


def _reference_binding_tasks(
    question_contract: QuestionContract,
    *,
    resolver_row_sources: RowSourceCatalog,
) -> tuple[KnownInputBindingTask, ...]:
    tasks: list[KnownInputBindingTask] = []
    facts_by_id = {fact.id: fact for fact in question_contract.requested_facts}
    for known, requested_fact_ids in _reference_known_input_bindings(question_contract):
        if known.kind != KnownInputKind.REFERENCE:
            continue
        requested_fact_id = requested_fact_ids[0] if requested_fact_ids else ""
        requested_facts = tuple(
            fact for fact_id in requested_fact_ids if (fact := facts_by_id.get(fact_id))
        )
        requested_fact_cards = tuple(
            _requested_fact_card(fact) for fact in requested_facts
        )
        options = _reference_binding_options(
            known,
            resolver_row_sources=resolver_row_sources,
        )
        if not options:
            tasks.append(
                KnownInputBindingTask(
                    known_input_id=known.id,
                    known_input_text=known.text,
                    known_input_kind=known.kind.value,
                    requested_fact_id=requested_fact_id,
                    options=(),
                    known_input_description=known.description,
                    lookup_text=known.lookup_text or known.text,
                    applies_to_requested_fact_ids=requested_fact_ids,
                    requested_facts=requested_fact_cards,
                )
            )
            continue
        task = KnownInputBindingTask(
            known_input_id=known.id,
            known_input_text=known.text,
            known_input_kind=known.kind.value,
            requested_fact_id=requested_fact_id,
            options=options,
            known_input_description=known.description,
            lookup_text=known.lookup_text or known.text,
            applies_to_requested_fact_ids=requested_fact_ids,
            requested_facts=requested_fact_cards,
        )
        tasks.append(task)
    return tuple(tasks)


def _requested_fact_card(fact: RequestedFact) -> GroundingRequestedFactCard:
    population = fact.answer_population
    return GroundingRequestedFactCard(
        requested_fact_id=fact.id,
        answer_fact=fact.description,
        answer_population_label=(
            population.population_label if population is not None else ""
        ),
        answer_population_counted_unit=(
            population.counted_unit if population is not None else ""
        ),
        answer_outputs=tuple(
            {
                "answer_output_id": output.id,
                "description": output.description,
            }
            for output in fact.answer_outputs
        ),
    )


def _reference_known_input_bindings(
    question_contract: QuestionContract,
) -> tuple[tuple[RequestedFactKnownInput, tuple[str, ...]], ...]:
    if question_contract.question_inputs:
        return tuple(
            (
                known,
                question_contract.requested_fact_ids_for_input(known.id),
            )
            for known in question_contract.question_inputs
        )
    return tuple(
        (
            known,
            (fact.id,),
        )
        for fact in question_contract.requested_facts
        for known in fact.known_inputs
    )


def _reference_binding_options(
    known: RequestedFactKnownInput,
    *,
    resolver_row_sources: RowSourceCatalog,
) -> tuple[InputBindingOption, ...]:
    route_options: list[InputBindingOption] = []
    resolvers = tuple(
        source
        for source in resolver_row_sources.sources
        if source.kind == RowSourceKind.API_READ
    )
    option_index = 1
    for resolver_source in resolvers:
        for return_field in _resolver_identity_fields(resolver_source):
            return_identity = return_field.identity
            if return_identity is None:
                continue
            for lookup_param, lookup_fields in _resolver_lookup_groups(
                resolver_source,
                return_field=return_field,
            ):
                route = InputBindingRoute(
                    known_input_id=known.id,
                    resolver_row_source_id=resolver_source.id,
                    resolver_read_id=resolver_source.read_id,
                    resolver_endpoint_name=resolver_source.label,
                    resolver_resource_names=resolver_source.resource_names,
                    lookup_param_id=lookup_param.id if lookup_param is not None else "",
                    lookup_param_ref=(
                        lookup_param.param_ref if lookup_param is not None else ""
                    ),
                    lookup_field_ids=tuple(field.id for field in lookup_fields),
                    lookup_field_refs=(
                        tuple(field.field_ref for field in lookup_fields)
                    ),
                    return_field_id=return_field.id,
                    return_field_ref=return_field.field_ref,
                    identity_type=return_identity.entity_ref,
                    identity_field=return_identity.identity_field,
                    display=_binding_path(
                        resolver_source=resolver_source,
                        lookup_param=lookup_param,
                        lookup_fields=lookup_fields,
                        return_field=return_field,
                    ),
                    query_params=_resolver_query_param_cards(resolver_source),
                    selected_output_fields=_resolver_selected_output_field_cards(
                        resolver_source,
                        lookup_fields=lookup_fields,
                        return_field=return_field,
                    ),
                )
                route_options.append(
                    InputBindingOption(
                        id=f"bind_{_symbol(known.id)}_{option_index}",
                        known_input_id=known.id,
                        path=route.display,
                        route=route,
                    )
                )
                option_index += 1
    return tuple(route_options)


def _resolve_reference_tasks(
    tasks: tuple[KnownInputBindingTask, ...],
    *,
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: Any,
    memory_identity_values: tuple[MemoryIdentityValue, ...] = (),
) -> _ResolvedReferenceTasks:
    values: list[FactValue] = []
    issues: list[GroundingIssue] = []
    model_tasks: list[KnownInputBindingTask] = []
    for task in tasks:
        route_options = tuple(
            option for option in task.options if option.route is not None
        )
        memory_candidates = _memory_identity_candidates(
            task,
            route_options=route_options,
            memory_identity_values=memory_identity_values,
        )
        if len(memory_candidates) == 1:
            value = memory_candidates[0].resolved_value
            if value is not None:
                values.append(value)
            continue
        if len(memory_candidates) > 1:
            issues.append(
                GroundingIssue(
                    kind=GroundingTerminalKind.AMBIGUOUS_REFERENCE,
                    known_input_id=task.known_input_id,
                    requested_fact_id=task.requested_fact_id,
                    message="memory contains multiple canonical identity matches",
                    known_input_text=task.lookup_text or task.known_input_text,
                    known_input_description=task.known_input_description,
                    candidates=tuple(candidate.path for candidate in memory_candidates),
                    proof_refs=(f"known_input:{task.known_input_id}",),
                )
            )
            continue
        if not route_options:
            issues.append(
                GroundingIssue(
                    kind=GroundingTerminalKind.UNSUPPORTED_REFERENCE,
                    known_input_id=task.known_input_id,
                    requested_fact_id=task.requested_fact_id,
                    message="no catalog binding route was available for known input",
                    known_input_text=task.lookup_text or task.known_input_text,
                    known_input_description=task.known_input_description,
                    proof_refs=(f"known_input:{task.known_input_id}",),
                )
            )
            continue
        if route_options:
            model_tasks.append(
                replace(
                    task,
                    options=tuple(
                        replace(
                            candidate,
                            id=f"bind_{_symbol(task.known_input_id)}_{index}",
                        )
                        for index, candidate in enumerate(route_options, start=1)
                    ),
                )
            )
            continue
        issues.append(
            GroundingIssue(
                kind=GroundingTerminalKind.UNRESOLVED_REFERENCE,
                known_input_id=task.known_input_id,
                requested_fact_id=task.requested_fact_id,
                message="resolver returned no canonical identity match",
                known_input_text=task.lookup_text or task.known_input_text,
                known_input_description=task.known_input_description,
                proof_refs=(f"known_input:{task.known_input_id}",),
            )
        )
    return _ResolvedReferenceTasks(
        values=tuple(values),
        issues=tuple(issues),
        model_tasks=tuple(model_tasks),
    )


def _memory_identity_candidates(
    task: KnownInputBindingTask,
    *,
    route_options: tuple[InputBindingOption, ...],
    memory_identity_values: tuple[MemoryIdentityValue, ...],
) -> tuple[InputBindingOption, ...]:
    lookup_text = task.lookup_text or task.known_input_text
    expected = _normalize_lookup_text(lookup_text)
    if not expected:
        return ()
    compatible_routes = tuple(
        route
        for option in route_options
        for route in (option.route,)
        if route is not None
    )
    output: list[InputBindingOption] = []
    seen: set[tuple[str, str, str]] = set()
    for identity in memory_identity_values:
        if _normalize_lookup_text(identity.lookup_text) != expected:
            continue
        if not _memory_identity_is_compatible(
            identity,
            route_options=compatible_routes,
        ):
            continue
        key = (identity.identity_type, identity.identity_field, identity.value)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            InputBindingOption(
                id=f"bind_{_symbol(task.known_input_id)}_memory_{len(output) + 1}",
                known_input_id=task.known_input_id,
                path=(
                    f"{identity.display_label} ({identity.identity_type} from memory)"
                ),
                resolved_value=FactValue.identity(
                    id=_grounded_value_id(task.known_input_id),
                    identity_type=identity.identity_type,
                    identity_field=identity.identity_field,
                    value=identity.value,
                    display_value=identity.display_label or identity.lookup_text,
                    proof_refs=(
                        *identity.proof_refs,
                        f"known_input:{task.known_input_id}",
                    ),
                    applies_to_requested_fact_ids=_task_requested_fact_ids(task),
                ),
            )
        )
    return tuple(output)


def _memory_identity_is_compatible(
    identity: MemoryIdentityValue,
    *,
    route_options: tuple[InputBindingRoute, ...],
) -> bool:
    if not route_options:
        return True
    for route in route_options:
        if (
            route.identity_type == identity.identity_type
            and route.identity_field == identity.identity_field
        ):
            return True
    return False


def _resolver_identity_fields(source: RowSource) -> tuple[RowSourceField, ...]:
    return tuple(
        field
        for field in source.fields
        if source_field_has_primary_stable_identity(field)
        and field.identity is not None
        and field.identity.entity_ref
        and field.identity.identity_field
        and _identity_field_belongs_to_row_resource(source, field)
    )


def _identity_field_belongs_to_row_resource(
    source: RowSource,
    field: RowSourceField,
) -> bool:
    if field.identity is None or not source.resource_names:
        return True
    return field.identity.entity_ref in set(source.resource_names)


def _resolver_lookup_groups(
    source: RowSource,
    *,
    return_field: RowSourceField,
) -> tuple[tuple[RowSourceParam | None, tuple[RowSourceField, ...]], ...]:
    if any(param.required and param.default is None for param in source.params):
        return ()
    text_params = tuple(
        param
        for param in source.params
        if (
            not param.required
            and param.type in {"string", "any"}
            and param.semantics != RowSourceParamSemantics.RESPONSE_SHAPE
        )
    )
    text_fields = _identity_lookup_fields(source, return_field=return_field)
    if not text_fields:
        return ()
    groups: list[tuple[RowSourceParam | None, tuple[RowSourceField, ...]]] = [
        (param, fields)
        for param in text_params
        if (fields := _lookup_fields_for_param(param, fields=text_fields))
    ]
    groups.append((None, text_fields))
    return tuple(groups)


def _identity_lookup_fields(
    source: RowSource,
    *,
    return_field: RowSourceField,
) -> tuple[RowSourceField, ...]:
    identity = return_field.identity
    if identity is None:
        return ()
    display_fields = set(identity.display_fields)
    if display_fields:
        return tuple(
            field
            for field in source.fields
            if _field_can_carry_lookup_text(field)
            and (field.id in display_fields or field.field_ref in display_fields)
        )
    return tuple(
        field
        for field in source.fields
        if _field_can_carry_lookup_text(field)
        and _same_identity_object(field.path, return_field.path)
    )


def _field_can_carry_lookup_text(field: RowSourceField) -> bool:
    return field.type in {"string", "any"} and not (
        field.identity is not None and field.identity.primary_key
    )


def _same_identity_object(field_path: str, return_field_path: str) -> bool:
    return _parent_object_path(field_path) == _parent_object_path(return_field_path)


def _parent_object_path(field_path: str) -> str:
    parts = tuple(part for part in str(field_path or "").split(".") if part)
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])


def _lookup_fields_for_param(
    param: RowSourceParam,
    *,
    fields: tuple[RowSourceField, ...],
) -> tuple[RowSourceField, ...]:
    del param
    return fields


def _binding_path(
    *,
    resolver_source: RowSource,
    lookup_param: RowSourceParam | None,
    lookup_fields: tuple[RowSourceField, ...],
    return_field: RowSourceField,
) -> str:
    lookup_label = (
        " / ".join(field.label for field in lookup_fields)
        if lookup_fields
        else (lookup_param.name if lookup_param is not None else "text")
    )
    return (
        f"{resolver_source.label}.{lookup_label} -> "
        f"{resolver_source.label}.{return_field.label}"
    )


def _resolver_query_param_cards(
    source: RowSource,
) -> tuple[ResolverQueryParamCard, ...]:
    cards: list[ResolverQueryParamCard] = []
    for param in source.params:
        if str(param.source) != "query":
            continue
        if param.semantics == RowSourceParamSemantics.RESPONSE_SHAPE:
            continue
        cards.append(
            ResolverQueryParamCard(
                param_ref=param.param_ref,
                name=param.name,
                type=param.type,
                choices=param.choices,
            )
        )
    return tuple(cards)


def _resolver_selected_output_field_cards(
    source: RowSource,
    *,
    lookup_fields: tuple[RowSourceField, ...],
    return_field: RowSourceField,
) -> tuple[ResolverOutputFieldCard, ...]:
    selected_refs = {
        return_field.field_ref,
        *(field.field_ref for field in lookup_fields),
    }
    for field in source.fields:
        if field.identity is not None:
            selected_refs.add(field.field_ref)
            selected_refs.update(field.identity.display_fields)
    cards: list[ResolverOutputFieldCard] = []
    for field in source.fields:
        if not (
            field.field_ref in selected_refs
            or field.id in selected_refs
            or bool(field.choices)
        ):
            continue
        cards.append(
            ResolverOutputFieldCard(
                field_ref=field.field_ref,
                field_path=field.path,
                type=field.type,
                choices=field.choices,
                identity=field.identity,
            )
        )
    return tuple(cards)


def _execute_reference_compatibilities(
    *,
    compatibilities: tuple[InputBindingCompatibility, ...],
    tasks: tuple[KnownInputBindingTask, ...],
    resolver_selections: tuple[EntityTargetResolverSelection, ...] = (),
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: Any,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> CanonicalInputLedger:
    options = {option.id: (task, option) for task in tasks for option in task.options}
    compatibilities_by_input = {
        compatibility.known_input_id: compatibility for compatibility in compatibilities
    }
    values: list[FactValue] = []
    uses: list[GroundedInputUse] = []
    issues: list[GroundingIssue] = []
    for task in tasks:
        compatibility = compatibilities_by_input.get(task.known_input_id)
        if compatibility is None:
            issues.append(_unsupported_reference_issue(task))
            continue
        compatible_option_ids = compatibility.binding_option_ids
        if not compatible_option_ids:
            issues.append(
                _unsupported_reference_issue(
                    task,
                    message="no shown resolver could resolve lookup text",
                )
            )
            continue
        task_ledger = _execute_compatible_reference_options(
            task=task,
            compatible_option_ids=compatible_option_ids,
            options=options,
            resolver_selections=resolver_selections,
            full_catalog=full_catalog,
            resolver_row_sources=resolver_row_sources,
            data_access_port=data_access_port,
            source_read_lineage=source_read_lineage,
        )
        values.extend(task_ledger.values)
        uses.extend(task_ledger.uses)
        issues.extend(task_ledger.issues)
    return CanonicalInputLedger(
        values=tuple(values),
        uses=tuple(uses),
        issues=tuple(issues),
    )


def _execute_compatible_reference_options(
    *,
    task: KnownInputBindingTask,
    compatible_option_ids: tuple[str, ...],
    options: dict[str, tuple[KnownInputBindingTask, InputBindingOption]],
    resolver_selections: tuple[EntityTargetResolverSelection, ...] = (),
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: Any,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> CanonicalInputLedger:
    if resolver_selections:
        return _execute_compatible_reference_options_by_resolver_priority(
            task=task,
            compatible_option_ids=compatible_option_ids,
            options=options,
            resolver_selections=resolver_selections,
            full_catalog=full_catalog,
            resolver_row_sources=resolver_row_sources,
            data_access_port=data_access_port,
            source_read_lineage=source_read_lineage,
        )
    values_by_identity: dict[tuple[str, str, str], FactValue] = {}
    uses_by_identity: dict[tuple[str, str, str], GroundedInputUse] = {}
    route_issues: list[GroundingIssue] = []
    for option_id in compatible_option_ids:
        option_task, option = options[option_id]
        if option_task.known_input_id != task.known_input_id:
            raise ValueError("compatible grounding option references wrong input")
        if option.resolved_value is not None:
            key = _identity_value_key(option.resolved_value)
            if key is not None:
                values_by_identity.setdefault(key, option.resolved_value)
            continue
        if option.route is None:
            route_issues.append(_unsupported_reference_issue(task))
            continue
        ledger = _execute_reference_route(
            task=task,
            route=option.route,
            source_read_key=option.id,
            full_catalog=full_catalog,
            resolver_row_sources=resolver_row_sources,
            data_access_port=data_access_port,
            source_read_lineage=source_read_lineage,
        )
        uses_by_value_id = {use.value_id: use for use in ledger.uses}
        for value in ledger.values:
            key = _identity_value_key(value)
            if key is None:
                continue
            values_by_identity.setdefault(key, value)
            use = uses_by_value_id.get(value.id)
            if use is not None:
                uses_by_identity.setdefault(key, use)
        route_issues.extend(ledger.issues)
    ambiguous_route_issues = _dedupe_grounding_issues(
        issue
        for issue in route_issues
        if issue.kind == GroundingTerminalKind.AMBIGUOUS_REFERENCE
    )
    if ambiguous_route_issues:
        return CanonicalInputLedger(issues=ambiguous_route_issues)
    if not values_by_identity:
        return CanonicalInputLedger(
            issues=(
                tuple(route_issues)
                if route_issues
                else (_unsupported_reference_issue(task),)
            )
        )
    if len(values_by_identity) > 1:
        return CanonicalInputLedger(
            issues=(
                _ambiguous_reference_issue(
                    task,
                    values=tuple(values_by_identity.values()),
                    message="resolver returned multiple canonical identity matches",
                ),
            )
        )
    return CanonicalInputLedger(
        values=tuple(values_by_identity.values()),
        uses=tuple(uses_by_identity.values()),
    )


def _execute_compatible_reference_options_by_resolver_priority(
    *,
    task: KnownInputBindingTask,
    compatible_option_ids: tuple[str, ...],
    options: dict[str, tuple[KnownInputBindingTask, InputBindingOption]],
    resolver_selections: tuple[EntityTargetResolverSelection, ...],
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: Any,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> CanonicalInputLedger:
    route_options = _compatible_route_options(
        task=task,
        compatible_option_ids=compatible_option_ids,
        options=options,
    )
    issues: list[GroundingIssue] = []
    ambiguous_candidates: dict[str, GroundingCandidate] = {}
    ambiguity_seen = False
    for priority_group in _resolver_priority_groups(
        task=task,
        route_options=route_options,
        resolver_selections=resolver_selections,
    ):
        ledger = _execute_compatible_reference_options(
            task=task,
            compatible_option_ids=tuple(option.id for option in priority_group),
            options=options,
            full_catalog=full_catalog,
            resolver_row_sources=resolver_row_sources,
            data_access_port=data_access_port,
            source_read_lineage=source_read_lineage,
        )
        if ledger.issues:
            issues.extend(ledger.issues)
            if any(
                issue.kind == GroundingTerminalKind.AMBIGUOUS_REFERENCE
                for issue in ledger.issues
            ):
                ambiguity_seen = True
                _collect_issue_candidates(ambiguous_candidates, ledger.issues)
            continue
        if len(ledger.values) == 1:
            if ambiguity_seen:
                _collect_value_candidates(ambiguous_candidates, ledger.values)
                continue
            return ledger
        if len(ledger.values) > 1:
            ambiguity_seen = True
            _collect_value_candidates(ambiguous_candidates, ledger.values)
    if ambiguity_seen:
        return CanonicalInputLedger(
            issues=(
                _ambiguous_reference_issue(
                    task,
                    values=(),
                    candidate_options=tuple(ambiguous_candidates.values()),
                    message="resolver returned multiple canonical identity matches",
                ),
            )
        )
    return CanonicalInputLedger(
        issues=tuple(issues) if issues else (_unsupported_reference_issue(task),)
    )


def _collect_issue_candidates(
    output: dict[str, GroundingCandidate],
    issues: tuple[GroundingIssue, ...],
) -> None:
    for issue in issues:
        for option in issue.candidate_options:
            output.setdefault(option.id, option)


def _collect_value_candidates(
    output: dict[str, GroundingCandidate],
    values: tuple[FactValue, ...],
) -> None:
    for value in values:
        candidate = _identity_candidate(value)
        output.setdefault(candidate.id, candidate)


def _compatible_route_options(
    *,
    task: KnownInputBindingTask,
    compatible_option_ids: tuple[str, ...],
    options: dict[str, tuple[KnownInputBindingTask, InputBindingOption]],
) -> tuple[InputBindingOption, ...]:
    output: list[InputBindingOption] = []
    for option_id in compatible_option_ids:
        option_task, option = options[option_id]
        if option_task.known_input_id != task.known_input_id:
            raise ValueError("compatible grounding option references wrong input")
        output.append(option)
    return tuple(output)


def _resolver_priority_groups(
    *,
    task: KnownInputBindingTask,
    route_options: tuple[InputBindingOption, ...],
    resolver_selections: tuple[EntityTargetResolverSelection, ...],
) -> tuple[tuple[InputBindingOption, ...], ...]:
    selected_read_ids = _selected_resolver_read_ids(
        task=task,
        resolver_selections=resolver_selections,
    )
    if not selected_read_ids:
        return (route_options,)
    grouped: list[tuple[InputBindingOption, ...]] = []
    seen: set[str] = set()
    for read_id in selected_read_ids:
        group = tuple(
            option
            for option in route_options
            if option.route is not None and option.route.resolver_read_id == read_id
        )
        if group:
            grouped.append(group)
            seen.add(read_id)
    remainder = tuple(
        option
        for option in route_options
        if option.route is None or option.route.resolver_read_id not in seen
    )
    if remainder:
        grouped.append(remainder)
    return tuple(grouped)


def _selected_resolver_read_ids(
    *,
    task: KnownInputBindingTask,
    resolver_selections: tuple[EntityTargetResolverSelection, ...],
) -> tuple[str, ...]:
    for selection in resolver_selections:
        if selection.target_id == task.known_input_id:
            return selection.selected_read_ids
    return ()


def _identity_value_key(value: FactValue) -> tuple[str, str, str] | None:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return None
    return (payload.identity_type, payload.identity_field, str(payload.value))


def _unsupported_reference_issue(
    task: KnownInputBindingTask,
    *,
    message: str = "known input was not resolved to a canonical identity",
) -> GroundingIssue:
    return GroundingIssue(
        kind=GroundingTerminalKind.UNSUPPORTED_REFERENCE,
        known_input_id=task.known_input_id,
        requested_fact_id=task.requested_fact_id,
        message=message,
        known_input_text=task.lookup_text or task.known_input_text,
        known_input_description=task.known_input_description,
        proof_refs=(f"known_input:{task.known_input_id}",),
    )


def _dedupe_grounding_issues(issues: Any) -> tuple[GroundingIssue, ...]:
    output: list[GroundingIssue] = []
    seen: set[tuple[object, ...]] = set()
    for issue in issues:
        key = (
            issue.kind,
            issue.known_input_id,
            issue.requested_fact_id,
            issue.message,
            issue.candidates,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(issue)
    return tuple(output)


def _ambiguous_reference_issue(
    task: KnownInputBindingTask,
    *,
    values: tuple[FactValue, ...],
    message: str,
    candidate_options: tuple[GroundingCandidate, ...] = (),
) -> GroundingIssue:
    options = candidate_options or tuple(_identity_candidate(value) for value in values)
    return GroundingIssue(
        kind=GroundingTerminalKind.AMBIGUOUS_REFERENCE,
        known_input_id=task.known_input_id,
        requested_fact_id=task.requested_fact_id,
        message=message,
        known_input_text=task.lookup_text or task.known_input_text,
        known_input_description=task.known_input_description,
        candidates=tuple(option.id for option in options),
        candidate_options=options,
        proof_refs=(f"known_input:{task.known_input_id}",),
    )


def _identity_candidate(value: FactValue) -> GroundingCandidate:
    return GroundingCandidate(
        id=_identity_candidate_ref(value),
        label=_identity_candidate_label(value),
    )


def _identity_candidate_ref(value: FactValue) -> str:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return value.id
    return f"{payload.identity_type}:{payload.identity_field}:{payload.value}"


def _identity_candidate_label(value: FactValue) -> str:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return value.label or value.id
    identity_label = payload.identity_type.replace("_", " ").title()
    display = payload.display_value or value.label or str(payload.value)
    return f"{identity_label}: {display} [{payload.identity_field}={payload.value}]"


def _execute_reference_route(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    source_read_key: str,
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: Any,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> CanonicalInputLedger:
    resolver_source = resolver_row_sources.source(route.resolver_row_source_id)
    resolver_read = full_catalog.read(route.resolver_read_id)
    rows = _resolved_lookup_rows(
        task=task,
        route=route,
        resolver_read=resolver_read,
        resolver_source=resolver_source,
        data_access_port=data_access_port,
        source_read_lineage=source_read_lineage,
        source_read_key=source_read_key,
    )
    if not rows:
        return CanonicalInputLedger(
            issues=(
                GroundingIssue(
                    kind=GroundingTerminalKind.UNRESOLVED_REFERENCE,
                    known_input_id=task.known_input_id,
                    requested_fact_id=task.requested_fact_id,
                    message="resolver returned no canonical identity match",
                    known_input_text=task.lookup_text or task.known_input_text,
                    known_input_description=task.known_input_description,
                    proof_refs=(f"known_input:{task.known_input_id}",),
                ),
            )
        )
    values: list[FactValue] = []
    uses: list[GroundedInputUse] = []
    values_by_identity: dict[tuple[str, str, str], FactValue] = {}
    uses_by_identity: dict[tuple[str, str, str], GroundedInputUse] = {}
    for resolved in rows:
        fact_value = _fact_value_from_resolved_row(
            task=task,
            route=route,
            resolver_source=resolver_source,
            resolved=resolved,
        )
        if fact_value is None:
            continue
        values.append(fact_value)
        use = _grounded_input_use(task=task, route=route, value=fact_value)
        uses.append(use)
        key = _identity_value_key(fact_value)
        if key is not None:
            values_by_identity.setdefault(key, fact_value)
            uses_by_identity.setdefault(key, use)
    if not values:
        return CanonicalInputLedger(
            issues=(
                GroundingIssue(
                    kind=GroundingTerminalKind.UNRESOLVED_REFERENCE,
                    known_input_id=task.known_input_id,
                    requested_fact_id=task.requested_fact_id,
                    message="resolver row did not include canonical identity",
                    known_input_text=task.lookup_text or task.known_input_text,
                    known_input_description=task.known_input_description,
                    proof_refs=(f"known_input:{task.known_input_id}",),
                ),
            )
        )
    if len(values_by_identity) > 1:
        return CanonicalInputLedger(
            issues=(
                _ambiguous_reference_issue(
                    task,
                    values=tuple(values_by_identity.values()),
                    message="resolver returned multiple canonical identity matches",
                ),
            )
        )
    if values_by_identity:
        return CanonicalInputLedger(
            values=tuple(values_by_identity.values()),
            uses=tuple(uses_by_identity.values()),
        )
    return CanonicalInputLedger(values=tuple(values), uses=tuple(uses))


def _grounded_input_use(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    value: FactValue,
) -> GroundedInputUse:
    return GroundedInputUse(
        id=f"use_{_symbol(task.known_input_id)}_{_symbol(value.id)}",
        value_id=value.id,
        row_source_id=route.resolver_row_source_id,
        param_id=route.return_field_id,
        field_id=route.return_field_id,
    )


def _fact_value_from_resolved_row(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    resolver_source: RowSource,
    resolved: _ResolvedLookupRow,
) -> FactValue | None:
    row = resolved.row
    value = path_value(
        row,
        relative_response_path(
            resolver_source.field(route.return_field_id).path,
            resolver_source.row_path,
        ),
        missing=_MISSING,
    )
    if value in (_MISSING, None, ""):
        return None
    value_id = _grounded_identity_value_id(
        known_input_id=task.known_input_id,
        identity_type=route.identity_type,
        identity_field=route.identity_field,
        value=str(value),
    )
    return FactValue.identity(
        id=value_id,
        identity_type=route.identity_type,
        identity_field=route.identity_field,
        value=str(value),
        display_value=_display_value(row, resolver_source=resolver_source)
        or task.known_input_text,
        matched_field_ref=resolved.matched_field_ref,
        matched_field_path=resolved.matched_field_path,
        proof_refs=(f"known_input:{task.known_input_id}",),
        source_refs=(route.resolver_read_id, route.resolver_endpoint_name),
        applies_to_requested_fact_ids=_task_requested_fact_ids(task),
    )


def _grounded_identity_value_id(
    *,
    known_input_id: str,
    identity_type: str,
    identity_field: str,
    value: str,
) -> str:
    return _grounded_value_id(
        "_".join(
            (
                _symbol(known_input_id),
                _symbol(identity_type),
                _symbol(identity_field),
                _symbol(value),
            )
        )
    )


def _task_requested_fact_ids(task: KnownInputBindingTask) -> tuple[str, ...]:
    if task.applies_to_requested_fact_ids:
        return task.applies_to_requested_fact_ids
    if task.requested_fact_id:
        return (task.requested_fact_id,)
    return ()


def _resolved_lookup_rows(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    resolver_read: Any,
    resolver_source: RowSource,
    data_access_port: Any,
    source_read_lineage: SourceReadLineageScope | None,
    source_read_key: str,
) -> tuple[_ResolvedLookupRow, ...]:
    output: list[_ResolvedLookupRow] = []
    seen_rows: set[str] = set()
    lookup_text = task.lookup_text or task.known_input_text
    args = {route.lookup_param_ref: lookup_text} if route.lookup_param_ref else {}
    require_catalog_endpoint_for_lineage(
        source_read_lineage=source_read_lineage,
        endpoint_name=resolver_read.endpoint_name,
        catalog_endpoint=resolver_read.catalog_endpoint,
    )
    try:
        body = data_access_port.read(
            endpoint_name=resolver_read.endpoint_name,
            args=args,
        )
    except Exception as exc:
        record_source_read_error(
            source_read_lineage,
            source_read_key=source_read_key,
            endpoint_name=resolver_read.endpoint_name,
            catalog_endpoint=resolver_read.catalog_endpoint,
            args=args,
            error_json={"error": str(exc), "errorType": type(exc).__name__},
        )
        raise SourceReadFailedError(
            endpoint_name=resolver_read.endpoint_name,
            error_json={"error": str(exc), "errorType": type(exc).__name__},
        ) from exc
    observation = observe_source_read_response(
        body,
        endpoint_name=resolver_read.endpoint_name,
    )
    record_source_read_observation(
        source_read_lineage,
        source_read_key=source_read_key,
        endpoint_name=resolver_read.endpoint_name,
        catalog_endpoint=resolver_read.catalog_endpoint,
        args=args,
        observation=observation,
        completeness_json=source_read_completeness(body),
    )
    if not observation.succeeded:
        raise SourceReadFailedError(
            endpoint_name=resolver_read.endpoint_name,
            error_json=observation.error_json,
        )
    try:
        rows = extract_source_read_rows(
            body,
            endpoint_name=resolver_read.endpoint_name,
            row_source=resolver_source,
        )
    except EndpointResponseError as exc:
        raise SourceReadFailedError(
            endpoint_name=resolver_read.endpoint_name,
            error_json={"error": str(exc), "errorType": type(exc).__name__},
        ) from exc
    for resolved in _exact_lookup_rows(
        rows,
        route=route,
        resolver_source=resolver_source,
        text=lookup_text,
    ):
        key = repr(sorted(resolved.row.items()))
        if key in seen_rows:
            continue
        seen_rows.add(key)
        output.append(resolved)
    return tuple(output)


def _exact_lookup_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    route: InputBindingRoute,
    resolver_source: RowSource,
    text: str,
) -> tuple[_ResolvedLookupRow, ...]:
    if not route.lookup_field_ids:
        return tuple(_ResolvedLookupRow(row=row) for row in rows)
    lookup_fields = _lookup_fields_for_route(route, resolver_source=resolver_source)
    expected = _normalize_lookup_text(text)
    output: list[_ResolvedLookupRow] = []
    for row in rows:
        matches = _lookup_field_matches(row, lookup_fields, expected=expected)
        if len(matches) != 1:
            continue
        matched = matches[0]
        output.append(
            _ResolvedLookupRow(
                row=row,
                matched_field_ref=matched.ref,
                matched_field_path=matched.path,
            )
        )
    return tuple(output)


def _lookup_fields_for_route(
    route: InputBindingRoute,
    *,
    resolver_source: RowSource,
) -> tuple[_LookupField, ...]:
    return tuple(
        _LookupField(
            ref=field.field_ref,
            path=field.path,
            relative_path=relative_response_path(field.path, resolver_source.row_path),
        )
        for field_id in route.lookup_field_ids
        for field in (resolver_source.field(field_id),)
    )


def _lookup_field_matches(
    row: dict[str, Any],
    lookup_fields: tuple[_LookupField, ...],
    *,
    expected: str,
) -> tuple[_LookupField, ...]:
    return tuple(
        field
        for field in lookup_fields
        if _normalize_lookup_text(path_value(row, field.relative_path, missing=""))
        == expected
    )


def _display_value(row: dict[str, Any], *, resolver_source: RowSource) -> str:
    display_fields = tuple(
        display_field
        for field in resolver_source.fields
        if field.identity is not None
        for display_field in field.identity.display_fields
    )
    parts: list[str] = []
    for display_field in display_fields:
        try:
            field = resolver_source.field(display_field)
        except KeyError:
            continue
        value = path_value(
            row,
            relative_response_path(field.path, resolver_source.row_path),
            missing="",
        )
        if value not in ("", None):
            parts.append(str(value))
    return " ".join(parts).strip()
