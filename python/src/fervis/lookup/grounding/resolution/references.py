"""Catalog-backed named-reference grounding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.relation_catalog import (
    EndpointRead,
    RelationCatalog,
    RelationDataAccessPort,
)
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogParameterValueError,
    parse_catalog_parameter_value,
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
    GroundedValueCertification,
    GroundedValueCertificationMethod,
    GroundedInputUse,
    GroundingCandidate,
    GroundingIssue,
    GroundingTerminalKind,
    GroundingRequestedFactCard,
    ExpectedInputIdentity,
    InputBindingOption,
    InputBindingPurpose,
    InputBindingResultKind,
    InputBindingRoute,
    InputBindingSelection,
    KnownInputBindingTask,
    ResolverOutputFieldCard,
    ResolverQueryParamCard,
    ResolvedInputBinding,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCandidateKey,
    RowSourceCatalog,
    RowSourceField,
    RowSourceKind,
    RowSourceParam,
    RowSourceParamSemantics,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentityValuePayload,
    NamedValuePayload,
    ValueFilterOperator,
)
from fervis.lookup.canonical_data import RuntimeScalar, RuntimeValue
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFact,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
)

from .values import _grounded_value_id, _normalize_lookup_text, _symbol

_MISSING = object()


@dataclass(frozen=True)
class _ResolvedReferenceTasks:
    values: tuple[FactValue, ...] = ()
    certifications: tuple[GroundedValueCertification, ...] = ()
    issues: tuple[GroundingIssue, ...] = ()
    model_tasks: tuple[KnownInputBindingTask, ...] = ()


@dataclass(frozen=True)
class _ResolvedLookupRow:
    row: dict[str, Any]
    matched_field_id: str = ""
    matched_field_ref: str = ""
    matched_field_path: str = ""
    matched_value: RuntimeScalar = None
    matched_collection_member: bool = False


@dataclass(frozen=True)
class _LookupField:
    id: str
    ref: str
    path: str
    relative_path: str


@dataclass(frozen=True)
class _LookupMatch:
    field: _LookupField
    value: RuntimeScalar
    collection_member: bool = False


def reference_input_binding_tasks(
    question_contract: QuestionContract,
    *,
    resolver_row_sources: RowSourceCatalog,
    expected_input_identities: Mapping[str, ExpectedInputIdentity] | None = None,
) -> tuple[KnownInputBindingTask, ...]:
    tasks: list[KnownInputBindingTask] = []
    expected_identities = expected_input_identities or {}
    facts_by_id = {fact.id: fact for fact in question_contract.requested_facts}
    for known, requested_fact_ids in _reference_known_input_bindings(question_contract):
        if (
            not isinstance(known, RequestedFactLiteralInput)
            or not known.is_reference_value
        ):
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
            expected_identity=expected_identities.get(known.id),
        )
        if not options:
            tasks.append(
                KnownInputBindingTask(
                    known_input_id=known.id,
                    known_input_text=known.text,
                    known_input_kind=known.kind.value,
                    requested_fact_id=requested_fact_id,
                    options=(),
                    field_label_text=known.field_label_text,
                    known_input_description=known.value_meaning_hint,
                    lookup_text=known.resolved_value_text,
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
            field_label_text=known.field_label_text,
            known_input_description=known.value_meaning_hint,
            lookup_text=known.resolved_value_text,
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
    known: RequestedFactLiteralInput,
    *,
    resolver_row_sources: RowSourceCatalog,
    expected_identity: ExpectedInputIdentity | None,
) -> tuple[InputBindingOption, ...]:
    route_options: list[InputBindingOption] = []
    resolvers = tuple(
        source
        for source in resolver_row_sources.sources
        if source.kind == RowSourceKind.API_READ
    )
    option_index = 1
    for resolver_source in resolvers:
        for resolver_return in _resolver_return_fields(resolver_source):
            return_field = resolver_return.field
            for lookup_param, lookup_fields in _resolver_lookup_groups(
                resolver_source,
                resolver_return=resolver_return,
            ):
                route = InputBindingRoute(
                    known_input_id=known.id,
                    resolver_row_source_id=resolver_source.id,
                    resolver_read_id=resolver_source.read_id,
                    resolver_endpoint_name=resolver_source.label,
                    resolver_description=resolver_source.description,
                    resolver_resource_names=resolver_source.resource_names,
                    lookup_param_id=lookup_param.id if lookup_param is not None else "",
                    lookup_param_ref=(
                        lookup_param.param_ref if lookup_param is not None else ""
                    ),
                    lookup_param_type=(
                        lookup_param.type.value if lookup_param is not None else "string"
                    ),
                    lookup_field_ids=tuple(field.id for field in lookup_fields),
                    lookup_field_refs=(
                        tuple(field.field_ref for field in lookup_fields)
                    ),
                    return_field_id=return_field.id,
                    return_field_ref=return_field.field_ref,
                    entity_kind=resolver_return.entity_kind,
                    key_id=resolver_return.key_id,
                    key_component_id=resolver_return.component_id,
                    context_field_ids=resolver_return.context_field_ids,
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
                if not _route_matches_expected_identity(
                    route,
                    expected=expected_identity,
                ):
                    continue
                route_options.append(
                    InputBindingOption(
                        id=f"bind_{_symbol(known.id)}_{option_index}",
                        known_input_id=known.id,
                        path=route.display,
                        purpose=_binding_purpose(
                            resolver_source,
                            lookup_param=lookup_param,
                            resolver_return=resolver_return,
                        ),
                        route=route,
                    )
                )
                option_index += 1
    return tuple(route_options)


def _route_matches_expected_identity(
    route: InputBindingRoute,
    *,
    expected: ExpectedInputIdentity | None,
) -> bool:
    if expected is None:
        return True
    return (
        route.entity_kind == expected.entity_kind
        and route.key_id == expected.key_id
        and route.key_component_id == expected.key_component_id
    )


def reference_binding_row_sources(
    *,
    full_row_sources: RowSourceCatalog,
    resolver_row_sources: RowSourceCatalog,
) -> RowSourceCatalog:
    resolver_entity_kinds = {
        key.entity_kind
        for source in resolver_row_sources.sources
        for key in source.candidate_keys
        if key.primary and key.stable
    }
    validation_sources = tuple(
        source
        for source in full_row_sources.sources
        if _source_validates_one_of(source, entity_kinds=resolver_entity_kinds)
    )
    sources: list[RowSource] = []
    source_ids: set[str] = set()
    for source in (*resolver_row_sources.sources, *validation_sources):
        if source.id in source_ids:
            continue
        source_ids.add(source.id)
        sources.append(source)
    return RowSourceCatalog(sources=tuple(sources))


def _source_validates_one_of(
    source: RowSource,
    *,
    entity_kinds: set[str],
) -> bool:
    return any(
        str(param.source) == "path"
        and param.entity_target is not None
        and param.entity_target.entity_kind in entity_kinds
        and any(
            key.entity_kind == param.entity_target.entity_kind
            and key.id == param.entity_target.key_id
            and any(
                component.id == param.entity_target.component_id
                for component in key.components
            )
            for key in source.candidate_keys
        )
        for param in source.params
    )


def _binding_purpose(
    source: RowSource,
    *,
    lookup_param: RowSourceParam | None,
    resolver_return: _ResolverReturnField,
) -> InputBindingPurpose:
    if (
        lookup_param is not None
        and str(lookup_param.source) == "path"
        and _param_targets_resolver_key(lookup_param, resolver_return)
    ):
        return InputBindingPurpose.IDENTITY_VALIDATION
    return InputBindingPurpose.REFERENCE_GROUNDING


def _resolve_reference_tasks(
    tasks: tuple[KnownInputBindingTask, ...],
    *,
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> _ResolvedReferenceTasks:
    values: list[FactValue] = []
    issues: list[GroundingIssue] = []
    certifications: list[GroundedValueCertification] = []
    model_tasks: list[KnownInputBindingTask] = []
    for task in tasks:
        route_options = tuple(
            option for option in task.options if option.route is not None
        )
        if not route_options:
            issues.append(
                GroundingIssue(
                    kind=GroundingTerminalKind.UNSUPPORTED_REFERENCE,
                    known_input_id=task.known_input_id,
                    requested_fact_id=task.requested_fact_id,
                    message="no catalog binding route was available for known input",
                    known_input_text=task.lookup_text,
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
                known_input_text=task.lookup_text,
                known_input_description=task.known_input_description,
                proof_refs=(f"known_input:{task.known_input_id}",),
            )
        )
    return _ResolvedReferenceTasks(
        values=tuple(values),
        certifications=tuple(certifications),
        issues=tuple(issues),
        model_tasks=tuple(model_tasks),
    )


@dataclass(frozen=True)
class _ResolverReturnField:
    field: RowSourceField
    entity_kind: str
    key_id: str
    component_id: str
    context_field_ids: tuple[str, ...]


def _resolver_return_fields(source: RowSource) -> tuple[_ResolverReturnField, ...]:
    return tuple(
        _ResolverReturnField(
            field=source.field(key.components[0].field_id),
            entity_kind=key.entity_kind,
            key_id=key.id,
            component_id=key.components[0].id,
            context_field_ids=_resolver_identity_context_field_ids(source, key=key),
        )
        for key in source.candidate_keys
        if key.primary
        and key.stable
        and len(key.components) == 1
    )


def _resolver_identity_context_field_ids(
    source: RowSource,
    *,
    key: RowSourceCandidateKey,
) -> tuple[str, ...]:
    alternate_key_field_ids = (
        candidate.components[0].field_id
        for candidate in source.candidate_keys
        if candidate.entity_kind == key.entity_kind
        and candidate.id != key.id
        and candidate.stable
        and len(candidate.components) == 1
    )
    return tuple(dict.fromkeys((*key.context_field_ids, *alternate_key_field_ids)))


def _resolver_lookup_groups(
    source: RowSource,
    *,
    resolver_return: _ResolverReturnField,
) -> tuple[tuple[RowSourceParam | None, tuple[RowSourceField, ...]], ...]:
    groups: list[tuple[RowSourceParam | None, tuple[RowSourceField, ...]]] = list(
        _identity_param_lookup_groups(
            source,
            resolver_return=resolver_return,
        )
    )
    if any(param.required and param.default is None for param in source.params):
        return tuple(groups)
    text_params = tuple(
        param
        for param in source.params
        if (
            not param.required
            and param.accepts_lookup_text
            and param.semantics != RowSourceParamSemantics.RESPONSE_SHAPE
        )
    )
    text_fields = _resolver_text_lookup_fields(source)
    groups.extend((param, text_fields) for param in text_params)
    if text_fields:
        groups.append((None, text_fields))
    return tuple(groups)


def _identity_param_lookup_groups(
    source: RowSource,
    *,
    resolver_return: _ResolverReturnField,
) -> tuple[tuple[RowSourceParam, tuple[RowSourceField, ...]], ...]:
    return tuple(
        (param, (resolver_return.field,))
        for param in source.params
        if _param_targets_resolver_key(param, resolver_return)
        and param.semantics != RowSourceParamSemantics.RESPONSE_SHAPE
    )


def _param_targets_resolver_key(
    param: RowSourceParam,
    resolver_return: _ResolverReturnField,
) -> bool:
    target = param.entity_target
    return (
        target is not None
        and target.entity_kind == resolver_return.entity_kind
        and target.key_id == resolver_return.key_id
        and target.component_id == resolver_return.component_id
    )


def _resolver_text_lookup_fields(
    source: RowSource,
) -> tuple[RowSourceField, ...]:
    return tuple(field for field in source.fields if field.can_carry_lookup_text)


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
                type=param.type.value,
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
                type=field.type.value,
                choices=field.choices,
            )
        )
    return tuple(cards)


def _execute_reference_selections(
    *,
    selections: tuple[InputBindingSelection, ...],
    tasks: tuple[KnownInputBindingTask, ...],
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> CanonicalInputLedger:
    selections_by_input = {
        selection.known_input_id: selection for selection in selections
    }
    values: list[FactValue] = []
    uses: list[GroundedInputUse] = []
    issues: list[GroundingIssue] = []
    certifications: list[GroundedValueCertification] = []
    for task in tasks:
        selection = selections_by_input.get(task.known_input_id)
        if selection is None:
            issues.append(_unsupported_reference_issue(task))
            continue
        binding = selection.binding
        if binding is None:
            issues.append(
                _unsupported_reference_issue(
                    task,
                    message="no shown resolver could resolve lookup text",
                )
            )
            continue
        task_ledger = _execute_selected_reference_option(
            task=task,
            binding=binding,
            full_catalog=full_catalog,
            resolver_row_sources=resolver_row_sources,
            data_access_port=data_access_port,
            source_read_lineage=source_read_lineage,
        )
        values.extend(task_ledger.values)
        uses.extend(task_ledger.uses)
        issues.extend(task_ledger.issues)
        certifications.extend(task_ledger.certifications)
    return CanonicalInputLedger(
        values=tuple(values),
        uses=tuple(uses),
        issues=tuple(issues),
        certifications=tuple(certifications),
    )


def _execute_selected_reference_option(
    *,
    task: KnownInputBindingTask,
    binding: ResolvedInputBinding,
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> CanonicalInputLedger:
    options = {option.id: option for option in task.options}
    option = options.get(binding.option_id)
    if option is None:
        raise ValueError("selected grounding option references wrong input")
    if option.route is None:
        return CanonicalInputLedger(issues=(_unsupported_reference_issue(task),))
    certification_method = (
        GroundedValueCertificationMethod.IDENTITY_VALIDATION_READ
        if option.purpose is InputBindingPurpose.IDENTITY_VALIDATION
        else GroundedValueCertificationMethod.RESOLVER_SOURCE_READ
    )
    return _execute_reference_route(
        task=task,
        route=option.route,
        input_value=binding.input_value,
        result_kind=binding.result_kind,
        matched_field_ref=binding.matched_field_ref,
        source_read_key=binding.option_id,
        full_catalog=full_catalog,
        resolver_row_sources=resolver_row_sources,
        data_access_port=data_access_port,
        source_read_lineage=source_read_lineage,
        certification_method=certification_method,
    )


def _grounded_value_key(value: FactValue) -> tuple[str, ...]:
    payload = value.payload
    if isinstance(payload, IdentityValuePayload):
        return (
            "identity",
            payload.entity_kind,
            payload.key_id,
            payload.key_component_id,
            str(payload.value),
        )
    if isinstance(payload, NamedValuePayload):
        return ("named", payload.matched_field_ref, payload.text)
    return (value.kind.value, value.id)


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
        known_input_text=task.lookup_text,
        known_input_description=task.known_input_description,
        proof_refs=(f"known_input:{task.known_input_id}",),
    )


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
        known_input_text=task.lookup_text,
        known_input_description=task.known_input_description,
        candidates=tuple(option.id for option in options),
        candidate_options=options,
        proof_refs=(f"known_input:{task.known_input_id}",),
    )


def _identity_candidate(value: FactValue) -> GroundingCandidate:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return GroundingCandidate(
            id=_identity_candidate_ref(value),
            label=_identity_candidate_label(value),
        )
    resolver_read_id = value.source_refs[0] if value.source_refs else ""
    resolver_endpoint_name = (
        value.source_refs[1] if len(value.source_refs) > 1 else resolver_read_id
    )
    return GroundingCandidate(
        id=_identity_candidate_ref(value),
        label=_identity_candidate_label(value),
        entity_kind=payload.entity_kind,
        key_id=payload.key_id,
        matched_label=payload.display_value or value.label or payload.value,
        matched_field=payload.key_component_id,
        matched_value=payload.value,
        resolver_read_id=resolver_read_id,
        resolver_label=_title_words(resolver_read_id or resolver_endpoint_name),
    )


def _identity_candidate_ref(value: FactValue) -> str:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return value.id
    return (
        f"{payload.entity_kind}:{payload.key_id}:"
        f"{payload.key_component_id}:{payload.value}"
    )


def _identity_candidate_label(value: FactValue) -> str:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return value.label or value.id
    identity_label = payload.entity_kind.replace("_", " ").title()
    display = payload.display_value or value.label or str(payload.value)
    return f"{identity_label}: {display} [{payload.key_component_id}={payload.value}]"


def _title_words(value: str) -> str:
    return " ".join(
        word.capitalize() for word in value.replace("_", " ").replace("-", " ").split()
    )


def _execute_reference_route(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    input_value: str | int | float | bool,
    result_kind: InputBindingResultKind,
    matched_field_ref: str = "",
    source_read_key: str,
    full_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_lineage: SourceReadLineageScope | None = None,
    certification_method: GroundedValueCertificationMethod = (
        GroundedValueCertificationMethod.RESOLVER_SOURCE_READ
    ),
) -> CanonicalInputLedger:
    resolver_source = resolver_row_sources.source(route.resolver_row_source_id)
    resolver_read = full_catalog.read(route.resolver_read_id)
    rows = _resolved_lookup_rows(
        task=task,
        route=route,
        input_value=input_value,
        result_kind=result_kind,
        matched_field_ref=matched_field_ref,
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
                    known_input_text=task.lookup_text,
                    known_input_description=task.known_input_description,
                    proof_refs=(f"known_input:{task.known_input_id}",),
                    resolver_read_id=route.resolver_read_id,
                    resolver_endpoint_name=route.resolver_endpoint_name,
                    resolver_field_id=route.return_field_id,
                    identity_field=route.key_component_id,
                ),
            )
        )
    values_by_key: dict[tuple[str, ...], FactValue] = {}
    uses_by_key: dict[tuple[str, ...], GroundedInputUse] = {}
    certifications_by_key: dict[tuple[str, ...], GroundedValueCertification] = {}
    for resolved in rows:
        fact_value = _fact_value_from_resolved_row(
            task=task,
            route=route,
            resolver_source=resolver_source,
            resolved=resolved,
            result_kind=result_kind,
        )
        if fact_value is None:
            continue
        use = _grounded_input_use(task=task, route=route, value=fact_value)
        certification = _resolver_source_read_certification(
            value=fact_value,
            route=route,
            method=certification_method,
        )
        key = _grounded_value_key(fact_value)
        values_by_key.setdefault(key, fact_value)
        uses_by_key.setdefault(key, use)
        certifications_by_key.setdefault(key, certification)
    if not values_by_key:
        return CanonicalInputLedger(
            issues=(
                GroundingIssue(
                    kind=GroundingTerminalKind.UNRESOLVED_REFERENCE,
                    known_input_id=task.known_input_id,
                    requested_fact_id=task.requested_fact_id,
                    message="resolver row did not include canonical identity",
                    known_input_text=task.lookup_text,
                    known_input_description=task.known_input_description,
                    proof_refs=(f"known_input:{task.known_input_id}",),
                    resolver_read_id=route.resolver_read_id,
                    resolver_endpoint_name=route.resolver_endpoint_name,
                    resolver_field_id=route.return_field_id,
                    identity_field=route.key_component_id,
                ),
            )
        )
    if len(values_by_key) > 1:
        return CanonicalInputLedger(
            issues=(
                _ambiguous_reference_issue(
                    task,
                    values=tuple(values_by_key.values()),
                    message="resolver returned multiple canonical matches",
                ),
            )
        )
    return CanonicalInputLedger(
        values=tuple(values_by_key.values()),
        uses=tuple(uses_by_key.values()),
        certifications=tuple(certifications_by_key.values()),
    )


def _resolver_source_read_certification(
    *,
    value: FactValue,
    route: InputBindingRoute,
    method: GroundedValueCertificationMethod,
) -> GroundedValueCertification:
    return GroundedValueCertification(
        value_id=value.id,
        method=method,
        authority_refs=(route.resolver_read_id,),
        lineage_refs=tuple(value.proof_refs),
    )


def _grounded_input_use(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    value: FactValue,
) -> GroundedInputUse:
    field_id = route.return_field_id
    if isinstance(value.payload, NamedValuePayload):
        field_id = next(
            (
                candidate_id
                for candidate_id, candidate_ref in zip(
                    route.lookup_field_ids,
                    route.lookup_field_refs,
                    strict=True,
                )
                if candidate_ref == value.payload.matched_field_ref
            ),
            "",
        )
    return GroundedInputUse(
        id=f"use_{_symbol(task.known_input_id)}_{_symbol(value.id)}",
        value_id=value.id,
        row_source_id=route.resolver_row_source_id,
        param_id=field_id,
        field_id=field_id,
        entity_kind=route.entity_kind,
    )


def _fact_value_from_resolved_row(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    resolver_source: RowSource,
    resolved: _ResolvedLookupRow,
    result_kind: InputBindingResultKind,
) -> FactValue | None:
    if result_kind is InputBindingResultKind.MATCHED_VALUE:
        return _named_value_from_resolved_match(task=task, route=route, resolved=resolved)
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
        entity_kind=route.entity_kind,
        key_id=route.key_id,
        key_component_id=route.key_component_id,
        value=str(value),
    )
    return FactValue.identity(
        id=value_id,
        known_input_id=task.known_input_id,
        entity_kind=route.entity_kind,
        key_id=route.key_id,
        key_component_id=route.key_component_id,
        value=str(value),
        display_value=_display_value(
            row,
            route=route,
            resolver_source=resolver_source,
        ),
        matched_field_ref=resolved.matched_field_ref,
        matched_field_path=resolved.matched_field_path,
        proof_refs=(f"known_input:{task.known_input_id}",),
        source_refs=(route.resolver_read_id, route.resolver_endpoint_name),
        applies_to_requested_fact_ids=_task_requested_fact_ids(task),
    )


def _named_value_from_resolved_match(
    *,
    task: KnownInputBindingTask,
    route: InputBindingRoute,
    resolved: _ResolvedLookupRow,
) -> FactValue | None:
    if resolved.matched_value is None:
        return None
    text = str(resolved.matched_value)
    value_id = _grounded_value_id(
        "_".join(
            (
                _symbol(task.known_input_id),
                _symbol(resolved.matched_field_ref),
                _symbol(text),
            )
        )
    )
    return FactValue.named(
        id=value_id,
        known_input_id=task.known_input_id,
        text=text,
        reference_text=task.lookup_text,
        matched_field_ref=resolved.matched_field_ref,
        matched_field_path=resolved.matched_field_path,
        filter_operator=(
            ValueFilterOperator.CONTAINS
            if resolved.matched_collection_member
            else ValueFilterOperator.EQUALS
        ),
        proof_refs=(f"known_input:{task.known_input_id}",),
        source_refs=(route.resolver_read_id, route.resolver_endpoint_name),
        applies_to_requested_fact_ids=_task_requested_fact_ids(task),
    )


def _grounded_identity_value_id(
    *,
    known_input_id: str,
    entity_kind: str,
    key_id: str,
    key_component_id: str,
    value: str,
) -> str:
    return _grounded_value_id(
        "_".join(
            (
                _symbol(known_input_id),
                _symbol(entity_kind),
                _symbol(key_id),
                _symbol(key_component_id),
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
    input_value: str | int | float | bool,
    result_kind: InputBindingResultKind,
    matched_field_ref: str,
    resolver_read: EndpointRead,
    resolver_source: RowSource,
    data_access_port: RelationDataAccessPort,
    source_read_lineage: SourceReadLineageScope | None,
    source_read_key: str,
) -> tuple[_ResolvedLookupRow, ...]:
    output: list[_ResolvedLookupRow] = []
    seen_rows: set[str] = set()
    args = _typed_lookup_args(
        input_value,
        route=route,
        resolver_read=resolver_read,
    )
    if args is None:
        return ()
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
        text=input_value,
        result_kind=result_kind,
        matched_field_ref=matched_field_ref,
    ):
        key = repr(sorted(resolved.row.items()))
        if key in seen_rows:
            continue
        seen_rows.add(key)
        output.append(resolved)
    return tuple(output)


def _typed_lookup_args(
    lookup_value: str | int | float | bool,
    *,
    route: InputBindingRoute,
    resolver_read: EndpointRead,
) -> dict[str, object] | None:
    if not route.lookup_param_ref:
        return {}
    lookup_param = next(
        (
            param
            for param in resolver_read.params
            if param.ref == route.lookup_param_ref
        ),
        None,
    )
    if lookup_param is None:
        raise ValueError("grounding route references unknown lookup parameter")
    try:
        value = parse_catalog_parameter_value(
            lookup_value,
            type_name=lookup_param.type,
            choices=lookup_param.choices,
        )
    except CatalogParameterValueError:
        return None
    return {route.lookup_param_ref: value}


def _exact_lookup_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    route: InputBindingRoute,
    resolver_source: RowSource,
    text: str | int | float | bool,
    result_kind: InputBindingResultKind,
    matched_field_ref: str = "",
) -> tuple[_ResolvedLookupRow, ...]:
    if not route.lookup_field_ids:
        return tuple(_ResolvedLookupRow(row=row) for row in rows)
    lookup_fields = _lookup_fields_for_route(route, resolver_source=resolver_source)
    if (
        result_kind is InputBindingResultKind.CANONICAL_IDENTITY
        and not route.lookup_param_ref
    ):
        identity_field_ids = set(route.identity_lookup_field_ids)
        lookup_fields = tuple(
            field for field in lookup_fields if field.id in identity_field_ids
        )
    elif result_kind is InputBindingResultKind.MATCHED_VALUE and matched_field_ref:
        lookup_fields = tuple(
            field for field in lookup_fields if field.ref == matched_field_ref
        )
        if not lookup_fields:
            raise ValueError("grounding selected an unknown lookup field")
    expected = _normalize_lookup_text(text)
    output: list[_ResolvedLookupRow] = []
    for row in rows:
        matches = _lookup_field_matches(row, lookup_fields, expected=expected)
        if not matches:
            continue
        matched = matches[0]
        output.append(
            _ResolvedLookupRow(
                row=row,
                matched_field_id=matched.field.id,
                matched_field_ref=matched.field.ref,
                matched_field_path=matched.field.path,
                matched_value=matched.value,
                matched_collection_member=matched.collection_member,
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
            id=field.id,
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
) -> tuple[_LookupMatch, ...]:
    matches: list[_LookupMatch] = []
    for field in lookup_fields:
        value = path_value(row, field.relative_path, missing="")
        matched, matched_value, collection_member = _matching_lookup_value(
            value,
            expected=expected,
        )
        if not matched:
            continue
        matches.append(
            _LookupMatch(
                field=field,
                value=matched_value,
                collection_member=collection_member,
            )
        )
    return tuple(matches)


def _matching_lookup_value(
    value: RuntimeValue,
    *,
    expected: str,
) -> tuple[bool, RuntimeScalar, bool]:
    if isinstance(value, list | tuple):
        matched = next(
            (
                item
                for item in value
                if not isinstance(item, list | tuple | Mapping)
                and _normalize_lookup_text(item) == expected
            ),
            None,
        )
        return matched is not None, matched, True
    if isinstance(value, Mapping):
        return False, None, False
    if _normalize_lookup_text(value) == expected:
        return True, value, False
    return False, None, False


def _display_value(
    row: dict[str, Any],
    *,
    route: InputBindingRoute,
    resolver_source: RowSource,
) -> str:
    parts: list[str] = []
    for context_field_id in route.context_field_ids:
        field = resolver_source.field(context_field_id)
        value = path_value(
            row,
            relative_response_path(field.path, resolver_source.row_path),
            missing="",
        )
        if value not in ("", None):
            parts.append(str(value))
    return " ".join(parts).strip()
