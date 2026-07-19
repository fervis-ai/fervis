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
    CatalogParameterValue,
    CatalogParameterValueError,
    CatalogScalarParameterValue,
    parse_catalog_parameter_value,
)
from fervis.lookup.relation_catalog.selection import EntityTargetResolverSelection
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
    GroundingCandidate,
    InputBindingCompatibility,
    GroundingIssue,
    GroundingTerminalKind,
    ExpectedInputIdentity,
    CompatibleInputBinding,
    InputBindingOption,
    InputBindingPurpose,
    InputBindingKeyComponent,
    KnownInputBindingTask,
    ResolverCandidate,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCandidateKey,
    RowSourceCatalog,
    RowSourceKind,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentityValuePayload,
)
from fervis.lookup.canonical_data import (
    EntityKeyComponentValue,
    EntityKeyValue,
    RuntimeScalar,
    RuntimeValue,
    canonical_runtime_json,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
)

from .values import _grounded_value_id, _symbol

_MISSING = object()


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
    type: str
    choices: tuple[str, ...] = ()


_ReferenceExecutionKey = tuple[str, str, tuple[str, ...]]


@dataclass(frozen=True)
class _LookupMatch:
    field: _LookupField
    value: RuntimeScalar
    collection_member: bool = False


@dataclass(frozen=True)
class _ResolvedLookupResult:
    rows: tuple[_ResolvedLookupRow, ...]
    truncated: bool = False


@dataclass(frozen=True)
class ReferenceOptionExecution:
    ledger: CanonicalInputLedger | None = None
    failure: SourceReadFailedError | None = None
    truncated: bool = False
    matched_field_is_stable_unique: bool = False

    def __post_init__(self) -> None:
        if (self.ledger is None) == (self.failure is None):
            raise ValueError(
                "reference option execution requires one ledger or read failure"
            )


def reference_input_binding_tasks(
    question_contract: QuestionContract,
    *,
    resolver_catalog: RelationCatalog,
    resolver_sources_by_known_input: dict[str, RowSourceCatalog],
    expected_input_identities: Mapping[str, ExpectedInputIdentity] | None = None,
) -> tuple[KnownInputBindingTask, ...]:
    tasks: list[KnownInputBindingTask] = []
    expected_identities = expected_input_identities or {}
    for known, requested_fact_ids in _reference_known_input_bindings(question_contract):
        if (
            not isinstance(known, RequestedFactLiteralInput)
            or not known.is_reference_value
        ):
            continue
        requested_fact_id = requested_fact_ids[0] if requested_fact_ids else ""
        options = _reference_binding_options(
            known,
            resolver_catalog=resolver_catalog,
            resolver_row_sources=resolver_sources_by_known_input.get(
                known.id,
                RowSourceCatalog(),
            ),
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
        )
        tasks.append(task)
    return tuple(tasks)


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
    resolver_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    expected_identity: ExpectedInputIdentity | None,
) -> tuple[InputBindingOption, ...]:
    options: list[InputBindingOption] = []
    resolvers = tuple(
        source
        for source in resolver_row_sources.sources
        if source.kind == RowSourceKind.API_READ
    )
    for resolver_source in resolvers:
        resolver_read = resolver_catalog.read(resolver_source.read_id)
        if resolver_read.endpoint_name != resolver_source.endpoint_name:
            raise ValueError("resolver row source disagrees with its catalog read")
        for resolver_return in _resolver_return_fields(resolver_source):
            candidate = ResolverCandidate(
                known_input_id=known.id,
                resolver_source=resolver_source,
                resolver_resource_names=resolver_source.resource_names,
                entity_kind=resolver_return.entity_kind,
                key_id=resolver_return.key_id,
                key_components=tuple(
                    InputBindingKeyComponent(
                        component_id=component.id,
                        field_id=component.field_id,
                        field_ref=resolver_source.field(component.field_id).field_ref,
                    )
                    for component in resolver_return.key.components
                ),
            )
            if (
                expected_identity is not None
                and not _candidate_matches_expected_identity(
                    candidate,
                    expected=expected_identity,
                )
            ):
                continue
            options.append(
                InputBindingOption(
                    id=(
                        f"bind_{_symbol(known.id)}_"
                        f"{_symbol(resolver_source.id)}_"
                        f"{_symbol(resolver_return.entity_kind)}_"
                        f"{_symbol(resolver_return.key_id)}"
                    ),
                    known_input_id=known.id,
                    candidate=candidate,
                )
            )
    return tuple(options)


def _candidate_matches_expected_identity(
    candidate: ResolverCandidate,
    *,
    expected: ExpectedInputIdentity | None,
) -> bool:
    if expected is None:
        return True
    return (
        candidate.entity_kind == expected.entity_kind
        and candidate.key_id == expected.key_id
        and tuple(component.component_id for component in candidate.key_components)
        == expected.key_component_ids
    )


def reference_binding_sources_by_known_input(
    *,
    full_row_sources: RowSourceCatalog,
    resolver_selections: tuple[EntityTargetResolverSelection, ...],
) -> dict[str, RowSourceCatalog]:
    return {
        selection.target_id: _reference_binding_sources(
            full_row_sources=full_row_sources,
            resolver_read_ids=selection.selected_read_ids,
        )
        for selection in resolver_selections
    }


def _reference_binding_sources(
    *,
    full_row_sources: RowSourceCatalog,
    resolver_read_ids: tuple[str, ...],
) -> RowSourceCatalog:
    resolver_read_id_set = set(resolver_read_ids)
    resolver_sources = tuple(
        source
        for source in full_row_sources.sources
        if source.read_id in resolver_read_id_set
    )
    resolver_entity_kinds = {
        key.entity_kind
        for source in resolver_sources
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
    for source in (*resolver_sources, *validation_sources):
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


@dataclass(frozen=True)
class _ResolverReturnField:
    key: RowSourceCandidateKey
    entity_kind: str
    key_id: str


def _resolver_return_fields(source: RowSource) -> tuple[_ResolverReturnField, ...]:
    return tuple(
        _ResolverReturnField(
            key=key,
            entity_kind=key.entity_kind,
            key_id=key.id,
        )
        for key in source.candidate_keys
        if key.primary and key.stable
    )


def _grounded_value_key(value: FactValue) -> tuple[str, ...]:
    payload = value.payload
    if isinstance(payload, IdentityValuePayload):
        return (
            "identity",
            payload.entity_kind,
            payload.key_id,
            canonical_runtime_json(payload.key.component_values()),
        )
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


def unresolved_reference_issues(
    tasks: tuple[KnownInputBindingTask, ...],
    *,
    compatibilities: tuple[InputBindingCompatibility, ...],
) -> tuple[GroundingIssue, ...]:
    """Return clarification needs for inputs with no usable canonical resolver."""

    compatible_bindings_by_input = {
        compatibility.known_input_id: compatibility.bindings
        for compatibility in compatibilities
    }
    return tuple(
        _unsupported_reference_issue(
            task,
            message="no shown resolver could resolve the known input",
        )
        for task in tasks
        if not compatible_bindings_by_input.get(task.known_input_id)
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
        key=payload.key,
        matched_label=payload.display_value or value.label,
        matched_field=payload.matched_field_ref,
        matched_value=(
            str(payload.matched_value) if payload.matched_value is not None else ""
        ),
        resolver_read_id=resolver_read_id,
        resolver_label=_title_words(resolver_read_id or resolver_endpoint_name),
    )


def _identity_candidate_ref(value: FactValue) -> str:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return value.id
    return f"{payload.entity_kind}:{payload.key_id}:{canonical_runtime_json(payload.key.component_values())}"


def _identity_candidate_label(value: FactValue) -> str:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return value.label or value.id
    identity_label = payload.entity_kind.replace("_", " ").title()
    display = (
        payload.display_value or value.label or str(payload.key.component_values())
    )
    components = ", ".join(
        f"{component.component_id}={component.value}"
        for component in payload.key.components
    )
    return f"{identity_label}: {display} [{components}]"


def _title_words(value: str) -> str:
    return " ".join(
        word.capitalize() for word in value.replace("_", " ").replace("-", " ").split()
    )


def _reference_ledger_from_rows(
    *,
    task: KnownInputBindingTask,
    candidate: ResolverCandidate,
    resolver_source: RowSource,
    rows: tuple[_ResolvedLookupRow, ...],
    certification_method: GroundedValueCertificationMethod,
) -> CanonicalInputLedger:
    values_by_key: dict[tuple[str, ...], FactValue] = {}
    certifications_by_key: dict[tuple[str, ...], GroundedValueCertification] = {}
    for resolved in rows:
        fact_values = _fact_values_from_resolved_row(
            task=task,
            candidate=candidate,
            resolver_source=resolver_source,
            resolved=resolved,
        )
        for fact_value in fact_values:
            certification = _resolver_source_read_certification(
                value=fact_value,
                candidate=candidate,
                method=certification_method,
            )
            key = _grounded_value_key(fact_value)
            values_by_key.setdefault(key, fact_value)
            certifications_by_key.setdefault(key, certification)
    return CanonicalInputLedger(
        values=tuple(values_by_key.values()),
        certifications=tuple(certifications_by_key.values()),
    )


def reference_binding_issue(
    task: KnownInputBindingTask,
    *,
    candidate: ResolverCandidate,
    values: tuple[FactValue, ...],
    truncated: bool = False,
    matched_field_is_stable_unique: bool = False,
) -> GroundingIssue | None:
    if truncated and len(values) <= 1 and not matched_field_is_stable_unique:
        return replace(
            _unsupported_reference_issue(
                task,
                message=(
                    "resolver response was incomplete and cannot prove absence or "
                    "uniqueness"
                ),
            ),
            kind=GroundingTerminalKind.INCOMPLETE_REFERENCE,
            resolver_read_id=candidate.resolver_read_id,
            resolver_endpoint_name=candidate.resolver_endpoint_name,
        )
    issue = reference_values_issue(
        task,
        values=values,
        no_match_message="resolver binding found no exact match for the known input",
        multiple_match_message=(
            "resolver binding found multiple exact matches for the known input"
        ),
    )
    if issue is None:
        return None
    return replace(
        issue,
        resolver_read_id=candidate.resolver_read_id,
        resolver_endpoint_name=candidate.resolver_endpoint_name,
    )


def reference_values_issue(
    task: KnownInputBindingTask,
    *,
    values: tuple[FactValue, ...],
    no_match_message: str,
    multiple_match_message: str,
) -> GroundingIssue | None:
    if not values:
        return _unsupported_reference_issue(task, message=no_match_message)
    if len(values) == 1:
        return None
    return _ambiguous_reference_issue(
        task,
        values=values,
        message=multiple_match_message,
        candidate_options=tuple(_identity_candidate(value) for value in values),
    )


def execute_compatible_reference_bindings(
    *,
    tasks: tuple[KnownInputBindingTask, ...],
    bindings: tuple[CompatibleInputBinding, ...],
    source_read_key_prefix: str,
    full_catalog: RelationCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> dict[str, ReferenceOptionExecution]:
    tasks_by_input_id = {task.known_input_id: task for task in tasks}
    options_by_id = {option.id: option for task in tasks for option in task.options}
    executions: dict[str, ReferenceOptionExecution] = {}
    results_by_execution: dict[
        _ReferenceExecutionKey,
        _ResolvedLookupResult | SourceReadFailedError,
    ] = {}
    for binding in bindings:
        option = options_by_id[binding.option_id]
        candidate = option.candidate
        task = tasks_by_input_id[option.known_input_id]
        resolver_source = candidate.resolver_source
        args = _resolver_request_args(resolver_source, binding=binding)
        execution_key = _reference_execution_key(
            resolver_source=resolver_source,
            args=args,
            binding=binding,
        )
        result = results_by_execution.get(execution_key)
        if result is None:
            catalog_read = full_catalog.read(candidate.resolver_read_id)
            try:
                result = _resolved_lookup_rows(
                    binding=binding,
                    args=args,
                    catalog_read=catalog_read,
                    resolver_source=resolver_source,
                    data_access_port=data_access_port,
                    source_read_lineage=source_read_lineage,
                    source_read_key=(
                        f"{source_read_key_prefix}:{task.known_input_id}:"
                        f"{_reference_invocation_address(execution_key)}"
                    ),
                )
            except SourceReadFailedError as exc:
                result = exc
            results_by_execution[execution_key] = result
        if isinstance(result, SourceReadFailedError):
            executions[binding.option_id] = ReferenceOptionExecution(failure=result)
            continue
        ledger = _reference_ledger_from_rows(
            task=task,
            candidate=candidate,
            resolver_source=resolver_source,
            rows=result.rows,
            certification_method=(
                GroundedValueCertificationMethod.IDENTITY_VALIDATION_READ
                if _binding_purpose(
                    binding,
                    candidate=candidate,
                )
                is InputBindingPurpose.IDENTITY_VALIDATION
                else GroundedValueCertificationMethod.RESOLVER_SOURCE_READ
            ),
        )
        executions[binding.option_id] = ReferenceOptionExecution(
            ledger=ledger,
            truncated=result.truncated,
            matched_field_is_stable_unique=_matched_fields_are_stable_unique(
                result.rows,
                candidate=candidate,
                resolver_source=resolver_source,
            ),
        )
    return executions


def _reference_execution_key(
    *,
    resolver_source: RowSource,
    args: Mapping[str, CatalogParameterValue],
    binding: CompatibleInputBinding,
) -> _ReferenceExecutionKey:
    return (
        resolver_source.id,
        canonical_runtime_json(dict(args)),
        binding.returned_identity_verification_field_paths,
    )


def _reference_invocation_address(key: _ReferenceExecutionKey) -> str:
    row_source_id, args_json, match_fields = key
    return canonical_runtime_json(
        {
            "row_source_id": row_source_id,
            "args": args_json,
            "response_match_fields": list(match_fields),
        }
    )


def _resolver_request_args(
    resolver_source: RowSource,
    *,
    binding: CompatibleInputBinding,
) -> dict[str, CatalogParameterValue]:
    args: dict[str, CatalogParameterValue] = {
        parameter.param_ref: parameter.default
        for parameter in resolver_source.params
        if parameter.default is not None
    }
    args.update(
        {item.param_ref: item.value for item in binding.lookup_request_parameters}
    )
    return args


def _resolver_source_read_certification(
    *,
    value: FactValue,
    candidate: ResolverCandidate,
    method: GroundedValueCertificationMethod,
) -> GroundedValueCertification:
    return GroundedValueCertification(
        value_id=value.id,
        method=method,
        authority_refs=(candidate.resolver_read_id,),
        lineage_refs=tuple(value.proof_refs),
    )


def _fact_values_from_resolved_row(
    *,
    task: KnownInputBindingTask,
    candidate: ResolverCandidate,
    resolver_source: RowSource,
    resolved: _ResolvedLookupRow,
) -> tuple[FactValue, ...]:
    values: list[FactValue] = []
    key = _entity_key_from_resolved_row(
        candidate=candidate,
        resolver_source=resolver_source,
        row=resolved.row,
    )
    if key is not None:
        values.append(
            FactValue.identity(
                id=_grounded_identity_value_id(
                    known_input_id=task.known_input_id,
                    key=key,
                ),
                known_input_id=task.known_input_id,
                key=key,
                display_value=str(resolved.matched_value or ""),
                matched_field_ref=resolved.matched_field_ref,
                matched_field_path=resolved.matched_field_path,
                matched_value=resolved.matched_value,
                proof_refs=(f"known_input:{task.known_input_id}",),
                source_refs=(
                    candidate.resolver_read_id,
                    candidate.resolver_endpoint_name,
                ),
                applies_to_requested_fact_ids=_task_requested_fact_ids(task),
            )
        )
    return tuple(values)


def _entity_key_from_resolved_row(
    *,
    candidate: ResolverCandidate,
    resolver_source: RowSource,
    row: Mapping[str, RuntimeValue],
) -> EntityKeyValue | None:
    components: list[EntityKeyComponentValue] = []
    for component in candidate.key_components:
        value = path_value(
            row,
            relative_response_path(
                resolver_source.field(component.field_id).path,
                resolver_source.row_path,
            ),
            missing=_MISSING,
        )
        if value in (_MISSING, None, ""):
            return None
        components.append(
            EntityKeyComponentValue(
                component_id=component.component_id,
                value=value,
            )
        )
    return EntityKeyValue(
        entity_kind=candidate.entity_kind,
        key_id=candidate.key_id,
        components=tuple(components),
    )


def _grounded_identity_value_id(
    *,
    known_input_id: str,
    key: EntityKeyValue,
) -> str:
    return _grounded_value_id(
        "_".join(
            (
                _symbol(known_input_id),
                _symbol(key.entity_kind),
                _symbol(key.key_id),
                _symbol(canonical_runtime_json(key.component_values())),
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
    binding: CompatibleInputBinding,
    args: dict[str, CatalogParameterValue],
    catalog_read: EndpointRead,
    resolver_source: RowSource,
    data_access_port: RelationDataAccessPort,
    source_read_lineage: SourceReadLineageScope | None,
    source_read_key: str,
) -> _ResolvedLookupResult:
    output: list[_ResolvedLookupRow] = []
    seen_rows: set[str] = set()
    require_catalog_endpoint_for_lineage(
        source_read_lineage=source_read_lineage,
        endpoint_name=resolver_source.endpoint_name,
        catalog_endpoint=catalog_read.catalog_endpoint,
    )
    try:
        body = data_access_port.read(
            endpoint_name=resolver_source.endpoint_name,
            args=args,
        )
    except Exception as exc:
        record_source_read_error(
            source_read_lineage,
            source_read_key=source_read_key,
            endpoint_name=resolver_source.endpoint_name,
            catalog_endpoint=catalog_read.catalog_endpoint,
            args=args,
            error_json={"error": str(exc), "errorType": type(exc).__name__},
        )
        raise SourceReadFailedError(
            endpoint_name=resolver_source.endpoint_name,
            error_json={"error": str(exc), "errorType": type(exc).__name__},
        ) from exc
    observation = observe_source_read_response(
        body,
        endpoint_name=resolver_source.endpoint_name,
    )
    completeness = source_read_completeness(body)
    record_source_read_observation(
        source_read_lineage,
        source_read_key=source_read_key,
        endpoint_name=resolver_source.endpoint_name,
        catalog_endpoint=catalog_read.catalog_endpoint,
        args=args,
        observation=observation,
        completeness_json=completeness,
    )
    if not observation.succeeded:
        raise SourceReadFailedError(
            endpoint_name=resolver_source.endpoint_name,
            error_json=observation.error_json,
        )
    try:
        rows = extract_source_read_rows(
            body,
            endpoint_name=resolver_source.endpoint_name,
            row_source=resolver_source,
        )
    except EndpointResponseError as exc:
        raise SourceReadFailedError(
            endpoint_name=resolver_source.endpoint_name,
            error_json={"error": str(exc), "errorType": type(exc).__name__},
        ) from exc
    for resolved in _exact_lookup_rows(
        rows,
        binding=binding,
        resolver_source=resolver_source,
    ):
        key = repr(sorted(resolved.row.items()))
        if key in seen_rows:
            continue
        seen_rows.add(key)
        output.append(resolved)
    return _ResolvedLookupResult(
        rows=tuple(output),
        truncated=bool(completeness["truncated"]),
    )


def _exact_lookup_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    binding: CompatibleInputBinding,
    resolver_source: RowSource,
) -> tuple[_ResolvedLookupRow, ...]:
    lookup_fields = _lookup_fields_for_binding(
        binding,
        resolver_source=resolver_source,
    )
    output: list[_ResolvedLookupRow] = []
    for row in rows:
        matches = _lookup_field_matches(
            row,
            lookup_fields,
            expected=binding.lookup_value,
        )
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


def _lookup_fields_for_binding(
    binding: CompatibleInputBinding,
    *,
    resolver_source: RowSource,
) -> tuple[_LookupField, ...]:
    selected_field_paths = set(binding.returned_identity_verification_field_paths)
    selected_fields = tuple(
        field for field in resolver_source.fields if field.path in selected_field_paths
    )
    if len(selected_fields) != len(selected_field_paths):
        raise ValueError("resolver match field is absent from its row source")
    return tuple(
        _LookupField(
            id=field.id,
            ref=field.field_ref,
            path=field.path,
            relative_path=relative_response_path(field.path, resolver_source.row_path),
            type=field.type.value,
            choices=field.choices,
        )
        for field in selected_fields
    )


def _lookup_field_matches(
    row: dict[str, Any],
    lookup_fields: tuple[_LookupField, ...],
    *,
    expected: CatalogScalarParameterValue,
) -> tuple[_LookupMatch, ...]:
    matches: list[_LookupMatch] = []
    for field in lookup_fields:
        value = path_value(row, field.relative_path, missing="")
        matched, matched_value, collection_member = _matching_lookup_value(
            value,
            expected=expected,
            type_name=field.type,
            choices=field.choices,
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
    expected: CatalogScalarParameterValue,
    type_name: str,
    choices: tuple[str, ...],
) -> tuple[bool, RuntimeScalar, bool]:
    if isinstance(value, list | tuple):
        matched = next(
            (
                item
                for item in value
                if not isinstance(item, list | tuple | Mapping)
                and _typed_lookup_values_match(
                    item,
                    expected=expected,
                    type_name=type_name,
                    choices=choices,
                )
            ),
            None,
        )
        return matched is not None, matched, True
    if isinstance(value, Mapping):
        return False, None, False
    if _typed_lookup_values_match(
        value,
        expected=expected,
        type_name=type_name,
        choices=choices,
    ):
        return True, value, False
    return False, None, False


def _typed_lookup_values_match(
    value: RuntimeScalar,
    *,
    expected: CatalogScalarParameterValue,
    type_name: str,
    choices: tuple[str, ...],
) -> bool:
    try:
        parsed = parse_catalog_parameter_value(
            value,
            type_name=type_name,
            choices=choices,
        )
    except CatalogParameterValueError:
        return False
    return type(parsed) is type(expected) and parsed == expected


def _binding_purpose(
    binding: CompatibleInputBinding,
    *,
    candidate: ResolverCandidate,
) -> InputBindingPurpose:
    selected_params = {
        item.param_ref: next(
            parameter
            for parameter in candidate.resolver_source.params
            if parameter.param_ref == item.param_ref
        )
        for item in binding.lookup_request_parameters
    }
    selected_targets = {
        (
            parameter.entity_target.entity_kind,
            parameter.entity_target.key_id,
            parameter.entity_target.component_id,
        )
        for parameter in selected_params.values()
        if parameter.entity_target is not None and str(parameter.source) == "path"
    }
    candidate_components = {
        (candidate.entity_kind, candidate.key_id, component.component_id)
        for component in candidate.key_components
    }
    if selected_targets == candidate_components:
        return InputBindingPurpose.IDENTITY_VALIDATION
    return InputBindingPurpose.REFERENCE_GROUNDING


def _matched_fields_are_stable_unique(
    rows: tuple[_ResolvedLookupRow, ...],
    *,
    candidate: ResolverCandidate,
    resolver_source: RowSource,
) -> bool:
    matched_field_ids = {row.matched_field_id for row in rows if row.matched_field_id}
    if len(matched_field_ids) != 1:
        return False
    matched_field_id = next(iter(matched_field_ids))
    return any(
        key.entity_kind == candidate.entity_kind
        and key.stable
        and len(key.components) == 1
        and key.components[0].field_id == matched_field_id
        for key in resolver_source.candidate_keys
    )
