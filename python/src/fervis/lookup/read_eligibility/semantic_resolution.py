"""Execute the identity route selected by semantic Read Eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.answer_program.values import FactValue, IdentityValuePayload
from fervis.lookup.canonical_data import (
    EntityKeyComponentValue,
    EntityKeyValue,
    canonical_runtime_json,
)
from fervis.lookup.grounding import (
    CanonicalInputValue,
    IdentityExecutionClarification,
    IdentityExecutionCandidate,
    IdentityExecutionFailureReason,
    IdentityResolutionTask,
    IdentityResolverRoute,
    ResolvedIdentity,
)
from fervis.lookup.grounding.surface import resolver_option_surface_from_catalog
from fervis.lookup.lineage.source_reads import (
    SourceReadLineageScope,
    record_source_read_error,
    record_source_read_observation,
    require_catalog_endpoint_for_lineage,
)
from fervis.lookup.question_contract import InputTerm
from fervis.lookup.read_eligibility.semantic import IdentityRouteSelection
from fervis.lookup.relation_catalog import RelationCatalog, RelationDataAccessPort
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogParameterValue,
    CatalogParameterValueError,
    CatalogScalarParameterValue,
    parse_catalog_parameter_value,
)
from fervis.lookup.source_reads.response import (
    EndpointResponseError,
    extract_source_read_rows,
    observe_source_read_response,
    path_value,
    relative_response_path,
    source_read_completeness,
)


_MISSING = object()


@dataclass(frozen=True)
class _IdentityMatch:
    key: EntityKeyValue
    matched_field_id: str
    matched_field_ref: str
    matched_field_path: str
    matched_value: CatalogScalarParameterValue


@dataclass(frozen=True)
class _IdentityResolutionFailure:
    reason: IdentityExecutionFailureReason
    matches: tuple[_IdentityMatch, ...] = ()


def execute_identity_selection(
    *,
    task: IdentityResolutionTask,
    selection: IdentityRouteSelection,
    input_term: InputTerm,
    full_catalog: RelationCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_key_prefix: str,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> ResolvedIdentity | IdentityExecutionClarification:
    if selection.task_ref != task.task_ref or input_term.id != task.input_ref:
        raise ValueError("identity execution subject does not match its selection")
    route = _selected_route(task, selection=selection)
    operands = (
        (input_term.operand,)
        if isinstance(input_term.operand, str)
        else input_term.operand
    )
    resolved_values: list[FactValue] = []
    certification_refs: list[str] = []
    for position, operand in enumerate(operands, start=1):
        result = _resolve_operand(
            route=route,
            operand=operand,
            input_ref=input_term.id,
            value_id=f"{input_term.id}:identity:{position}",
            full_catalog=full_catalog,
            data_access_port=data_access_port,
            source_read_key=f"{source_read_key_prefix}:item:{position}",
            source_read_lineage=source_read_lineage,
        )
        if isinstance(result, _IdentityResolutionFailure):
            return _execution_clarification(
                task,
                reason=result.reason,
                evidence_refs=(
                    route.route_ref,
                    route.option.candidate.resolver_read_id,
                ),
                candidates=tuple(
                    IdentityExecutionCandidate(
                        key=match.key,
                        display_value=str(match.matched_value),
                        matched_field_ref=match.matched_field_ref,
                        matched_field_path=match.matched_field_path,
                        resolver_read_id=route.option.candidate.resolver_read_id,
                    )
                    for match in result.matches
                ),
            )
        resolved_values.append(result)
        certification_refs.extend(result.proof_refs)
        certification_refs.extend(result.source_refs)
    value = resolved_values[0]
    if len(resolved_values) > 1:
        keys = []
        for resolved in resolved_values:
            if not isinstance(resolved.payload, IdentityValuePayload):
                raise ValueError("identity resolver returned a non-identity value")
            keys.append(resolved.payload.key)
        value = FactValue.identity_set(
            id=f"{input_term.id}:identity_set",
            keys=tuple(keys),
            identity_evidence=tuple(evidence for resolved in resolved_values for evidence in resolved.identity_evidence),
            display_value=", ".join(operands),
            proof_refs=tuple(dict.fromkeys(certification_refs)),
            source_refs=tuple(
                dict.fromkeys(
                    ref for resolved in resolved_values for ref in resolved.source_refs
                )
            ),
            known_input_id=input_term.id,
        )
    certification_ref_tuple = tuple(dict.fromkeys(certification_refs))
    return ResolvedIdentity(
        task_ref=task.task_ref,
        input_ref=task.input_ref,
        use_refs=task.use_refs,
        canonical_option_id=selection.canonical_option_id,
        resolver_route_id=selection.resolver_route_id,
        canonical_value=CanonicalInputValue(
            canonical_value_id=value.id,
            input_ref=task.input_ref,
            use_refs=task.use_refs,
            typed_value=value,
            certification_refs=certification_ref_tuple,
        ),
    )


def _selected_route(
    task: IdentityResolutionTask,
    *,
    selection: IdentityRouteSelection,
) -> IdentityResolverRoute:
    route = next(
        (
            item
            for item in task.resolver_routes
            if item.route_ref == selection.resolver_route_id
        ),
        None,
    )
    if route is None:
        raise ValueError("identity execution references an unknown resolver route")
    option = next(
        (
            item
            for item in task.canonical_options
            if item.canonical_option_id == selection.canonical_option_id
        ),
        None,
    )
    if option is None or route.route_ref not in option.resolver_route_refs:
        raise ValueError("identity route does not produce the selected meaning")
    return route


def _resolve_operand(
    *,
    route: IdentityResolverRoute,
    operand: str,
    input_ref: str,
    value_id: str,
    full_catalog: RelationCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_key: str,
    source_read_lineage: SourceReadLineageScope | None,
) -> FactValue | _IdentityResolutionFailure:
    option = route.option
    candidate = option.candidate
    surface = resolver_option_surface_from_catalog(full_catalog, option)
    args: dict[str, CatalogParameterValue] = {
        parameter.param_ref: parameter.default
        for parameter in surface.request_parameters
        if parameter.default is not None
    }
    lookup_value: CatalogScalarParameterValue | None = None
    for param_ref in route.compatibility.lookup_request_param_refs:
        _parameter, value = surface.compiled_request_value(
            param_ref,
            lookup_text=operand,
        )
        args[param_ref] = value
        lookup_value = lookup_value if lookup_value is not None else value
    if lookup_value is None:
        raise ValueError("identity route lacks a lookup request value")
    read = full_catalog.read(candidate.resolver_read_id)
    require_catalog_endpoint_for_lineage(
        source_read_lineage=source_read_lineage,
        endpoint_name=candidate.resolver_endpoint_name,
        catalog_endpoint=read.catalog_endpoint,
    )
    try:
        response = data_access_port.read(
            endpoint_name=candidate.resolver_endpoint_name,
            args=args,
        )
    except Exception as exc:
        record_source_read_error(
            source_read_lineage,
            source_read_key=source_read_key,
            endpoint_name=candidate.resolver_endpoint_name,
            catalog_endpoint=read.catalog_endpoint,
            args=args,
            error_json={"error": str(exc), "errorType": type(exc).__name__},
        )
        return _IdentityResolutionFailure(
            IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT
        )
    observation = observe_source_read_response(
        response,
        endpoint_name=candidate.resolver_endpoint_name,
    )
    completeness = source_read_completeness(response)
    source_read_id = record_source_read_observation(
        source_read_lineage,
        source_read_key=source_read_key,
        endpoint_name=candidate.resolver_endpoint_name,
        catalog_endpoint=read.catalog_endpoint,
        args=args,
        observation=observation,
        response_body=response.get("responseBody"),
        completeness_json=completeness,
    )
    if not observation.succeeded:
        return _IdentityResolutionFailure(
            IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT
        )
    try:
        rows = extract_source_read_rows(
            response,
            endpoint_name=candidate.resolver_endpoint_name,
            row_source=candidate.resolver_source,
        )
        matches = _identity_matches(
            rows,
            route=route,
            lookup_value=lookup_value,
        )
    except (CatalogParameterValueError, EndpointResponseError, ValueError):
        return _IdentityResolutionFailure(
            IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT
        )
    unique_matches = tuple(
        {
            canonical_runtime_json(match.key.component_values()): match
            for match in matches
        }.values()
    )
    if not unique_matches:
        return _IdentityResolutionFailure(IdentityExecutionFailureReason.NOT_FOUND)
    if len(unique_matches) > 1:
        return _IdentityResolutionFailure(
            IdentityExecutionFailureReason.AMBIGUOUS_RESULT,
            unique_matches,
        )
    [match] = unique_matches
    if bool(completeness["truncated"]) and not _matched_field_is_stable_unique(
        match,
        route=route,
    ):
        return _IdentityResolutionFailure(
            IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT
        )
    proof_refs = [f"question_input:{input_ref}"]
    if source_read_id:
        proof_refs.append(f"source_read:{source_read_id}")
    return FactValue.identity(
        id=value_id,
        known_input_id=input_ref,
        key=match.key,
        display_value=str(match.matched_value),
        matched_field_ref=match.matched_field_ref,
        matched_field_path=match.matched_field_path,
        matched_value=match.matched_value,
        proof_refs=tuple(proof_refs),
        source_refs=(candidate.resolver_read_id, candidate.resolver_endpoint_name),
    )


def _identity_matches(
    rows: tuple[dict[str, Any], ...],
    *,
    route: IdentityResolverRoute,
    lookup_value: CatalogScalarParameterValue,
) -> tuple[_IdentityMatch, ...]:
    candidate = route.option.candidate
    source = candidate.resolver_source
    selected_fields = tuple(
        source_field
        for path in route.compatibility.returned_identity_verification_field_paths
        if (
            source_field := next(
                (field for field in source.fields if field.path == path), None
            )
        )
        is not None
    )
    if len(selected_fields) != len(
        route.compatibility.returned_identity_verification_field_paths
    ):
        raise ValueError("identity route references an unavailable verification field")
    matches: list[_IdentityMatch] = []
    for row in rows:
        for field in selected_fields:
            raw_value = path_value(
                row,
                relative_response_path(field.path, source.row_path),
                missing=_MISSING,
            )
            if raw_value is _MISSING or isinstance(raw_value, (dict, list, tuple)):
                continue
            try:
                parsed = parse_catalog_parameter_value(
                    raw_value,
                    type_name=field.type.value,
                    choices=field.choices,
                )
            except CatalogParameterValueError:
                continue
            if type(parsed) is not type(lookup_value) or parsed != lookup_value:
                continue
            key = _identity_key(row, route=route)
            if key is not None:
                matches.append(
                    _IdentityMatch(
                        key=key,
                        matched_field_id=field.id,
                        matched_field_ref=field.field_ref,
                        matched_field_path=field.path,
                        matched_value=parsed,
                    )
                )
            break
    return tuple(matches)


def _identity_key(
    row: dict[str, Any],
    *,
    route: IdentityResolverRoute,
) -> EntityKeyValue | None:
    candidate = route.option.candidate
    source = candidate.resolver_source
    components: list[EntityKeyComponentValue] = []
    for component in candidate.key_components:
        field = source.field(component.field_id)
        value = path_value(
            row,
            relative_response_path(field.path, source.row_path),
            missing=_MISSING,
        )
        if value is _MISSING or value is None or value == "":
            return None
        components.append(EntityKeyComponentValue(component.component_id, value))
    return EntityKeyValue(
        entity_kind=candidate.entity_kind,
        key_id=candidate.key_id,
        components=tuple(components),
    )


def _matched_field_is_stable_unique(
    match: _IdentityMatch,
    *,
    route: IdentityResolverRoute,
) -> bool:
    candidate = route.option.candidate
    return any(
        key.entity_kind == candidate.entity_kind
        and key.stable
        and len(key.components) == 1
        and key.components[0].field_id == match.matched_field_id
        for key in candidate.resolver_source.candidate_keys
    )


def _execution_clarification(
    task: IdentityResolutionTask,
    *,
    reason: IdentityExecutionFailureReason,
    evidence_refs: tuple[str, ...],
    candidates: tuple[IdentityExecutionCandidate, ...] = (),
) -> IdentityExecutionClarification:
    return IdentityExecutionClarification(
        task_ref=task.task_ref,
        input_ref=task.input_ref,
        use_refs=task.use_refs,
        reason=reason,
        evidence_refs=evidence_refs,
        candidates=candidates,
    )


__all__ = ["execute_identity_selection"]
