"""Raw source-binding candidate payload assembly."""

from ._shared import (
    Any,
    FactValue,
    RelationCatalog,
    SourceCandidateInputRequest,
    ValueKind,
    known_input_id_for_value,
    operation_input_values_payload,
    selected_relation_catalog_payload,
)
from fervis.lookup.source_binding.candidates.contracts import (
    JsonValue,
)
from .api_sources import _api_candidate_payload
from .eligibility import _memory_candidate_with_fact_eligibility
from .memory import (
    _has_answer_evidence_fields,
    _memory_candidate_payloads,
    _source_contexts_for_fact,
)
from .same_scope import _same_scope_api_candidate_payloads
from .values import (
    _calendar_candidate_payload,
    _memory_value_candidate_payloads,
    _value_candidate_payload,
)
from fervis.lookup.fact_planning.row_set_filters import (
    filter_row_set_filters_for_requested_fact,
    value_applies_to_requested_fact,
)
from fervis.lookup.read_eligibility.candidate_identity import (
    read_candidate_signature,
)
from fervis.lookup.read_eligibility import (
    retained_source_candidate_ids_by_signature,
)
from fervis.lookup.question_contract import RequestedFactLiteralInput


def _raw_source_binding_candidate_payload(
    request: SourceCandidateInputRequest,
    *,
    selected_value_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    requested_fact_ids = tuple(fact.id for fact in request.requested_facts)
    relation_payload = selected_relation_catalog_payload(
        request.relation_catalog,
        catalog_selection=request.catalog_selection,
        memory_inputs=request.memory_inputs,
        available_values=request.available_values,
        available_value_uses=(),
    )
    value_payload = operation_input_values_payload(
        available_values=request.available_values,
        available_value_uses=request.available_value_uses,
    )
    same_scope_sources = _same_scope_api_candidate_payloads(
        request.memory_inputs,
        relation_catalog=_same_scope_memory_candidate_catalog(request),
    )
    read_eligibility_source_candidate_ids = _read_eligibility_source_candidate_ids(
        request
    )
    memory_sources_by_id = {}
    for candidate in _memory_candidate_payloads(request.memory_inputs):
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("source_candidate_id") or "")
        if not candidate_id:
            continue
        if not _candidate_is_active_memory_source(candidate, request=request):
            continue
        eligible_candidate = _memory_candidate_with_fact_eligibility(
            candidate,
            request=request,
        )
        if eligible_candidate is not None:
            memory_sources_by_id[candidate_id] = eligible_candidate
    memory_value_sources = _memory_value_candidate_payloads(request.memory_inputs)
    if selected_value_ids:
        memory_value_sources.extend(
            _memory_value_candidate_payloads(
                request.memory_inputs,
                source_linked=True,
            )
        )
    utility_sources = _with_default_requested_fact_applicability(
        _utility_source_candidates(
            relation_payload,
            request=request,
        ),
        requested_fact_ids=requested_fact_ids,
    )
    memory_source_candidates = _question_scoped_memory_source_candidates(
        memory_sources=tuple(memory_sources_by_id.values()),
        same_scope_sources=tuple(
            eligible_candidate
            for candidate in same_scope_sources
            if _candidate_is_active_memory_source(candidate, request=request)
            for eligible_candidate in (
                _memory_candidate_with_fact_eligibility(candidate, request=request),
            )
            if eligible_candidate is not None
        ),
    )
    requested_fact_sources: list[dict[str, Any]] = []
    promoted_utility_sources = False
    relation_items_by_fact = {
        str(item.get("requested_fact_id") or ""): item
        for item in relation_payload.get("requested_fact_relations") or ()
        if isinstance(item, dict) and str(item.get("requested_fact_id") or "")
    }
    for fact in request.requested_facts:
        requested_fact_id = fact.id
        item = relation_items_by_fact.get(requested_fact_id, {})
        source_contexts = _source_contexts_for_fact(
            requested_fact_id,
            api_sources=_api_sources_for_fact(
                item,
                request=request,
                source_candidate_ids=read_eligibility_source_candidate_ids,
            ),
        )
        memory_context = _memory_source_context_for_fact(
            requested_fact_id,
            memory_source_candidates=memory_source_candidates,
        )
        if memory_context is not None:
            source_contexts.append(memory_context)
        if (
            not source_contexts
            and utility_sources
            and not _fact_has_filtered_api_candidates(
                requested_fact_id, request=request
            )
        ):
            promoted_utility_sources = True
            source_options: list[JsonValue] = list(utility_sources)
            source_contexts = [
                {
                    "context_id": f"requested_fact:{requested_fact_id}:generated_relations",
                    "kind": "generated_relations",
                    "ordering_rationale": (
                        "backend-generated relations derived from grounded inputs"
                    ),
                    "source_options": source_options,
                }
            ]
        requested_fact_sources.append(
            {
                "requested_fact_id": requested_fact_id,
                "source_contexts": source_contexts,
            }
        )
    current_value_sources = [
        _value_candidate_payload(item)
        for item in value_payload.get("values") or ()
        if isinstance(item, dict)
        and not _is_identity_binding_value(item, request.available_values)
        and not _is_formula_modifier_value(item, request=request)
    ]
    visible_memory_value_candidates = [
        eligible_candidate
        for candidate in memory_value_sources
        if _candidate_is_active_memory_value(
            candidate,
            request=request,
            selected_value_ids=selected_value_ids,
        )
        for eligible_candidate in (
            _memory_candidate_with_fact_eligibility(candidate, request=request),
        )
        if eligible_candidate is not None
    ]
    value_sources = _with_default_requested_fact_applicability(
        _distinct_value_candidates(
            [*current_value_sources, *visible_memory_value_candidates]
        ),
        requested_fact_ids=requested_fact_ids,
    )
    payload: dict[str, Any] = {
        "requested_fact_sources": requested_fact_sources,
    }
    if (
        utility_sources
        and not promoted_utility_sources
        and request.read_eligibility is None
    ):
        payload["utility_source_candidates"] = utility_sources
    if value_sources:
        payload["value_source_candidates"] = value_sources
    if relation_payload.get("missing_required_inputs"):
        payload["missing_required_inputs"] = relation_payload["missing_required_inputs"]
    return payload


def _is_formula_modifier_value(
    payload: dict[str, Any],
    *,
    request: SourceCandidateInputRequest,
) -> bool:
    value_id = str(payload.get("value_id") or "")
    value = next((item for item in request.available_values if item.id == value_id), None)
    if value is None or not value.known_input_id:
        return False
    known_input = next(
        (
            item
            for item in request.question_contract.question_inputs
            if item.id == value.known_input_id
        ),
        None,
    )
    return isinstance(known_input, RequestedFactLiteralInput) and known_input.is_formula_value


def _with_default_requested_fact_applicability(
    candidates: list[dict[str, Any]],
    *,
    requested_fact_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("applies_to_requested_facts"):
            output.append(candidate)
            continue
        output.append(
            {
                **candidate,
                "applies_to_requested_facts": list(requested_fact_ids),
            }
        )
    return output


def _distinct_value_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_value_ids: set[str] = set()
    for candidate in candidates:
        value_id = str(candidate.get("value_id") or "")
        if not value_id or value_id in seen_value_ids:
            continue
        seen_value_ids.add(value_id)
        output.append(candidate)
    return output


def _same_scope_memory_candidate_catalog(
    request: SourceCandidateInputRequest,
) -> RelationCatalog:
    if request.read_eligibility is not None:
        return request.relation_catalog
    return request.same_scope_relation_catalog or request.relation_catalog


def _utility_source_candidates(
    relation_payload: dict[str, Any],
    *,
    request: SourceCandidateInputRequest,
) -> list[dict[str, Any]]:
    return [
        _calendar_candidate_payload(item, available_values=request.available_values)
        for item in relation_payload.get("utility_relations") or ()
        if isinstance(item, dict)
    ]


def _fact_has_filtered_api_candidates(
    requested_fact_id: str,
    *,
    request: SourceCandidateInputRequest,
) -> bool:
    for selection in request.catalog_selection.requested_fact_selections:
        if selection.requested_fact_id != requested_fact_id:
            continue
        return bool(selection.unselected_positive_read_ids)
    return False


def _memory_source_context_for_fact(
    requested_fact_id: str,
    *,
    memory_source_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    source_options = [
        candidate
        for candidate in memory_source_candidates
        if requested_fact_id in (candidate.get("applies_to_requested_facts") or ())
    ]
    if not source_options:
        return None
    return {
        "context_id": f"requested_fact:{requested_fact_id}:memory_sources",
        "kind": "memory_sources",
        "ordering_rationale": (
            "active memory sources whose scopes apply to this requested fact"
        ),
        "source_options": source_options,
    }


def _question_scoped_memory_source_candidates(
    *,
    memory_sources: tuple[dict[str, Any], ...],
    same_scope_sources: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in (*same_scope_sources, *memory_sources):
        candidate_id = str(candidate.get("source_candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        output.append(dict(candidate))
    return output


def _candidate_is_active_memory_source(
    candidate: dict[str, Any],
    *,
    request: SourceCandidateInputRequest,
) -> bool:
    active_ids = {str(item) for item in request.active_memory_ids if str(item)}
    if not active_ids:
        return True
    return any(
        str(candidate.get(key) or "") in active_ids
        for key in ("source_candidate_id", "memory_relation_id")
    )


def _candidate_is_active_memory_value(
    candidate: dict[str, Any],
    *,
    request: SourceCandidateInputRequest,
    selected_value_ids: frozenset[str] = frozenset(),
) -> bool:
    if str(candidate.get("value_id") or "") in selected_value_ids:
        return True
    active_ids = {str(item) for item in request.active_memory_ids if str(item)}
    if not active_ids:
        return True
    return any(
        str(candidate.get(key) or "") in active_ids
        for key in ("source_candidate_id", "value_id")
    )


def _is_identity_binding_value(
    payload: dict[str, Any],
    available_values: tuple[FactValue, ...],
) -> bool:
    value_id = str(payload.get("value_id") or "")
    return any(
        value.id == value_id
        and value.kind in {ValueKind.IDENTITY, ValueKind.IDENTITY_SET}
        for value in available_values
    )


def _api_sources_for_fact(
    item: dict[str, Any],
    *,
    request: SourceCandidateInputRequest,
    source_candidate_ids: dict[str, str],
) -> list[dict[str, Any]]:
    requested_fact_id = str(item.get("requested_fact_id") or "")
    available_values = _source_parameter_values_for_fact(
        requested_fact_id,
        request=request,
    )
    available_relations = tuple(
        candidate
        for candidate in item.get("available_relations") or ()
        if isinstance(candidate, dict)
    )
    return [
        api_candidate
        for candidate in available_relations
        for api_candidate in (
            _api_candidate_payload_for_fact(
                candidate,
                requested_fact_id=requested_fact_id,
                available_values=available_values,
                request=request,
                source_candidate_ids=source_candidate_ids,
            ),
        )
        if api_candidate is not None
        if _has_answer_evidence_fields(api_candidate)
    ]


def _api_candidate_payload_for_fact(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
    available_values: tuple[FactValue, ...],
    request: SourceCandidateInputRequest,
    source_candidate_ids: dict[str, str],
) -> dict[str, Any] | None:
    output = _api_candidate_payload(
        candidate,
        available_values=available_values,
    )
    source_candidate_signature = read_candidate_signature(
        _read_eligibility_signature_candidate(output, original_candidate=candidate),
        requested_fact_id=requested_fact_id,
    )
    if request.read_eligibility is not None and (
        source_candidate_signature not in source_candidate_ids
    ):
        return None
    output["source_candidate_signature"] = source_candidate_signature
    output["applied_filters"] = filter_row_set_filters_for_requested_fact(
        tuple(
            item
            for item in output.get("applied_filters") or ()
            if isinstance(item, dict)
        ),
        requested_fact_id=requested_fact_id,
        available_values=request.available_values,
    )
    if not output["applied_filters"]:
        output.pop("applied_filters", None)
    source_candidate_id = source_candidate_ids.get(source_candidate_signature)
    if source_candidate_id:
        output["source_candidate_id"] = source_candidate_id
    return output


def _read_eligibility_signature_candidate(
    candidate: dict[str, Any],
    *,
    original_candidate: dict[str, Any],
) -> dict[str, Any]:
    catalog_default_param_ids = _catalog_default_param_ids(original_candidate)
    if not catalog_default_param_ids:
        return candidate
    output = dict(candidate)
    output["bound_params"] = [
        param
        for param in output.get("bound_params") or ()
        if not (
            isinstance(param, dict)
            and str(param.get("source") or "") == "source_default"
            and str(param.get("param_id") or "") in catalog_default_param_ids
        )
    ]
    if not output["bound_params"]:
        output.pop("bound_params", None)
    return output


def _catalog_default_param_ids(candidate: dict[str, Any]) -> set[str]:
    return {
        str(param.get("param_id") or "")
        for param in candidate.get("params") or ()
        if isinstance(param, dict)
        and str(param.get("param_id") or "")
        and param.get("default") is not None
        and str(param.get("default_source") or "") != "source_variant"
    }


def _read_eligibility_source_candidate_ids(
    request: SourceCandidateInputRequest,
) -> dict[str, str]:
    if request.read_eligibility is None:
        return {}
    return retained_source_candidate_ids_by_signature(request.read_eligibility)


def _source_parameter_values_for_fact(
    requested_fact_id: str,
    *,
    request: SourceCandidateInputRequest,
) -> tuple[FactValue, ...]:
    result_limit_input_ids = {
        known.id
        for fact in request.requested_facts
        if fact.id == requested_fact_id
        for known in fact.known_inputs
        if known.is_result_limit
    }
    return tuple(
        value
        for value in request.available_values
        if value_applies_to_requested_fact(value, requested_fact_id)
        and known_input_id_for_value(value) not in result_limit_input_ids
    )
