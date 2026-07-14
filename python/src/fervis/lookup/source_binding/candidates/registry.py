"""Source-binding candidate registry orchestration."""

from dataclasses import replace
from typing import cast
from fervis.lookup.question_contract import RequestedFact

from ._shared import (
    BoundSource,
    RelationCatalog,
    SourceBindingRequest,
    SourceCandidateDiscoveryRequest,
)
from .bound_payload import _bound_sources_prompt_payload
from .candidate_tree import map_source_candidate_tree
from .compact import _compact_prompt_payload, _visible_fulfillment_support_sets
from .handles import _with_fulfillment_slots, _with_stable_source_candidate_handles
from .field_scope import SourceBindingFieldScope
from .indexes import (
    _candidate_fulfillment_answer_output_ids,
    _candidate_fulfillment_support_set_ids_by_answer_output,
    _candidate_population_binding_ids,
    _candidate_requested_fact_ids,
    _candidate_required_param_decision_ids,
)
from .model import SourceCandidate, SourceCandidateRegistry
from .population import _with_population_bindings
from .population_roles import _with_source_population_roles
from .raw_payload import _raw_source_binding_candidate_payload
from .registry_builder import _source_candidates_from_cards
from fervis.lookup.source_binding.candidates.contracts import JsonObject
from .same_scope import _same_scope_read_scopes
from .plan_selection_filter import filter_prompt_payload_by_plan_selection
from .row_predicates import with_row_predicates
from ..normal_instance_roles import with_normal_instance_role_profiles


def same_scope_read_ids(
    memory_inputs: dict[str, object],
    *,
    relation_catalog: RelationCatalog,
) -> tuple[str, ...]:
    """Return catalog reads proven reusable by prior memory relation scopes."""

    scopes = _same_scope_read_scopes(
        memory_inputs,
        relation_catalog=relation_catalog,
    )
    return tuple(dict.fromkeys(scope.read_id for scope in scopes))


def source_binding_candidate_payload(
    request: SourceBindingRequest,
) -> dict[str, object]:
    payload = source_candidate_registry(request).prompt_payload
    return {key: value for key, value in payload.items()}


def source_candidate_discovery_registry(
    request: SourceCandidateDiscoveryRequest,
) -> SourceCandidateRegistry:
    field_scope = SourceBindingFieldScope.from_read_eligibility(
        request.read_eligibility
    )
    candidate_payload = _source_candidate_payload(
        request,
        field_scope=field_scope,
    )
    candidate_payload = with_row_predicates(
        candidate_payload,
        relation_catalog=request.relation_catalog,
        field_scope=field_scope,
    )
    candidate_cards = cast(JsonObject, candidate_payload)
    candidates_by_id = _source_candidates_from_cards(candidate_cards)
    return SourceCandidateRegistry(
        prompt_payload=candidate_cards,
        candidates_by_id=candidates_by_id,
        prompt_candidate_ids=_prompt_candidate_ids(candidate_payload),
    )


def source_candidate_discovery_payload(
    request: SourceCandidateDiscoveryRequest,
) -> dict[str, object]:
    payload = source_candidate_discovery_registry(request).prompt_payload
    return {key: value for key, value in payload.items()}


def source_binding_prompt_candidate_requested_fact_ids(
    request: SourceBindingRequest,
) -> dict[str, tuple[str, ...]]:
    registry = source_candidate_registry(request)
    return _candidate_requested_fact_ids(
        registry.candidates_by_id,
        candidate_ids=registry.prompt_candidate_ids,
    )


def source_candidate_required_param_decision_ids(
    candidate: SourceCandidate,
) -> tuple[str, ...]:
    return _candidate_required_param_decision_ids(candidate)


def source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output(
    request: SourceBindingRequest,
) -> dict[str, dict[str, tuple[str, ...]]]:
    registry = source_candidate_registry(request)
    return _candidate_fulfillment_support_set_ids_by_answer_output(
        registry.candidates_by_id,
        candidate_ids=registry.prompt_candidate_ids,
    )


def source_binding_prompt_candidate_fulfillment_answer_output_ids(
    request: SourceBindingRequest,
) -> dict[str, tuple[str, ...]]:
    registry = source_candidate_registry(request)
    return _candidate_fulfillment_answer_output_ids(
        registry.candidates_by_id,
        candidate_ids=registry.prompt_candidate_ids,
    )


def source_binding_prompt_candidate_population_binding_ids(
    request: SourceBindingRequest,
) -> dict[str, tuple[str, ...]]:
    registry = source_candidate_registry(request)
    return _candidate_population_binding_ids(
        registry.candidates_by_id,
        candidate_ids=registry.prompt_candidate_ids,
    )


def source_candidates(request: SourceBindingRequest) -> dict[str, SourceCandidate]:
    return source_candidate_registry(request).candidates_by_id


def source_candidate_registry(request: SourceBindingRequest) -> SourceCandidateRegistry:
    candidate_payload = cast(
        dict[str, object],
        request.source_candidates.prompt_payload,
    )
    selected_candidate_payload = filter_prompt_payload_by_plan_selection(
        candidate_payload,
        request,
    )
    selected_candidate_payload = _with_visible_fulfillment_choice_ids(
        selected_candidate_payload,
        requested_facts=request.requested_facts,
    )
    prompt_payload = _compact_prompt_payload(
        selected_candidate_payload,
        relation_catalog=request.relation_catalog,
        requested_facts=request.requested_facts,
    )
    prompt_candidate_ids = _prompt_candidate_ids(prompt_payload)
    choice_ids_by_candidate = _fulfillment_choice_ids_by_candidate(
        selected_candidate_payload
    )
    candidates_by_id = {
        candidate_id: _candidate_with_fulfillment_choice_ids(
            request.source_candidates.candidates_by_id[candidate_id],
            choice_ids=choice_ids_by_candidate.get(candidate_id, {}),
        )
        for candidate_id in prompt_candidate_ids
    }
    return SourceCandidateRegistry(
        prompt_payload=cast(JsonObject, prompt_payload),
        candidates_by_id=candidates_by_id,
        prompt_candidate_ids=prompt_candidate_ids,
    )


def _candidate_with_fulfillment_choice_ids(
    candidate: SourceCandidate,
    *,
    choice_ids: dict[str, str],
) -> SourceCandidate:
    support_sets = tuple(
        replace(
            support_set,
            fulfillment_choice_id=choice_ids.get(
                support_set.fulfillment_support_set_id,
                support_set.fulfillment_choice_id,
            ),
        )
        for support_set in candidate.fulfillment_support_sets
    )
    return replace(candidate, fulfillment_support_sets=support_sets)


def _fulfillment_choice_ids_by_candidate(
    payload: dict[str, object],
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for candidate in _candidate_cards(payload):
        candidate_id = str(candidate.get("source_candidate_id") or "")
        if not candidate_id:
            continue
        choice_ids = output.setdefault(candidate_id, {})
        support_sets = candidate.get("fulfillment_support_sets")
        if not isinstance(support_sets, (list, tuple)):
            continue
        for support_set in support_sets:
            if not isinstance(support_set, dict):
                continue
            support_set_id = str(
                support_set.get("fulfillment_support_set_id") or ""
            )
            choice_id = str(support_set.get("fulfillment_choice_id") or "")
            if support_set_id and choice_id:
                choice_ids[support_set_id] = choice_id
    return output


def _candidate_cards(payload: dict[str, object]) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    fact_sources_list = payload.get("requested_fact_sources")
    if isinstance(fact_sources_list, (list, tuple)):
        for fact_sources in fact_sources_list:
            if not isinstance(fact_sources, dict):
                continue
            contexts = fact_sources.get("source_contexts")
            if not isinstance(contexts, (list, tuple)):
                continue
            for context in contexts:
                if not isinstance(context, dict):
                    continue
                candidates = context.get("source_options")
                if not isinstance(candidates, (list, tuple)):
                    continue
                output.extend(
                    candidate for candidate in candidates if isinstance(candidate, dict)
                )
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        candidates = payload.get(key)
        if isinstance(candidates, (list, tuple)):
            output.extend(
                candidate for candidate in candidates if isinstance(candidate, dict)
            )
    return tuple(output)


def _source_candidate_payload(
    request: SourceCandidateDiscoveryRequest | SourceBindingRequest,
    *,
    field_scope: SourceBindingFieldScope,
    selected_value_ids: frozenset[str] = frozenset(),
) -> dict[str, object]:
    payload = _raw_source_binding_candidate_payload(
        request,
        selected_value_ids=selected_value_ids,
    )
    candidate_payload = _with_stable_source_candidate_handles(
        payload,
        requested_facts=request.requested_facts,
        project_fulfillment_slots=False,
    )
    candidate_payload = _with_source_population_roles(
        candidate_payload,
        request=request,
    )
    candidate_payload = _with_fulfillment_slots(
        candidate_payload,
        requested_facts=request.requested_facts,
        field_scope=field_scope,
    )
    candidate_payload = _with_population_bindings(candidate_payload, request=request)
    return with_normal_instance_role_profiles(
        candidate_payload,
        request=request,
    )


def bound_sources_prompt_payload(
    *,
    bound_sources: tuple[BoundSource, ...],
) -> dict[str, object]:
    return _bound_sources_prompt_payload(bound_sources=bound_sources)


def _prompt_candidate_ids(payload: dict[str, object]) -> tuple[str, ...]:
    output: list[str] = []
    fact_sources_list = payload.get("requested_fact_sources")
    if not isinstance(fact_sources_list, (list, tuple)):
        fact_sources_list = ()
    for fact_sources in fact_sources_list:
        if not isinstance(fact_sources, dict):
            continue
        contexts = fact_sources.get("source_contexts")
        if not isinstance(contexts, (list, tuple)):
            continue
        for context in contexts:
            if not isinstance(context, dict):
                continue
            output.extend(_candidate_ids(context.get("source_options")))
    for key in (
        "utility_source_candidates",
        "value_source_candidates",
    ):
        output.extend(_candidate_ids(payload.get(key)))
    return tuple(dict.fromkeys(output))


def _candidate_ids(candidates: object) -> tuple[str, ...]:
    if not isinstance(candidates, (list, tuple)):
        return ()
    return tuple(
        candidate_id
        for candidate in candidates
        if isinstance(candidate, dict)
        for candidate_id in (str(candidate.get("source_candidate_id") or ""),)
        if candidate_id
    )


def _with_visible_fulfillment_choice_ids(
    payload: dict[str, object],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> dict[str, object]:
    return map_source_candidate_tree(
        payload,
        lambda candidate, context: _candidate_with_visible_fulfillment_choice_ids(
            candidate,
            requested_fact=_requested_fact_for_context(
                context.requested_fact_id,
                requested_facts=requested_facts,
            ),
        ),
        top_level_keys=("utility_source_candidates", "value_source_candidates"),
    )


def _candidate_with_visible_fulfillment_choice_ids(
    candidate: dict[str, object],
    *,
    requested_fact: RequestedFact | None,
) -> dict[str, object]:
    choices_by_slots = {
        _support_set_slot_ids(choice): str(choice.get("fulfillment_choice_id") or "")
        for choice in _visible_fulfillment_support_sets(
            candidate,
            requested_fact=requested_fact,
        )
        if _support_set_slot_ids(choice) and choice.get("fulfillment_choice_id")
    }
    if not choices_by_slots:
        return candidate
    output = dict(candidate)
    support_sets = candidate.get("fulfillment_support_sets")
    if not isinstance(support_sets, (list, tuple)):
        support_sets = ()
    output["fulfillment_support_sets"] = [
        _support_set_with_visible_choice_id(
            support_set,
            choices_by_slots=choices_by_slots,
        )
        for support_set in support_sets
        if isinstance(support_set, dict)
    ]
    return output


def _requested_fact_for_context(
    requested_fact_id: str,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> RequestedFact | None:
    if not requested_fact_id:
        return None
    return next(
        (
            fact
            for fact in requested_facts
            if str(getattr(fact, "id", "") or "") == requested_fact_id
        ),
        None,
    )


def _support_set_with_visible_choice_id(
    support_set: dict[str, object],
    *,
    choices_by_slots: dict[tuple[str, ...], str],
) -> dict[str, object]:
    choice_id = choices_by_slots.get(_support_set_slot_ids(support_set))
    if not choice_id:
        return support_set
    return {**support_set, "fulfillment_choice_id": choice_id}


def _support_set_slot_ids(support_set: dict[str, object]) -> tuple[str, ...]:
    slots = support_set.get("fulfillment_slots")
    if not isinstance(slots, (list, tuple)):
        return ()
    return tuple(
        str(slot.get("fulfillment_slot_id") or "")
        for slot in slots
        if isinstance(slot, dict) and str(slot.get("fulfillment_slot_id") or "")
    )
