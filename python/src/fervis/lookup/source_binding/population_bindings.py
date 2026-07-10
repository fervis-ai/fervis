"""Population binding requirements for source binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.answer_program.values import (
    IdentitySetValuePayload,
    ValueKind,
    known_input_id_for_value,
)
from fervis.lookup.question_inputs import KnownInputKind
from fervis.lookup.source_binding.model import (
    SourceBindingRequest,
    SourceCandidateDiscoveryRequest,
)


@dataclass(frozen=True)
class RowSetPopulationRequirement:
    input_ref: str
    resolved_input_ref: str
    requested_fact_id: str
    value_id: str
    memory_relation_id: str


@dataclass(frozen=True)
class PopulationBindingIndex:
    row_set_requirements: tuple[RowSetPopulationRequirement, ...] = ()

    @classmethod
    def from_request(
        cls,
        request: SourceCandidateDiscoveryRequest | SourceBindingRequest,
    ) -> "PopulationBindingIndex":
        return cls(row_set_requirements=_row_set_requirements(request))

    def bindings_for_candidate(
        self,
        candidate: Any,
        *,
        requested_fact_id: str = "",
    ) -> tuple[dict[str, Any], ...]:
        candidate_id = _candidate_id(candidate)
        relation_id = _candidate_memory_relation_id(candidate)
        row_set_bindings = tuple(
            _exact_row_set_binding(candidate_id, row_set)
            for row_set in self.row_set_requirements
            if row_set.memory_relation_id == relation_id
            and (
                not requested_fact_id or row_set.requested_fact_id == requested_fact_id
            )
        )
        if self._requested_fact_requires_row_set(requested_fact_id):
            return row_set_bindings
        if row_set_bindings:
            return row_set_bindings
        if _candidate_kind(candidate) == "same_scope_api_read":
            return (_prior_scope_replay_binding(candidate_id, candidate),)
        return (_candidate_population_binding(candidate_id),)

    def validate_selection(
        self,
        *,
        requested_fact_id: str,
        candidate: Any,
        population_binding_id: str,
    ) -> None:
        allowed = {
            str(item.get("population_binding_id") or "")
            for item in self.bindings_for_candidate(
                candidate,
                requested_fact_id=requested_fact_id,
            )
            if isinstance(item, dict)
        }
        if population_binding_id not in allowed:
            raise ValueError(
                "answer population is not admissible for requested fact population"
            )

    def _requested_fact_requires_row_set(self, requested_fact_id: str) -> bool:
        return any(
            row_set.requested_fact_id == requested_fact_id
            for row_set in self.row_set_requirements
        )


def _exact_row_set_binding(
    candidate_id: str,
    row_set: RowSetPopulationRequirement,
) -> dict[str, Any]:
    return {
        "population_binding_id": f"pop.{candidate_id}.{row_set.input_ref}.exact_row_set",
        "kind": "exact_row_set",
        "input_ref": row_set.input_ref,
        "basis": {
            "resolved_input_ref": row_set.resolved_input_ref,
            "memory_relation_id": row_set.memory_relation_id,
            "value_id": row_set.value_id,
            "proof_refs": [f"known_input:{row_set.input_ref}"],
        },
    }


def _prior_scope_replay_binding(
    candidate_id: str,
    candidate: Any,
) -> dict[str, Any]:
    return {
        "population_binding_id": f"pop.{candidate_id}.prior_scope_replay",
        "kind": "prior_scope_replay",
        "basis": {
            "memory_relation_id": _candidate_memory_relation_id(candidate),
            "bound_params": list(_prior_scope_bound_params(candidate)),
        },
    }


def _candidate_population_binding(candidate_id: str) -> dict[str, Any]:
    return {
        "population_binding_id": f"pop.{candidate_id}.candidate_population",
        "kind": "candidate_population",
    }


def _row_set_requirements(
    request: SourceCandidateDiscoveryRequest | SourceBindingRequest,
) -> tuple[RowSetPopulationRequirement, ...]:
    known_row_sets = {
        known.id: (fact.id, known.resolved_input_ref)
        for fact in request.requested_facts
        for known in fact.known_inputs
        if known.kind == KnownInputKind.ROW_SET_REFERENCE
    }
    output: list[RowSetPopulationRequirement] = list(
        _conversation_resolution_row_set_requirements(
            request,
            known_row_sets=known_row_sets,
        )
    )
    seen = {(item.input_ref, item.memory_relation_id) for item in output}
    for value in request.available_values:
        input_ref = known_input_id_for_value(value)
        if input_ref not in known_row_sets:
            continue
        if value.kind != ValueKind.IDENTITY_SET:
            continue
        payload = value.payload
        if not isinstance(payload, IdentitySetValuePayload):
            continue
        if not payload.source_relation_id:
            continue
        requested_fact_id, resolved_input_ref = known_row_sets[input_ref]
        if requested_fact_id not in value.applies_to_requested_fact_ids:
            continue
        key = (input_ref, payload.source_relation_id)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            RowSetPopulationRequirement(
                input_ref=input_ref,
                resolved_input_ref=resolved_input_ref,
                requested_fact_id=requested_fact_id,
                value_id=value.id,
                memory_relation_id=payload.source_relation_id,
            )
        )
    return tuple(output)


def _conversation_resolution_row_set_requirements(
    request: SourceCandidateDiscoveryRequest | SourceBindingRequest,
    *,
    known_row_sets: dict[str, tuple[str, str]],
) -> tuple[RowSetPopulationRequirement, ...]:
    overlay = request.conversation_resolution_overlay
    if overlay is None or not known_row_sets:
        return ()
    active_ids = {str(item) for item in request.active_memory_ids if str(item)}
    known_by_resolved_ref = {
        resolved_input_ref: (input_ref, requested_fact_id)
        for input_ref, (requested_fact_id, resolved_input_ref) in known_row_sets.items()
        if resolved_input_ref
    }
    output: list[RowSetPopulationRequirement] = []
    for item in overlay.resolved_question_inputs:
        if item.kind != KnownInputKind.ROW_SET_REFERENCE:
            continue
        known = known_by_resolved_ref.get(item.resolved_input_ref)
        if known is None:
            continue
        memory_ids = tuple(str(value) for value in item.memory_ids if str(value))
        if len(memory_ids) != 1:
            continue
        memory_id = memory_ids[0]
        if active_ids and memory_id not in active_ids:
            continue
        input_ref, requested_fact_id = known
        output.append(
            RowSetPopulationRequirement(
                input_ref=input_ref,
                resolved_input_ref=item.resolved_input_ref,
                requested_fact_id=requested_fact_id,
                value_id=item.resolved_input_ref,
                memory_relation_id=memory_id,
            )
        )
    return tuple(output)


def _candidate_id(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("source_candidate_id") or "")
    return str(getattr(candidate, "id", "") or "")


def _candidate_kind(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("kind") or "")
    return str(getattr(candidate, "kind", "") or "")


def _candidate_memory_relation_id(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("memory_relation_id") or "")
    payload = getattr(candidate, "payload", None)
    if isinstance(payload, dict):
        memory_relation_id = str(payload.get("memory_relation_id") or "")
        if memory_relation_id:
            return memory_relation_id
    source = getattr(candidate, "source", None)
    return str(getattr(source, "memory_relation_id", "") or "")


def _prior_scope_bound_params(candidate: Any) -> tuple[dict[str, str], ...]:
    typed_sets = tuple(getattr(candidate, "applied_param_binding_sets", ()) or ())
    if typed_sets:
        return _prior_scope_bound_params_from_binding_sets(typed_sets)
    typed_bindings = tuple(getattr(candidate, "applied_param_bindings", ()) or ())
    if typed_bindings:
        return tuple(
            _prior_scope_bound_param_from_binding(binding) for binding in typed_bindings
        )
    if not isinstance(candidate, dict):
        candidate = getattr(candidate, "payload", None) or {}
    direct = tuple(
        item for item in candidate.get("bound_params") or () if isinstance(item, dict)
    )
    if direct:
        return tuple(_prior_scope_bound_param(item) for item in direct)
    invocations = tuple(
        item
        for item in candidate.get("source_invocations") or ()
        if isinstance(item, dict)
    )
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for invocation in invocations:
        for item in invocation.get("bound_params") or ():
            if not isinstance(item, dict):
                continue
            param = _prior_scope_bound_param(item)
            key = (param["param_id"], param["source"], param["value"])
            if key in seen:
                continue
            seen.add(key)
            output.append(param)
    return tuple(output)


def _prior_scope_bound_param(item: dict[str, Any]) -> dict[str, str]:
    return {
        "param_id": str(item.get("param_id") or ""),
        "source": str(item.get("source") or "prior_scope"),
        "value": str(item.get("value") or ""),
    }


def _prior_scope_bound_param_from_binding(binding: Any) -> dict[str, str]:
    return {
        "param_id": str(getattr(binding, "param_id", "") or ""),
        "source": "prior_scope",
        "value": str(getattr(binding, "value", "") or ""),
    }


def _prior_scope_bound_params_from_binding_sets(
    binding_sets: tuple[tuple[Any, ...], ...],
) -> tuple[dict[str, str], ...]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for bindings in binding_sets:
        for binding in bindings:
            param = _prior_scope_bound_param_from_binding(binding)
            key = (param["param_id"], param["source"], param["value"])
            if key in seen:
                continue
            seen.add(key)
            output.append(param)
    return tuple(output)
