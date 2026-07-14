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
from fervis.lookup.source_binding.candidates.model import SourceCandidate
from fervis.lookup.source_binding.candidates.contracts import JsonObject, JsonValue


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

    def bindings_for_card(
        self,
        candidate: JsonObject,
        *,
        requested_fact_id: str = "",
    ) -> tuple[dict[str, Any], ...]:
        candidate_id = str(candidate.get("source_candidate_id") or "")
        relation_id = str(candidate.get("memory_relation_id") or "")
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
        if str(candidate.get("kind") or "") == "same_scope_api_read":
            return (_prior_scope_replay_binding(candidate_id, candidate),)
        return (_candidate_population_binding(candidate_id),)

    def validate_selection(
        self,
        *,
        requested_fact_id: str,
        candidate: SourceCandidate,
        population_binding_id: str,
    ) -> None:
        allowed = {item.id for item in candidate.population_bindings}
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
    candidate: JsonObject,
) -> dict[str, Any]:
    return {
        "population_binding_id": f"pop.{candidate_id}.prior_scope_replay",
        "kind": "prior_scope_replay",
        "basis": {
            "memory_relation_id": str(candidate.get("memory_relation_id") or ""),
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
    resolution = request.conversation_resolution
    if resolution is None or not known_row_sets:
        return ()
    active_ids = {str(item) for item in request.active_memory_ids if str(item)}
    known_by_resolved_ref = {
        resolved_input_ref: (input_ref, requested_fact_id)
        for input_ref, (requested_fact_id, resolved_input_ref) in known_row_sets.items()
        if resolved_input_ref
    }
    output: list[RowSetPopulationRequirement] = []
    for item in resolution.inputs:
        known = known_by_resolved_ref.get(item.input_ref)
        if known is None:
            continue
        memory_ids = item.row_set_memory_references()
        if len(memory_ids) != 1:
            continue
        memory_id = memory_ids[0]
        if active_ids and memory_id not in active_ids:
            continue
        input_ref, requested_fact_id = known
        output.append(
            RowSetPopulationRequirement(
                input_ref=input_ref,
                resolved_input_ref=item.input_ref,
                requested_fact_id=requested_fact_id,
                value_id=item.input_ref,
                memory_relation_id=memory_id,
            )
        )
    return tuple(output)


def _prior_scope_bound_params(candidate: JsonObject) -> tuple[dict[str, str], ...]:
    direct = tuple(item for item in _json_objects(candidate.get("bound_params")))
    if direct:
        return tuple(_prior_scope_bound_param(item) for item in direct)
    invocations = tuple(
        item for item in _json_objects(candidate.get("source_invocations"))
    )
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for invocation in invocations:
        for item in _json_objects(invocation.get("bound_params")):
            param = _prior_scope_bound_param(item)
            key = (param["param_id"], param["source"], param["value"])
            if key in seen:
                continue
            seen.add(key)
            output.append(param)
    return tuple(output)


def _json_objects(value: JsonValue | None) -> tuple[JsonObject, ...]:
    if isinstance(value, dict):
        return (value,)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _prior_scope_bound_param(item: dict[str, Any]) -> dict[str, str]:
    return {
        "param_id": str(item.get("param_id") or ""),
        "source": str(item.get("source") or "prior_scope"),
        "value": str(item.get("value") or ""),
    }
