"""Deterministic closed-key source-binding param ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeVar

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    RelationInputOrigin,
    SourceAppliedFilter,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentityValuePayload,
    ValueKind,
    known_input_id_for_value,
)
from fervis.lookup.canonical_data import EntityKeyValue
from fervis.lookup.question_contract import (
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpressionFamily,
    RequestedFactGroupKey,
)
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.candidates.model import CandidateParameter
from fervis.lookup.source_binding.grounded_params import grounded_param_bindings
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    build_row_source_catalog,
    row_sources_for_read_id,
)
from fervis.lookup.source_binding.candidates.contracts import (
    FulfillmentSupportSet,
    entity_evidence_entity_kind,
    entity_evidence_key_id,
)
from fervis.lookup.source_binding.model import SourceBindingRequest, SourceFulfillment
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
)


_ParamMapValue = TypeVar("_ParamMapValue")


@dataclass(frozen=True)
class _KeyInputBinding:
    question_input_id: str
    value_id: str
    key: EntityKeyValue
    proof_refs: tuple[str, ...]

    @classmethod
    def from_value(cls, question_input_id: str, value: FactValue) -> _KeyInputBinding:
        payload = value.payload
        if not isinstance(payload, IdentityValuePayload):
            raise ValueError("closed key binding requires identity value")
        return cls(
            question_input_id=question_input_id,
            value_id=value.id,
            key=payload.key,
            proof_refs=value.proof_refs,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "question_input_id": self.question_input_id,
            "value_id": self.value_id,
            "key_components": {
                component.component_id: str(component.value)
                for component in self.key.components
            },
            "proof_refs": list(self.proof_refs),
        }


@dataclass(frozen=True)
class _ClosedKeyParamBinding:
    answer_output_id: str
    entity_kind: str
    key_id: str
    params_by_component_id: tuple[tuple[str, str], ...]
    key_input_bindings: tuple[_KeyInputBinding, ...]

    @property
    def question_input_ids(self) -> frozenset[str]:
        return frozenset(
            binding.question_input_id for binding in self.key_input_bindings
        )

    @property
    def param_binding_sets(self) -> tuple[tuple[DraftEndpointParamBinding, ...], ...]:
        return tuple(
            tuple(
                DraftEndpointParamBinding(
                    param_id=param_id,
                    value=binding.key.component_value(component_id),
                    origin_kind=RelationInputOrigin.QUESTION_INPUT,
                    value_id=binding.value_id,
                    proof_refs=binding.proof_refs,
                )
                for component_id, param_id in self.params_by_component_id
            )
            for binding in self.key_input_bindings
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "answer_output_id": self.answer_output_id,
            "entity_kind": self.entity_kind,
            "key_id": self.key_id,
            "params_by_component_id": dict(self.params_by_component_id),
            "key_input_bindings": [
                binding.to_payload() for binding in self.key_input_bindings
            ],
        }


@dataclass(frozen=True)
class _TargetClosedKeyBinding:
    target_id: str
    source_candidate_id: str
    binding: _ClosedKeyParamBinding


@dataclass(frozen=True)
class ClosedKeyParamBindingIndex:
    _target_bindings_by_id: dict[str, _TargetClosedKeyBinding]
    _candidate_id_by_target_id: dict[str, str]
    _grounded_bindings_by_target_id: dict[
        str,
        tuple[DraftEndpointParamBinding, ...],
    ]

    def model_visible_target_payload(
        self,
        target: SourceBindingTarget,
    ) -> dict[str, object]:
        payload = target.to_payload()
        binding = self._binding_for_target(target.binding_target_id)
        backend_bindings: list[dict[str, object]] = []
        if binding is not None:
            backend_bindings.append(binding.to_payload())
        backend_bindings.extend(
            _grounded_binding_payload(item)
            for item in self._grounded_bindings(target.binding_target_id)
        )
        if backend_bindings:
            payload["backend_owned_param_bindings"] = backend_bindings
        return payload

    def model_visible_candidate_payload(
        self,
        payload: dict[str, object],
    ) -> dict[str, object]:
        param_ids_by_candidate = self._owned_param_ids_by_candidate()
        if not param_ids_by_candidate:
            return payload
        scrubbed = _model_visible_candidate_payload(payload, param_ids_by_candidate)
        if not isinstance(scrubbed, dict):
            raise ValueError("source binding candidate payload must be an object")
        return scrubbed

    def model_visible_param_map(
        self,
        target_id: str,
        values_by_param: Mapping[str, _ParamMapValue],
    ) -> dict[str, _ParamMapValue]:
        owned_param_ids = self.owned_param_ids(target_id)
        if not owned_param_ids:
            return dict(values_by_param)
        return {
            candidate_param_id: value
            for candidate_param_id, value in values_by_param.items()
            if candidate_param_id not in owned_param_ids
        }

    def model_visible_fulfillment_supports(
        self,
        candidate: SourceCandidate,
        *,
        target: SourceBindingTarget,
        candidate_fulfillment_supports: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        return {
            answer_output_id: self._fulfillment_choice_ids(
                candidate,
                target_id=target.binding_target_id,
                answer_output_id=answer_output_id,
                choice_ids=choice_ids,
            )
            for answer_output_id, choice_ids in candidate_fulfillment_supports.items()
            if answer_output_id in target.answer_output_ids
        }

    def backend_param_binding_sets(
        self,
        target_id: str,
    ) -> tuple[tuple[DraftEndpointParamBinding, ...], ...]:
        binding = self._binding_for_target(target_id)
        closed_sets = binding.param_binding_sets if binding is not None else ((),)
        grounded = self._grounded_bindings(target_id)
        return tuple((*closed, *grounded) for closed in closed_sets)

    def owned_param_ids(self, target_id: str) -> frozenset[str]:
        binding = self._binding_for_target(target_id)
        closed_param_ids = (
            tuple(param_id for _, param_id in binding.params_by_component_id)
            if binding is not None
            else ()
        )
        grounded_param_ids = tuple(
            binding.param_id for binding in self._grounded_bindings(target_id)
        )
        return frozenset((*closed_param_ids, *grounded_param_ids))

    def source_level_applied_filters(
        self,
        target_id: str,
        applied_filters: tuple[SourceAppliedFilter, ...],
    ) -> tuple[SourceAppliedFilter, ...]:
        binding = self._binding_for_target(target_id)
        if binding is None:
            return applied_filters
        return tuple(
            applied_filter
            for applied_filter in applied_filters
            if applied_filter.known_input_id not in binding.question_input_ids
        )

    def require_compatible_fulfillments(
        self,
        target_id: str,
        *,
        candidate: SourceCandidate,
        fulfillments: tuple[SourceFulfillment, ...],
    ) -> None:
        binding = self._binding_for_target(target_id)
        if binding is None:
            return
        selected_support_set_ids = {
            fulfillment.fulfillment_support_set_id
            for fulfillment in fulfillments
            if fulfillment.answer_output_id == binding.answer_output_id
            and fulfillment.fulfillment_support_set_id
        }
        if not selected_support_set_ids:
            return
        compatible_support_set_ids = set(
            _closed_key_group_key_fulfillment_support_set_ids(candidate, binding)
        )
        if selected_support_set_ids <= compatible_support_set_ids:
            return
        raise ValueError(
            "backend-owned group key param does not match selected fulfillment"
        )

    def _binding_for_target(self, target_id: str) -> _ClosedKeyParamBinding | None:
        target_binding = self._target_bindings_by_id.get(target_id)
        return target_binding.binding if target_binding is not None else None

    def _owned_param_ids_by_candidate(self) -> dict[str, frozenset[str]]:
        grouped: dict[str, set[str]] = {}
        for target_id, candidate_id in self._candidate_id_by_target_id.items():
            grouped.setdefault(candidate_id, set()).update(self.owned_param_ids(target_id))
        return {
            candidate_id: frozenset(param_ids)
            for candidate_id, param_ids in grouped.items()
        }

    def _grounded_bindings(
        self,
        target_id: str,
    ) -> tuple[DraftEndpointParamBinding, ...]:
        return self._grounded_bindings_by_target_id.get(target_id, ())

    def _fulfillment_choice_ids(
        self,
        candidate: SourceCandidate,
        *,
        target_id: str,
        answer_output_id: str,
        choice_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        binding = self._binding_for_target(target_id)
        if binding is None or answer_output_id != binding.answer_output_id:
            return choice_ids
        compatible = set(
            _closed_key_group_key_fulfillment_choice_ids(candidate, binding)
        )
        return tuple(choice_id for choice_id in choice_ids if choice_id in compatible)


def closed_key_param_binding_index(
    request: SourceBindingRequest,
    *,
    targets: tuple[SourceBindingTarget, ...],
    candidates_by_id: Mapping[str, SourceCandidate],
) -> ClosedKeyParamBindingIndex:
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    row_sources = build_row_source_catalog(request.relation_catalog)
    closed_bindings: dict[str, _TargetClosedKeyBinding] = {}
    grounded_bindings: dict[str, tuple[DraftEndpointParamBinding, ...]] = {}
    candidate_ids: dict[str, str] = {}
    for target in targets:
        candidate = candidates_by_id.get(target.source_candidate_id)
        if candidate is None:
            continue
        target_id = target.binding_target_id
        candidate_ids[target_id] = target.source_candidate_id
        closed = _closed_key_param_binding(
            request,
            fact=_target_fact(target, facts_by_id),
            target=target,
            candidate=candidate,
        )
        excluded_param_ids: frozenset[str] = frozenset()
        if closed is not None:
            closed_bindings[target_id] = _TargetClosedKeyBinding(
                target_id=target_id,
                source_candidate_id=target.source_candidate_id,
                binding=closed,
            )
            excluded_param_ids = frozenset(
                param_id for _, param_id in closed.params_by_component_id
            )
        row_source = _candidate_row_source(candidate, row_sources=row_sources)
        if row_source is None:
            continue
        bindings = grounded_param_bindings(
            available_values=request.available_values,
            available_value_uses=request.available_value_uses,
            row_source=row_source,
            requested_fact_id=target.requested_fact_id,
            excluded_param_ids=excluded_param_ids,
        )
        if bindings:
            grounded_bindings[target_id] = bindings
    return ClosedKeyParamBindingIndex(
        _target_bindings_by_id=closed_bindings,
        _candidate_id_by_target_id=candidate_ids,
        _grounded_bindings_by_target_id=grounded_bindings,
    )


def _candidate_row_source(
    candidate: SourceCandidate,
    *,
    row_sources: RowSourceCatalog,
) -> RowSource | None:
    source = candidate.source
    if source is None:
        return None
    if source.read_id:
        candidates = row_sources_for_read_id(source.read_id, row_sources=row_sources)
        return candidates[0] if candidates else None
    return row_sources.find(source.row_source_id)


def _grounded_binding_payload(
    binding: DraftEndpointParamBinding,
) -> dict[str, object]:
    return {
        "param_id": binding.param_id,
        "value_id": binding.value_id,
        "proof_refs": list(binding.proof_refs),
    }


def _model_visible_candidate_payload(
    value: object,
    param_ids_by_candidate: dict[str, frozenset[str]],
) -> object:
    if isinstance(value, dict):
        output = {
            key: _model_visible_candidate_payload(raw_value, param_ids_by_candidate)
            for key, raw_value in value.items()
        }
        candidate_id = str(output.get("source_candidate_id") or "")
        owned_param_ids = param_ids_by_candidate.get(candidate_id, frozenset())
        if owned_param_ids:
            raw_params = output.get("params")
            params = raw_params if isinstance(raw_params, (list, tuple)) else ()
            output["params"] = [
                param
                for param in params
                if isinstance(param, dict)
                and str(param.get("param_id") or "") not in owned_param_ids
            ]
        return output
    if isinstance(value, list):
        return [
            _model_visible_candidate_payload(item, param_ids_by_candidate)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _model_visible_candidate_payload(item, param_ids_by_candidate)
            for item in value
        )
    return value


def _closed_key_group_key_fulfillment_choice_ids(
    candidate: SourceCandidate,
    binding: _ClosedKeyParamBinding,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            choice_id
            for support_set in _closed_key_group_key_fulfillment_supports(
                candidate,
                answer_output_id=binding.answer_output_id,
                entity_kind=binding.entity_kind,
                key_id=binding.key_id,
                key_component_ids=tuple(
                    component_id for component_id, _ in binding.params_by_component_id
                ),
            )
            for choice_id in (support_set.fulfillment_choice_id,)
            if choice_id
        )
    )


def _closed_key_group_key_fulfillment_support_set_ids(
    candidate: SourceCandidate,
    binding: _ClosedKeyParamBinding,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            support_set_id
            for support_set in _closed_key_group_key_fulfillment_supports(
                candidate,
                answer_output_id=binding.answer_output_id,
                entity_kind=binding.entity_kind,
                key_id=binding.key_id,
                key_component_ids=tuple(
                    component_id for component_id, _ in binding.params_by_component_id
                ),
            )
            for support_set_id in (support_set.fulfillment_support_set_id,)
            if support_set_id
        )
    )


def _closed_key_param_binding(
    request: SourceBindingRequest,
    *,
    fact: RequestedFact,
    target: SourceBindingTarget,
    candidate: SourceCandidate,
) -> _ClosedKeyParamBinding | None:
    group_key = _closed_key_group_key(fact, target=target)
    if group_key is None:
        return None
    bindings = _key_input_bindings(
        request,
        requested_fact_id=target.requested_fact_id,
        question_input_refs=group_key.question_input_refs,
    )
    identity_contract = _shared_identity_contract(bindings)
    if identity_contract is None:
        return None
    entity_kind, key_id, key_component_ids = identity_contract
    params_by_component_id = _candidate_identity_params(
        candidate,
        entity_kind=entity_kind,
        key_id=key_id,
        key_component_ids=key_component_ids,
        value_ids=frozenset(binding.value_id for binding in bindings),
    )
    if not params_by_component_id:
        return None
    if not _candidate_has_entity_evidence(
        candidate,
        answer_output_id=group_key.id,
        entity_kind=entity_kind,
        key_id=key_id,
        key_component_ids=key_component_ids,
    ):
        return None
    return _ClosedKeyParamBinding(
        answer_output_id=group_key.id,
        entity_kind=entity_kind,
        key_id=key_id,
        params_by_component_id=params_by_component_id,
        key_input_bindings=bindings,
    )


def _closed_key_group_key(
    fact: RequestedFact,
    *,
    target: SourceBindingTarget,
) -> RequestedFactGroupKey | None:
    if (
        fact.answer_expression is None
        or fact.answer_expression.family
        != RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE
        or fact.answer_expression.group_key is None
    ):
        return None
    group_key = fact.answer_expression.group_key
    if group_key.id not in target.answer_output_ids:
        return None
    if group_key.domain != GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS:
        return None
    return group_key


def _target_fact(
    target: SourceBindingTarget,
    facts_by_id: Mapping[str, RequestedFact],
) -> RequestedFact:
    try:
        return facts_by_id[target.requested_fact_id]
    except KeyError as exc:
        raise ValueError(
            "source binding target references unknown requested fact"
        ) from exc


def _key_input_bindings(
    request: SourceBindingRequest,
    *,
    requested_fact_id: str,
    question_input_refs: tuple[str, ...],
) -> tuple[_KeyInputBinding, ...]:
    values_by_input_ref = _identity_values_by_input_ref(
        request.available_values,
        requested_fact_id=requested_fact_id,
    )
    bindings: list[_KeyInputBinding] = []
    for input_ref in question_input_refs:
        values = values_by_input_ref.get(input_ref, ())
        if len(values) != 1:
            return ()
        bindings.append(_KeyInputBinding.from_value(input_ref, values[0]))
    return tuple(bindings)


def _identity_values_by_input_ref(
    values: tuple[FactValue, ...],
    *,
    requested_fact_id: str,
) -> dict[str, tuple[FactValue, ...]]:
    grouped: dict[str, list[FactValue]] = {}
    for value in values:
        input_ref = known_input_id_for_value(value)
        if (
            input_ref
            and _value_applies_to_fact(value, requested_fact_id)
            and _is_scalar_identity_value(value)
        ):
            grouped.setdefault(input_ref, []).append(value)
    return {input_ref: tuple(items) for input_ref, items in grouped.items()}


def _value_applies_to_fact(value: FactValue, requested_fact_id: str) -> bool:
    return (
        not value.applies_to_requested_fact_ids
        or requested_fact_id in value.applies_to_requested_fact_ids
    )


def _is_scalar_identity_value(value: FactValue) -> bool:
    return (
        value.kind == ValueKind.IDENTITY
        and isinstance(value.payload, IdentityValuePayload)
        and bool(value.payload.key_id)
        and bool(value.payload.key.components)
    )


def _shared_identity_contract(
    bindings: tuple[_KeyInputBinding, ...],
) -> tuple[str, str, tuple[str, ...]] | None:
    contracts = {
        (
            binding.key.entity_kind,
            binding.key.key_id,
            tuple(component.component_id for component in binding.key.components),
        )
        for binding in bindings
    }
    return next(iter(contracts)) if len(contracts) == 1 else None


def _candidate_identity_params(
    candidate: SourceCandidate,
    *,
    entity_kind: str,
    key_id: str,
    key_component_ids: tuple[str, ...],
    value_ids: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    output: list[tuple[str, str]] = []
    for component_id in key_component_ids:
        params = tuple(
            param
            for param in candidate.params
            if _param_matches_closed_key_identity(
                param,
                entity_kind=entity_kind,
                key_id=key_id,
                key_component_id=component_id,
                value_ids=value_ids,
            )
        )
        if len(params) != 1:
            return ()
        output.append((component_id, params[0].id))
    return tuple(output)


def _param_entity_target(param: CandidateParameter) -> tuple[str, str, str] | None:
    target = param.entity_target
    if target is None:
        return None
    entity_kind = target.entity_kind
    key_id = target.key_id
    component_id = target.component_id
    if not entity_kind or not key_id or not component_id:
        return None
    return entity_kind, key_id, component_id


def _param_bindable_value_ids(param: CandidateParameter) -> frozenset[str]:
    return frozenset(item.value for item in param.binding_values if item.value)


def _candidate_has_entity_evidence(
    candidate: SourceCandidate,
    *,
    answer_output_id: str,
    entity_kind: str,
    key_id: str,
    key_component_ids: tuple[str, ...],
) -> bool:
    return bool(
        _closed_key_group_key_fulfillment_supports(
            candidate,
            answer_output_id=answer_output_id,
            entity_kind=entity_kind,
            key_id=key_id,
            key_component_ids=key_component_ids,
        )
    )


def _closed_key_group_key_fulfillment_supports(
    candidate: SourceCandidate,
    *,
    answer_output_id: str,
    entity_kind: str,
    key_id: str,
    key_component_ids: tuple[str, ...],
) -> tuple[FulfillmentSupportSet, ...]:
    return tuple(
        support_set
        for support_set in _candidate_fulfillment_support_sets(candidate)
        if support_set.answer_output_id == answer_output_id
        and _support_set_has_entity_key(
            support_set,
            entity_kind=entity_kind,
            key_id=key_id,
            key_component_ids=key_component_ids,
        )
    )


def _support_set_has_entity_key(
    support_set: FulfillmentSupportSet,
    *,
    entity_kind: str,
    key_id: str,
    key_component_ids: tuple[str, ...],
) -> bool:
    return any(
        entity_evidence_entity_kind(item) == entity_kind
        and entity_evidence_key_id(item) == key_id
        and set(key_component_ids)
        <= {component.component_id for component in item.components}
        for slot in support_set.fulfillment_slots
        for item in slot.entity_evidence
    )

def _param_matches_closed_key_identity(
    param: CandidateParameter,
    *,
    entity_kind: str,
    key_id: str,
    key_component_id: str,
    value_ids: frozenset[str],
) -> bool:
    param_identity_matches = _param_entity_target(param) == (
        entity_kind,
        key_id,
        key_component_id,
    )
    param_can_bind_all_key_values = value_ids <= _param_bindable_value_ids(param)
    return param_identity_matches and param_can_bind_all_key_values


def _candidate_fulfillment_support_sets(
    candidate: SourceCandidate,
) -> tuple[FulfillmentSupportSet, ...]:
    return candidate.fulfillment_support_sets
