"""Deterministic closed-key source-binding param ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeVar

from fervis.lookup.fact_plan.relations import EndpointParamBinding
from fervis.lookup.fact_plan.values import (
    FactValue,
    IdentityValuePayload,
    ValueKind,
    known_input_id_for_value,
)
from fervis.lookup.question_contract import (
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpressionFamily,
    RequestedFactGroupKey,
)
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.model import SourceBindingRequest, SourceFulfillment
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
)


_ParamMapValue = TypeVar("_ParamMapValue")


@dataclass(frozen=True)
class _KeyInputBinding:
    question_input_id: str
    value_id: str
    value: str
    identity_field: str
    proof_refs: tuple[str, ...]

    @classmethod
    def from_value(cls, question_input_id: str, value: FactValue) -> _KeyInputBinding:
        payload = value.payload
        if not isinstance(payload, IdentityValuePayload):
            raise ValueError("closed key binding requires identity value")
        return cls(
            question_input_id=question_input_id,
            value_id=value.id,
            value=payload.value,
            identity_field=payload.identity_field,
            proof_refs=value.proof_refs,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "question_input_id": self.question_input_id,
            "value_id": self.value_id,
            "value": self.value,
            "proof_refs": list(self.proof_refs),
        }


@dataclass(frozen=True)
class _ClosedKeyParamBinding:
    answer_output_id: str
    param_id: str
    identity_field: str
    key_input_bindings: tuple[_KeyInputBinding, ...]

    @property
    def param_binding_sets(self) -> tuple[tuple[EndpointParamBinding, ...], ...]:
        return tuple(
            (
                EndpointParamBinding(
                    param_id=self.param_id,
                    value=binding.value,
                    proof_refs=binding.proof_refs,
                ),
            )
            for binding in self.key_input_bindings
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "answer_output_id": self.answer_output_id,
            "param_id": self.param_id,
            "identity_field": self.identity_field,
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

    def model_visible_target_payload(
        self,
        target: SourceBindingTarget,
    ) -> dict[str, object]:
        payload = target.to_payload()
        binding = self._binding_for_target(target.binding_target_id)
        if binding is not None:
            payload["backend_owned_param_bindings"] = [binding.to_payload()]
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
        param_id = self._owned_param_id(target_id)
        if not param_id:
            return dict(values_by_param)
        return {
            candidate_param_id: value
            for candidate_param_id, value in values_by_param.items()
            if candidate_param_id != param_id
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
    ) -> tuple[tuple[EndpointParamBinding, ...], ...]:
        binding = self._binding_for_target(target_id)
        if binding is None:
            return ((),)
        return binding.param_binding_sets

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

    def _owned_param_id(self, target_id: str) -> str:
        binding = self._binding_for_target(target_id)
        return binding.param_id if binding is not None else ""

    def _owned_param_ids_by_candidate(self) -> dict[str, frozenset[str]]:
        grouped: dict[str, set[str]] = {}
        for target_binding in self._target_bindings_by_id.values():
            grouped.setdefault(target_binding.source_candidate_id, set()).add(
                target_binding.binding.param_id
            )
        return {
            candidate_id: frozenset(param_ids)
            for candidate_id, param_ids in grouped.items()
        }

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
    return ClosedKeyParamBindingIndex(
        _target_bindings_by_id={
            target.binding_target_id: _TargetClosedKeyBinding(
                target_id=target.binding_target_id,
                source_candidate_id=target.source_candidate_id,
                binding=binding,
            )
            for target in targets
            if target.source_candidate_id in candidates_by_id
            for binding in (
                _closed_key_param_binding(
                    request,
                    fact=_target_fact(target, facts_by_id),
                    target=target,
                    candidate=candidates_by_id[target.source_candidate_id],
                ),
            )
            if binding is not None
        },
    )


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
            output["params"] = [
                param
                for param in output.get("params") or ()
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
                identity_field=binding.identity_field,
            )
            for choice_id in (str(support_set.get("fulfillment_choice_id") or ""),)
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
                identity_field=binding.identity_field,
            )
            for support_set_id in (
                str(support_set.get("fulfillment_support_set_id") or ""),
            )
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
    identity_field = _shared_identity_field(bindings)
    if not identity_field:
        return None
    param_id = _candidate_identity_param_id(
        candidate,
        identity_field=identity_field,
        value_ids=frozenset(binding.value_id for binding in bindings),
    )
    if not param_id:
        return None
    if not _candidate_has_group_key_evidence(
        candidate,
        answer_output_id=group_key.id,
        identity_field=identity_field,
    ):
        return None
    return _ClosedKeyParamBinding(
        answer_output_id=group_key.id,
        param_id=param_id,
        identity_field=identity_field,
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
        and bool(value.payload.identity_field)
        and bool(value.payload.value)
    )


def _shared_identity_field(bindings: tuple[_KeyInputBinding, ...]) -> str:
    fields = {binding.identity_field for binding in bindings}
    return next(iter(fields)) if len(fields) == 1 else ""


def _candidate_identity_param_id(
    candidate: SourceCandidate,
    *,
    identity_field: str,
    value_ids: frozenset[str],
) -> str:
    params = tuple(
        param
        for param in candidate.params
        if _param_identity_field(param) == identity_field
        and value_ids <= _param_bindable_value_ids(param)
    )
    if len(params) != 1:
        return ""
    return str(params[0].get("param_id") or "")


def _param_identity_field(param: Mapping[str, object]) -> str:
    identity = param.get("identity")
    if not isinstance(identity, Mapping):
        return ""
    return str(identity.get("identity_field") or "")


def _param_bindable_value_ids(param: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        str(item.get("value") or "")
        for item in param.get("binding_values") or ()
        if isinstance(item, Mapping) and str(item.get("value") or "")
    )


def _candidate_has_group_key_evidence(
    candidate: SourceCandidate,
    *,
    answer_output_id: str,
    identity_field: str,
) -> bool:
    return bool(
        _closed_key_group_key_fulfillment_supports(
            candidate,
            answer_output_id=answer_output_id,
            identity_field=identity_field,
        )
    )


def _closed_key_group_key_fulfillment_supports(
    candidate: SourceCandidate,
    *,
    answer_output_id: str,
    identity_field: str,
) -> tuple[Mapping[str, object], ...]:
    fields_by_id = _candidate_fields_by_id(candidate)
    return tuple(
        support_set
        for support_set in _candidate_fulfillment_support_sets(candidate)
        if str(support_set.get("answer_output_id") or "") == answer_output_id
        and _support_set_has_group_key_identity(
            support_set,
            fields_by_id=fields_by_id,
            identity_field=identity_field,
        )
    )


def _support_set_has_group_key_identity(
    support_set: Mapping[str, object],
    *,
    fields_by_id: dict[str, Mapping[str, object]],
    identity_field: str,
) -> bool:
    return any(
        _group_key_evidence_matches_identity(
            item,
            fields_by_id=fields_by_id,
            identity_field=identity_field,
        )
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, Mapping)
        for item in slot.get("group_key_evidence") or ()
        if isinstance(item, Mapping)
    )


def _group_key_evidence_matches_identity(
    item: Mapping[str, object],
    *,
    fields_by_id: dict[str, Mapping[str, object]],
    identity_field: str,
) -> bool:
    if _evidence_identity_field(item, fields_by_id=fields_by_id) == identity_field:
        return True
    return str(item.get("field_id") or "") == identity_field


def _candidate_fields_by_id(
    candidate: SourceCandidate,
) -> dict[str, Mapping[str, object]]:
    return {
        field_id: field
        for field in candidate.fields
        if isinstance(field, Mapping)
        for field_id in (str(field.get("field_id") or field.get("id") or ""),)
        if field_id
    }


def _candidate_fulfillment_support_sets(
    candidate: SourceCandidate,
) -> tuple[Mapping[str, object], ...]:
    payload = candidate.payload or {}
    return tuple(
        support_set
        for support_set in payload.get("fulfillment_support_sets") or ()
        if isinstance(support_set, Mapping)
    )


def _evidence_identity_field(
    item: Mapping[str, object],
    *,
    fields_by_id: dict[str, Mapping[str, object]],
) -> str:
    identity = item.get("identity")
    if not isinstance(identity, Mapping):
        field = fields_by_id.get(str(item.get("field_id") or ""))
        identity = field.get("identity") if field is not None else None
    if not isinstance(identity, Mapping):
        return ""
    return str(identity.get("identity_field") or "")
