"""Endpoint param decision parsing."""

from __future__ import annotations

from itertools import product
from typing import Any

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    RelationInputOrigin,
)
from fervis.lookup.answer_program.values import FactValue, TimeComponent, ValueComponent, ValueKind
from fervis.lookup.fact_planning.value_components import value_component
from fervis.lookup.relation_catalog.parameter_values import (
    parse_catalog_parameter_value,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import AnswerPopulation
from fervis.lookup.source_binding.param_surface import param_has_default_value
from fervis.lookup.source_binding.param_values import canonical_param_value
from fervis.lookup.source_binding.parser.types import ParamDecisionParse, PopulationChoiceSet
from fervis.lookup.source_binding.parser_common import _dict, _optional_text, _text


__all__ = [
    "merged_param_bindings",
    "normalize_param_decisions",
    "parse_param_decision_binding_sets",
]


def parse_param_decision_binding_sets(
    raw_decisions: Any,
    *,
    candidate: Any,
    available_values: tuple[FactValue, ...],
    answer_population: AnswerPopulation,
    parameter_namespace: str,
    effective_param_ids: tuple[str, ...] | None = None,
) -> ParamDecisionParse:
    params_by_id = {
        str(param.get("param_id") or ""): param
        for param in candidate.params
        if isinstance(param, dict) and _param_is_model_bindable(param)
    }
    if effective_param_ids is not None:
        effective = set(effective_param_ids)
        params_by_id = {
            param_id: param
            for param_id, param in params_by_id.items()
            if param_id in effective
        }
    options_by_id = _param_decision_options_by_id(params_by_id)
    output: list[tuple[tuple[DraftEndpointParamBinding, ...], ...]] = []
    normalized_decisions = normalize_param_decisions(raw_decisions)
    if not params_by_id and not normalized_decisions:
        return ParamDecisionParse(binding_sets=((),))
    for param_id, raw in normalized_decisions.items():
        if param_id not in params_by_id:
            raise ValueError("source param decision references unknown param")
        match_basis_explanation = _text(raw.get("match_basis_explanation")).strip()
        if not match_basis_explanation:
            raise ValueError("source param decision requires match basis explanation")
        _validate_param_population_intent(raw)
        param = params_by_id[param_id]
        if param.get("choices") and "population_choice_set" in raw:
            choice_set = _population_choice_set(raw, param=param)
            output.append(
                _param_binding_sets(
                    param_id=param_id,
                    value=choice_set.included_values,
                    param=param,
                    origin_kind=RelationInputOrigin.SEMANTIC_CONTROL,
                    parameter_id=f"{parameter_namespace}.param.{param_id}",
                )
            )
            continue
        decision_id = _text(raw.get("param_decision_id"))
        option = options_by_id.get(decision_id)
        if option is None:
            raise ValueError("source param decision references unknown option")
        if str(option.get("param_id") or "") != param_id:
            raise ValueError("source param decision references mismatched param")
        decision = str(option.get("decision") or "")
        if decision == "use_default":
            if not param_has_default_value(param):
                raise ValueError(
                    "source param decision uses default but param has no default"
                )
            output.append(((),))
            continue
        if decision == "omit":
            output.append(((),))
            continue
        if decision != "bind":
            raise ValueError("unsupported source param decision")
        value = str(option.get("value") or "")
        choices = param.get("choices")
        if choices and value not in {str(choice) for choice in choices}:
            raise ValueError("source binding param value is not an available choice")
        proof_refs: tuple[str, ...] = ()
        value_id = ""
        value_component = ValueComponent.VALUE
        binding_values = param.get("binding_values")
        if binding_values:
            allowed_value_ids = {
                str(item.get("value") or "")
                for item in binding_values
                if isinstance(item, dict)
            }
            if value not in allowed_value_ids:
                raise ValueError("source binding param value is not bindable")
            value, proof_refs, value_id, value_component = _resolved_binding_value(
                value,
                param=param,
                option=option,
                available_values=available_values,
            )
        else:
            value = parse_catalog_parameter_value(
                value,
                type_name=str(param.get("type") or ""),
                choices=tuple(str(choice) for choice in choices or ()),
            )
        output.append(
            _param_binding_sets(
                param_id=param_id,
                value=value,
                param=param,
                proof_refs=proof_refs,
                origin_kind=(
                    RelationInputOrigin.QUESTION_INPUT
                    if value_id
                    else RelationInputOrigin.SEMANTIC_CONTROL
                ),
                value_id=value_id,
                value_component=value_component.value,
                parameter_id=(
                    ""
                    if value_id
                    else f"{parameter_namespace}.param.{param_id}"
                ),
            )
        )
    missing_param_ids = {
        param_id
        for param_id, param in params_by_id.items()
        if param_id not in normalized_decisions
        and _param_requires_explicit_decision(param)
    }
    if missing_param_ids:
        raise ValueError("source binding missing explicit param decision")
    if not output:
        return ParamDecisionParse(binding_sets=((),))
    return ParamDecisionParse(
        binding_sets=tuple(
            tuple(binding for group in groups for binding in group)
            for groups in product(*output)
        ),
    )


def _population_choice_set(
    raw: dict[str, Any],
    *,
    param: dict[str, Any],
) -> PopulationChoiceSet:
    if "param_decision_id" in raw:
        raise ValueError("choice params require population choice set")
    choice_set = _dict(raw.get("population_choice_set"), "population_choice_set")
    include_values = tuple(
        canonical_param_value(value) for value in choice_set.get("include_values") or ()
    )
    exclude_values = tuple(
        canonical_param_value(value) for value in choice_set.get("exclude_values") or ()
    )
    if not include_values:
        raise ValueError("population choice set requires included values")
    choices = {canonical_param_value(choice) for choice in param.get("choices") or ()}
    include_set = set(include_values)
    exclude_set = set(exclude_values)
    if include_set & exclude_set:
        raise ValueError("population choice set cannot overlap")
    if include_set | exclude_set != choices:
        raise ValueError("population choice set must cover every choice")
    if any(value not in choices for value in include_values):
        raise ValueError("population choice set includes unknown choice")
    return PopulationChoiceSet(
        included_values=include_values,
        excluded_values=exclude_values,
    )


def _param_binding_sets(
    *,
    param_id: str,
    value: object,
    param: dict[str, Any],
    proof_refs: tuple[str, ...] = (),
    origin_kind: RelationInputOrigin,
    value_id: str = "",
    value_component: str = "value",
    parameter_id: str = "",
) -> tuple[tuple[DraftEndpointParamBinding, ...], ...]:
    if isinstance(value, tuple) and not _param_accepts_collection(param):
        return tuple(
            (
                DraftEndpointParamBinding(
                    param_id=param_id,
                    value=value,
                    origin_kind=origin_kind,
                    value_id=value_id,
                    value_component=value_component,
                    value_item_index=index,
                    parameter_id=parameter_id,
                    proof_refs=proof_refs,
                ),
            )
            for index, _item in enumerate(value)
        )
    return (
        (
            DraftEndpointParamBinding(
                param_id=param_id,
                value=value,
                origin_kind=origin_kind,
                value_id=value_id,
                value_component=value_component,
                parameter_id=parameter_id,
                proof_refs=proof_refs,
            ),
        ),
    )


def _param_accepts_collection(param: dict[str, Any]) -> bool:
    return str(param.get("type") or "").strip() in {"array", "list"}


def normalize_param_decisions(
    raw_decisions: Any,
    *,
    parse_provider_output: bool = False,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if isinstance(raw_decisions, dict):
        for raw_param_id, raw_value in raw_decisions.items():
            param_id = str(raw_param_id)
            if param_id in output:
                raise ValueError("duplicate source param decision")
            output[param_id] = (
                vars(provider_output.ParamDecisionOutput.parse(raw_value))
                if parse_provider_output
                else _dict(raw_value, f"param_decisions.{param_id}")
            )
        return output
    raise ValueError("param_decisions must be an object")


def _param_decision_options_by_id(
    params_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for param_id, param in params_by_id.items():
        for option in param.get("decision_options") or ():
            if not isinstance(option, dict):
                continue
            decision_id = str(option.get("param_decision_id") or "")
            if not decision_id:
                continue
            output[decision_id] = {**option, "param_id": param_id}
    return output


def _validate_param_population_intent(
    raw: dict[str, Any],
) -> str:
    if "population_intent" not in raw:
        raise ValueError("source param decision requires population intent")
    population_intent = _optional_text(raw.get("population_intent"))
    if not population_intent:
        raise ValueError("source param decision requires non-empty population intent")
    return population_intent


def _param_requires_explicit_decision(param: dict[str, Any]) -> bool:
    return bool(param.get("required")) or bool(param.get("choices"))


def _param_is_model_bindable(param: dict[str, Any]) -> bool:
    decision_options = param.get("decision_options")
    return isinstance(decision_options, list) and bool(decision_options)


def _resolved_binding_value(
    value: str,
    *,
    param: dict[str, Any],
    option: dict[str, Any],
    available_values: tuple[FactValue, ...],
) -> tuple[object, tuple[str, ...], str, ValueComponent | TimeComponent]:
    values_by_id = {item.id: item for item in available_values}
    fact_value = values_by_id.get(value)
    if fact_value is None:
        if str(param.get("type") or "") == "boolean" and value in {"true", "false"}:
            return value == "true", (), "", ValueComponent.VALUE
        return value, (), "", ValueComponent.VALUE
    component = _value_component_from_option(option)
    resolved = value_component(fact_value, component)
    if fact_value.kind == ValueKind.IDENTITY_SET:
        return resolved, tuple(fact_value.proof_refs), fact_value.id, component
    return str(resolved), tuple(fact_value.proof_refs), fact_value.id, component


def _value_component_from_option(
    option: dict[str, Any],
) -> ValueComponent | TimeComponent:
    raw_component = str(option.get("value_component") or "").strip()
    if raw_component == TimeComponent.START.value:
        return TimeComponent.START
    if raw_component == TimeComponent.END.value:
        return TimeComponent.END
    if raw_component == TimeComponent.INSTANT.value:
        return TimeComponent.INSTANT
    return ValueComponent.VALUE


def merged_param_bindings(
    applied: tuple[DraftEndpointParamBinding, ...],
    model_authored: tuple[DraftEndpointParamBinding, ...],
) -> tuple[DraftEndpointParamBinding, ...]:
    output: list[DraftEndpointParamBinding] = []
    seen: set[str] = set()
    for binding in (*applied, *model_authored):
        if binding.param_id in seen:
            raise ValueError("duplicate source param binding")
        seen.add(binding.param_id)
        output.append(binding)
    return tuple(output)
