"""Endpoint param decision parsing."""

from __future__ import annotations

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    RelationInputOrigin,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    TimeComponent,
    ValueComponent,
    ValueKind,
)
from fervis.lookup.fact_planning.value_components import value_component
from fervis.lookup.relation_catalog.parameter_values import (
    parse_catalog_parameter_value,
)
from fervis.lookup.source_binding.model import AnswerPopulation
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.source_binding.candidates.model import (
    CandidateParameter,
    CandidateParamDecision,
    SourceCandidate,
)
from fervis.lookup.source_binding.param_values import canonical_param_value
from fervis.lookup.source_binding.param_binding_sets import (
    combine_param_binding_sets,
    parameter_binding_sets,
)
from fervis.lookup.source_binding.parser.types import (
    NormalizedParamDecision,
    ParamDecisionParse,
    PopulationChoiceSet,
)


__all__ = [
    "merged_param_bindings",
    "parse_param_decision_binding_sets",
]


def parse_param_decision_binding_sets(
    decisions: dict[str, NormalizedParamDecision],
    *,
    candidate: SourceCandidate,
    available_values: tuple[FactValue, ...],
    answer_population: AnswerPopulation,
    parameter_namespace: str,
    effective_param_ids: tuple[str, ...] | None = None,
    prebound_param_ids: tuple[str, ...] = (),
) -> ParamDecisionParse:
    params_by_id = {
        param.id: param for param in candidate.params if _param_is_model_bindable(param)
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
    if not params_by_id and not decisions:
        return ParamDecisionParse(binding_sets=((),))
    for param_id, decision in decisions.items():
        if param_id not in params_by_id:
            raise ValueError("source param decision references unknown param")
        if not decision.match_basis_explanation.strip():
            raise ValueError("source param decision requires match basis explanation")
        if not decision.population_intent.strip():
            raise ValueError(
                "source param decision requires non-empty population intent"
            )
        param = params_by_id[param_id]
        if param.choices and decision.population_choice_set is not None:
            choice_set = _population_choice_set(decision, param=param)
            output.append(
                parameter_binding_sets(
                    param_id=param_id,
                    value=choice_set.included_values,
                    param=param,
                    origin_kind=RelationInputOrigin.SEMANTIC_CONTROL,
                    parameter_id=f"{parameter_namespace}.param.{param_id}",
                )
            )
            continue
        decision_id = decision.param_decision_id or ""
        indexed_option = options_by_id.get(decision_id)
        if indexed_option is None:
            raise ValueError("source param decision references unknown option")
        option_param_id, option = indexed_option
        if option_param_id != param_id:
            raise ValueError("source param decision references mismatched param")
        option_kind = option.decision
        if option_kind == "use_default":
            if not param.has_default:
                raise ValueError(
                    "source param decision uses default but param has no default"
                )
            output.append(((),))
            continue
        if option_kind == "omit":
            output.append(((),))
            continue
        if option_kind != "bind":
            raise ValueError("unsupported source param decision")
        selected_value = option.value
        value: RuntimeValue = selected_value
        choices = param.choices
        if choices and value not in {str(choice) for choice in choices}:
            raise ValueError("source binding param value is not an available choice")
        proof_refs: tuple[str, ...] = ()
        value_id = ""
        value_component: ValueComponent | TimeComponent = ValueComponent.VALUE
        binding_values = param.binding_values
        if binding_values:
            allowed_value_ids = {item.value for item in binding_values}
            if value not in allowed_value_ids:
                raise ValueError("source binding param value is not bindable")
            value, proof_refs, value_id, value_component = _resolved_binding_value(
                selected_value,
                param=param,
                option=option,
                available_values=available_values,
            )
        else:
            value = parse_catalog_parameter_value(
                value,
                type_name=param.type,
                choices=tuple(str(choice) for choice in choices or ()),
            )
        output.append(
            parameter_binding_sets(
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
                    "" if value_id else f"{parameter_namespace}.param.{param_id}"
                ),
            )
        )
    prebound = set(prebound_param_ids)
    missing_param_ids = {
        param_id
        for param_id, param in params_by_id.items()
        if (
            param_id not in decisions
            and param_id not in prebound
            and param.requires_explicit_decision
        )
    }
    if missing_param_ids:
        raise ValueError("source binding missing explicit param decision")
    if not output:
        return ParamDecisionParse(binding_sets=((),))
    return ParamDecisionParse(binding_sets=combine_param_binding_sets(output))


def _population_choice_set(
    decision: NormalizedParamDecision,
    *,
    param: CandidateParameter,
) -> PopulationChoiceSet:
    if decision.param_decision_id:
        raise ValueError("choice params require population choice set")
    choice_set = decision.population_choice_set
    if choice_set is None:
        raise ValueError("choice params require population choice set")
    include_values = tuple(map(canonical_param_value, choice_set.included_values))
    exclude_values = tuple(map(canonical_param_value, choice_set.excluded_values))
    if not include_values:
        raise ValueError("population choice set requires included values")
    choices = set(param.choices)
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


def _param_decision_options_by_id(
    params_by_id: dict[str, CandidateParameter],
) -> dict[str, tuple[str, CandidateParamDecision]]:
    output: dict[str, tuple[str, CandidateParamDecision]] = {}
    for param_id, param in params_by_id.items():
        for option in param.decision_options:
            decision_id = option.id
            if not decision_id:
                continue
            output[decision_id] = (param_id, option)
    return output


def _param_is_model_bindable(param: CandidateParameter) -> bool:
    return bool(param.decision_options)


def _resolved_binding_value(
    value: str,
    *,
    param: CandidateParameter,
    option: CandidateParamDecision,
    available_values: tuple[FactValue, ...],
) -> tuple[RuntimeValue, tuple[str, ...], str, ValueComponent | TimeComponent]:
    values_by_id = {item.id: item for item in available_values}
    fact_value = values_by_id.get(value)
    if fact_value is None:
        if param.type == "boolean" and value in {"true", "false"}:
            return value == "true", (), "", ValueComponent.VALUE
        return value, (), "", ValueComponent.VALUE
    component = _value_component_from_option(option)
    resolved = value_component(fact_value, component)
    if fact_value.kind == ValueKind.IDENTITY_SET:
        return resolved, tuple(fact_value.proof_refs), fact_value.id, component
    return str(resolved), tuple(fact_value.proof_refs), fact_value.id, component


def _value_component_from_option(
    option: CandidateParamDecision,
) -> ValueComponent | TimeComponent:
    raw_component = option.value_component
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
