"""Provider schema for source binding."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.fact_plan.fact_plan import (
    BlockedFactBasis,
    MissingCatalogInputKind,
    PlanOutcomeKind,
)
from fervis.lookup.question_contract import (
    NormalInstanceExplicitOverrideReason,
    NormalInstanceExcludedStateRole,
)
from fervis.lookup.source_binding.normal_instance_roles import (
    NORMAL_INSTANCE_NO_EXCLUDED_ROLE,
    NORMAL_INSTANCE_UNKNOWN_EXCLUDED_ROLE,
)
from fervis.lookup.source_binding.metric_fit import (
    METRIC_FIT_DECISIONS,
)
from fervis.lookup.source_binding import provider_contract as provider_output

_POPULATION_TEST_EFFECT_SCHEMA = {
    "enum": [
        "SATISFIES_TEST",
        "CONFLICTS_WITH_TEST",
        "DOES_NOT_DECIDE_TEST",
        "UNKNOWN_TEST_EFFECT",
    ]
}


@dataclass(frozen=True)
class _SourceBindingSchemaScope:
    required_catalog_input_ids: tuple[str, ...]
    required_catalog_choice_input_ids: tuple[str, ...]
    target_param_decision_ids_by_param: dict[str, dict[str, tuple[str, ...]]]
    target_finite_choice_values: dict[str, dict[str, tuple[str, ...]]]
    target_row_predicate_values: dict[str, dict[str, tuple[str, ...]]]
    target_finite_choice_test_ids: dict[str, dict[str, tuple[str, ...]]]
    target_finite_choice_normal_instance_test_ids: dict[
        str, dict[str, tuple[str, ...]]
    ]
    target_row_predicate_test_ids: dict[str, dict[str, tuple[str, ...]]]
    target_population_roles: dict[str, tuple[dict[str, object], ...]]
    target_requested_fact_ids: dict[str, str]
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]]
    target_fulfillment_support_set_ids_by_answer_output: dict[
        str, dict[str, tuple[str, ...]]
    ]
    target_required_fulfillment_answer_output_ids: dict[str, tuple[str, ...]]
    target_population_binding_ids: dict[str, tuple[str, ...]]

    @property
    def binding_target_ids(self) -> tuple[str, ...]:
        return tuple(self.target_param_decision_ids_by_param)

    @property
    def requested_fact_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.target_requested_fact_ids.values()))


def _handle_schema() -> dict[str, object]:
    return {"type": "string", "minLength": 1}


def _strict_object(
    properties: dict[str, object],
    *,
    required: tuple[str, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }


def _variant_schema(variants: tuple[dict[str, object], ...]) -> dict[str, object]:
    if len(variants) == 1:
        return variants[0]
    return {"oneOf": list(variants)}


def build_source_binding_schema(
    *,
    required_catalog_input_ids: tuple[str, ...] = (),
    required_catalog_choice_input_ids: tuple[str, ...] = (),
    target_param_decision_ids_by_param: dict[str, dict[str, tuple[str, ...]]],
    target_finite_choice_values: dict[str, dict[str, tuple[str, ...]]],
    target_row_predicate_values: dict[str, dict[str, tuple[str, ...]]],
    target_finite_choice_test_ids: dict[str, dict[str, tuple[str, ...]]],
    target_finite_choice_normal_instance_test_ids: dict[
        str,
        dict[str, tuple[str, ...]],
    ],
    target_row_predicate_test_ids: dict[str, dict[str, tuple[str, ...]]],
    target_population_roles: dict[str, tuple[dict[str, object], ...]],
    target_requested_fact_ids: dict[str, str],
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
    target_fulfillment_support_set_ids_by_answer_output: dict[
        str, dict[str, tuple[str, ...]]
    ],
    target_required_fulfillment_answer_output_ids: dict[str, tuple[str, ...]],
    target_population_binding_ids: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    scope = _SourceBindingSchemaScope(
        required_catalog_input_ids=required_catalog_input_ids,
        required_catalog_choice_input_ids=required_catalog_choice_input_ids,
        target_param_decision_ids_by_param=target_param_decision_ids_by_param,
        target_finite_choice_values=target_finite_choice_values,
        target_row_predicate_values=target_row_predicate_values,
        target_finite_choice_test_ids=target_finite_choice_test_ids,
        target_finite_choice_normal_instance_test_ids=(
            target_finite_choice_normal_instance_test_ids
        ),
        target_row_predicate_test_ids=target_row_predicate_test_ids,
        target_population_roles=target_population_roles,
        target_requested_fact_ids=target_requested_fact_ids,
        metric_evidence_ids_by_requested_fact=metric_evidence_ids_by_requested_fact,
        target_fulfillment_support_set_ids_by_answer_output=(
            target_fulfillment_support_set_ids_by_answer_output
        ),
        target_required_fulfillment_answer_output_ids=(
            target_required_fulfillment_answer_output_ids
        ),
        target_population_binding_ids=target_population_binding_ids or {},
    )
    outcome_schema = _source_binding_outcome_schema(scope)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"outcome": outcome_schema},
        "required": ["outcome"],
        "modelSchemas": {"outcome": outcome_schema},
    }


def _source_binding_outcome_schema(scope: _SourceBindingSchemaScope) -> dict[str, object]:
    variants: list[dict[str, object]] = []
    if scope.binding_target_ids:
        variants.append(_source_binding_plan_schema(scope))
        variants.append(
            _impossible_schema(
                allowed_bases=(BlockedFactBasis.POLICY_ACCESS.value,),
                requested_fact_ids=scope.requested_fact_ids,
            )
        )
    clarification = _clarification_schema(
        required_catalog_input_ids=scope.required_catalog_input_ids,
        required_catalog_choice_input_ids=scope.required_catalog_choice_input_ids,
    )
    if (
        not scope.binding_target_ids
        and not scope.required_catalog_input_ids
        and not scope.required_catalog_choice_input_ids
    ):
        variants.append(_impossible_schema(requested_fact_ids=scope.requested_fact_ids))
    if clarification is not None:
        variants.append(clarification)
    return {"oneOf": variants}


def _source_binding_plan_schema(scope: _SourceBindingSchemaScope) -> dict[str, object]:
    source_invocations_schema = {
        "type": "array",
        "minItems": 1,
        "items": _source_binding_invocation_items_schema(scope),
    }
    return provider_output.SourceBindingPlanOutput.schema(
        {
            "kind": {"enum": ["source_bindings"]},
            "metric_fit_bases": _metric_fit_bases_schema(
                scope.metric_evidence_ids_by_requested_fact
            ),
            "fit_basis_interpretations": _fit_basis_interpretations_schema(
                scope.metric_evidence_ids_by_requested_fact
            ),
            "source_invocations": source_invocations_schema,
        }
    )


def _source_binding_invocation_items_schema(
    scope: _SourceBindingSchemaScope,
) -> dict[str, object]:
    return _variant_schema(
        tuple(
            _source_binding_item_schema(
                target_id,
                scope=scope,
            )
            for target_id in scope.binding_target_ids
        )
    )


def _source_binding_item_schema(
    target_id: str,
    *,
    scope: _SourceBindingSchemaScope,
) -> dict[str, object]:
    return provider_output.SourceInvocationOutput.schema(
        {
            "binding_target_id": {"enum": [target_id]},
            "answer_population": _answer_population_schema(
                scope.target_population_binding_ids.get(target_id, ())
            ),
            "fulfillment_decisions": _fulfillment_decisions_schema(
                scope.target_fulfillment_support_set_ids_by_answer_output.get(
                    target_id, {}
                ),
                required_answer_output_ids=(
                    scope.target_required_fulfillment_answer_output_ids.get(
                        target_id,
                        (),
                    )
                ),
            ),
            "param_decisions": _param_decisions_schema(
                scope.target_param_decision_ids_by_param.get(target_id, {}),
                required_param_ids=(),
            ),
            "row_predicate_reviews": _row_predicate_reviews_schema(
                scope.target_row_predicate_values.get(target_id, {}),
                test_ids_by_predicate=scope.target_row_predicate_test_ids.get(
                    target_id, {}
                ),
            ),
            "finite_choice_param_reviews": _finite_choice_param_reviews_schema(
                scope.target_finite_choice_values.get(target_id, {}),
                test_ids_by_param=scope.target_finite_choice_test_ids.get(
                    target_id, {}
                ),
                normal_instance_test_ids_by_param=(
                    scope.target_finite_choice_normal_instance_test_ids.get(
                        target_id, {}
                    )
                ),
                population_roles=scope.target_population_roles.get(target_id, ()),
            ),
        }
    )


def _finite_choice_param_reviews_schema(
    finite_choice_values: dict[str, tuple[str, ...]],
    *,
    test_ids_by_param: dict[str, tuple[str, ...]],
    normal_instance_test_ids_by_param: dict[str, tuple[str, ...]],
    population_roles: tuple[dict[str, object], ...],
) -> dict[str, object]:
    reviewed_values = {
        param_id: choices
        for param_id, choices in finite_choice_values.items()
        if test_ids_by_param.get(param_id)
    }
    if not reviewed_values:
        return _empty_object_schema()
    return _strict_object(
        {
            param_id: _finite_choice_param_review_schema(
                choices,
                membership_test_ids=test_ids_by_param[param_id],
                normal_instance_test_ids=normal_instance_test_ids_by_param.get(
                    param_id,
                    (),
                ),
                population_roles=population_roles,
            )
            for param_id, choices in reviewed_values.items()
        },
        required=tuple(reviewed_values),
    )


def _finite_choice_param_review_schema(
    choices: tuple[str, ...],
    *,
    membership_test_ids: tuple[str, ...],
    normal_instance_test_ids: tuple[str, ...],
    population_roles: tuple[dict[str, object], ...],
) -> dict[str, object]:
    role_variants = tuple(
        _finite_choice_param_role_review_schema(
            choices,
            role=role,
            membership_test_ids=membership_test_ids,
            normal_instance_test_ids=normal_instance_test_ids,
        )
        for role in population_roles
        if str(role.get("role_id") or "")
    )
    if role_variants:
        return _variant_schema(role_variants)
    raise ValueError("finite choice param requires population roles")


def _finite_choice_param_role_review_schema(
    choices: tuple[str, ...],
    *,
    role: dict[str, object],
    membership_test_ids: tuple[str, ...],
    normal_instance_test_ids: tuple[str, ...],
) -> dict[str, object]:
    role_id = str(role.get("role_id") or "")
    return provider_output.FiniteChoiceParamReviewOutput.schema(
        {
            "controlled_population_role_id": (
                {"enum": [role_id]} if role_id else _handle_schema()
            ),
            "role_selection_basis": _handle_schema(),
            "population_test_basis": _population_test_basis_schema(membership_test_ids),
            "choice_reviews": {
                "type": "array",
                "minItems": len(choices),
                "maxItems": len(choices),
                "items": _finite_choice_review_schema(
                    choices,
                    membership_test_ids=membership_test_ids,
                    normal_instance_test_ids=normal_instance_test_ids,
                ),
            },
        }
    )


def _population_test_basis_schema(
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            test_id: provider_output.PopulationTestBasisOutput.schema(
                {
                    "test_question": _handle_schema(),
                    "role_scoped_test_question": _handle_schema(),
                }
            )
            for test_id in membership_test_ids
        },
        required=membership_test_ids,
    )


def _finite_choice_review_schema(
    choices: tuple[str, ...],
    *,
    membership_test_ids: tuple[str, ...],
    normal_instance_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return provider_output.FiniteChoiceReviewOutput.schema(
        {
            "choice_option_id": {"enum": list(choices)},
            "choice_domain_meaning": _handle_schema(),
            "choice_inclusion_basis": _handle_schema(),
            "choice_inclusion": {"enum": ["INCLUDE", "EXCLUDE"]},
            "population_test_results": _population_test_results_schema(
                membership_test_ids,
                normal_instance_test_ids=normal_instance_test_ids,
            ),
        }
    )


def _population_test_results_schema(
    membership_test_ids: tuple[str, ...],
    *,
    normal_instance_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            test_id: _population_test_result_schema(
                test_id,
                is_normal_instance=test_id in normal_instance_test_ids,
            )
            for test_id in membership_test_ids
        },
        required=membership_test_ids,
    )


def _population_test_result_schema(
    test_id: str,
    *,
    is_normal_instance: bool,
) -> dict[str, object]:
    if is_normal_instance:
        return _normal_instance_test_result_schema()
    return provider_output.StandardPopulationTestResultOutput.schema(
        {
            "test_basis": _handle_schema(),
            "population_consequence": _handle_schema(),
            "test_effect": _POPULATION_TEST_EFFECT_SCHEMA,
        }
    )


def _normal_instance_test_result_schema() -> dict[str, object]:
    return provider_output.NormalInstanceTestResultOutput.schema(
        {
            "role_match_basis": _handle_schema(),
            "explicit_user_override_evidence": _normal_instance_override_schema(),
            "explicit_user_override_applies": {"type": "boolean"},
            "population_consequence": _handle_schema(),
            "disposition": provider_output.NormalInstanceDispositionOutput.schema(
                {
                    "matched_excluded_role": _normal_instance_role_schema(),
                    "test_effect": _POPULATION_TEST_EFFECT_SCHEMA,
                }
            ),
        }
    )


def _normal_instance_override_schema() -> dict[str, object]:
    return {
        "type": "array",
        "items": provider_output.NormalInstanceOverrideEvidenceOutput.schema(
            {
                "source_text": _handle_schema(),
                "reason": {
                    "enum": [
                        reason.value for reason in NormalInstanceExplicitOverrideReason
                    ],
                },
            }
        ),
    }


def _normal_instance_role_schema() -> dict[str, object]:
    return {
        "enum": [
            *(role.value for role in NormalInstanceExcludedStateRole),
            NORMAL_INSTANCE_NO_EXCLUDED_ROLE,
            NORMAL_INSTANCE_UNKNOWN_EXCLUDED_ROLE,
        ],
    }


def _answer_population_schema(
    population_binding_ids: tuple[str, ...],
) -> dict[str, object]:
    population_binding_id_schema: dict[str, object] = _handle_schema()
    if population_binding_ids:
        population_binding_id_schema = {
            "type": "string",
            "enum": list(population_binding_ids),
        }
    return provider_output.AnswerPopulationOutput.schema(
        {
            "population_binding_id": population_binding_id_schema,
            "intent_text": _handle_schema(),
            "match_basis_explanation": _handle_schema(),
        }
    )


def _fulfillment_decisions_schema(
    fulfillment_support_set_ids_by_answer_output: dict[str, tuple[str, ...]],
    *,
    required_answer_output_ids: tuple[str, ...],
) -> dict[str, object]:
    properties = {
        answer_output_id: _fulfillment_decision_item_schema(support_set_ids)
        for answer_output_id, support_set_ids in (
            fulfillment_support_set_ids_by_answer_output.items()
        )
    }
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required_answer_output_ids),
    }
    return schema


def _fulfillment_decision_item_schema(
    fulfillment_choice_ids: tuple[str, ...],
) -> dict[str, object]:
    choice_id_schema = _handle_schema()
    if fulfillment_choice_ids:
        choice_id_schema = {
            "type": "string",
            "enum": list(fulfillment_choice_ids),
        }
    return provider_output.FulfillmentDecisionOutput.schema(
        {
            "match_basis_explanation": {"type": "string", "minLength": 1},
            "fulfillment_choice_id": choice_id_schema,
        }
    )


def _metric_fit_bases_schema(
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    return _metric_reviews_schema(
        metric_evidence_ids_by_requested_fact,
        item_schema=provider_output.MetricFitBasisOutput.schema(
            {
                "metric_meaning": {"type": "string", "minLength": 1},
                "fit_basis": {"type": "string", "minLength": 1},
            }
        ),
    )


def _fit_basis_interpretations_schema(
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    return _metric_reviews_schema(
        metric_evidence_ids_by_requested_fact,
        item_schema=provider_output.FitBasisInterpretationOutput.schema(
            {
                "interpretation": {
                    "type": "string",
                    "enum": list(METRIC_FIT_DECISIONS),
                },
            }
        ),
    )


def _metric_reviews_schema(
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
    *,
    item_schema: dict[str, object],
) -> dict[str, object]:
    properties = {
        requested_fact_id: _strict_object(
            {metric_evidence_id: item_schema for metric_evidence_id in metric_evidence_ids},
            required=metric_evidence_ids,
        )
        for requested_fact_id, metric_evidence_ids in (
            metric_evidence_ids_by_requested_fact.items()
        )
        if metric_evidence_ids
    }
    if not properties:
        return _empty_object_schema()
    return _strict_object(properties, required=tuple(properties))


def _param_decisions_schema(
    param_decision_ids_by_param: dict[str, tuple[str, ...]],
    *,
    required_param_ids: tuple[str, ...],
) -> dict[str, object]:
    if not param_decision_ids_by_param:
        return _empty_object_schema()
    param_ids = tuple(param_decision_ids_by_param)
    return _strict_object(
        {
            param_id: _param_decision_item_schema(
                param_decision_ids_by_param[param_id],
            )
            for param_id in param_ids
        },
        required=tuple(
            dict.fromkeys(
                param_id for param_id in required_param_ids if param_id in param_ids
            )
        ),
    )


def _row_predicate_reviews_schema(
    row_predicate_values: dict[str, tuple[str, ...]],
    *,
    test_ids_by_predicate: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    reviewed_values = {
        predicate_id: values
        for predicate_id, values in row_predicate_values.items()
        if test_ids_by_predicate.get(predicate_id)
    }
    if not reviewed_values:
        return _empty_object_schema()
    return _strict_object(
        {
            predicate_id: _row_predicate_review_schema(
                values,
                membership_test_ids=test_ids_by_predicate[predicate_id],
            )
            for predicate_id, values in reviewed_values.items()
        },
        required=(),
    )


def _row_predicate_review_schema(
    values: tuple[str, ...],
    *,
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return provider_output.RowPredicateReviewOutput.schema(
        {
            "choice_reviews": {
                "type": "array",
                "minItems": len(values),
                "maxItems": len(values),
                "items": _row_predicate_choice_review_schema(
                    values,
                    membership_test_ids=membership_test_ids,
                ),
            },
        }
    )


def _row_predicate_choice_review_schema(
    values: tuple[str, ...],
    *,
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return provider_output.RowPredicateChoiceReviewOutput.schema(
        {
            "choice_option_id": {"enum": list(values)},
            "choice_domain_meaning": _handle_schema(),
            "population_test_results": _row_predicate_population_test_results_schema(
                membership_test_ids,
            ),
        }
    )


def _row_predicate_population_test_results_schema(
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            test_id: _row_predicate_population_test_result_schema(test_id)
            for test_id in membership_test_ids
        },
        required=membership_test_ids,
    )


def _row_predicate_population_test_result_schema(test_id: str) -> dict[str, object]:
    return provider_output.RowPredicatePopulationTestResultOutput.schema(
        {
            "test_id": {"enum": [test_id]},
            "test_question": _handle_schema(),
            "role_scoped_test_question": _handle_schema(),
            "because": _handle_schema(),
            "test_effect": _POPULATION_TEST_EFFECT_SCHEMA,
        }
    )


def _empty_object_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }


def _param_decision_item_schema(
    param_decision_ids: tuple[str, ...],
) -> dict[str, object]:
    param_decision_id_schema: dict[str, object] = {"type": "string"}
    if param_decision_ids:
        param_decision_id_schema["enum"] = list(param_decision_ids)
    return provider_output.ParamDecisionOutput.schema(
        {
            "population_intent": _handle_schema(),
            "match_basis_explanation": _handle_schema(),
            "param_decision_id": param_decision_id_schema,
        }
    )


def _impossible_schema(
    *,
    allowed_bases: tuple[str, ...] = (
        BlockedFactBasis.CATALOG_ACCESS.value,
        BlockedFactBasis.POLICY_ACCESS.value,
    ),
    requested_fact_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    requested_fact_id_schema = _handle_schema()
    if requested_fact_ids:
        requested_fact_id_schema = {"enum": list(requested_fact_ids)}
    blocked_fact_schema = _strict_object(
        {
            "requested_fact_id": requested_fact_id_schema,
            "basis": {"enum": list(allowed_bases)},
            "evidence_refs": {
                "type": "array",
                "minItems": 1,
                "items": _handle_schema(),
            },
            "reviewed_read_ids": {
                "type": "array",
                "items": _handle_schema(),
            },
            "nearest_fields": {
                "type": "array",
                "items": _strict_object(
                    {
                        "read_id": _handle_schema(),
                        "field_id": _handle_schema(),
                    },
                    required=("read_id", "field_id"),
                ),
            },
            "explanation": {"type": "string"},
        },
        required=("requested_fact_id", "basis", "evidence_refs"),
    )
    return _strict_object(
        {
            "kind": {"enum": [PlanOutcomeKind.IMPOSSIBLE.value]},
            "blocked_facts": {
                "type": "array",
                "minItems": 1,
                "items": blocked_fact_schema,
            },
        },
        required=("kind", "blocked_facts"),
    )


def _clarification_schema(
    *,
    required_catalog_input_ids: tuple[str, ...],
    required_catalog_choice_input_ids: tuple[str, ...],
) -> dict[str, object] | None:
    variants: list[dict[str, object]] = []
    if required_catalog_input_ids:
        variants.append(
            _strict_object(
                {
                    "kind": {
                        "enum": [MissingCatalogInputKind.REQUIRED_INPUT.value],
                    },
                    "id": _handle_schema(),
                    "requested_fact_id": _handle_schema(),
                    "required_catalog_input_id": {
                        "enum": list(required_catalog_input_ids),
                    },
                },
                required=(
                    "kind",
                    "id",
                    "requested_fact_id",
                    "required_catalog_input_id",
                ),
            )
        )
    if required_catalog_choice_input_ids:
        variants.append(
            _strict_object(
                {
                    "kind": {
                        "enum": [MissingCatalogInputKind.CHOICE_INPUT.value],
                    },
                    "id": _handle_schema(),
                    "requested_fact_id": _handle_schema(),
                    "required_catalog_choice_input_id": {
                        "enum": list(required_catalog_choice_input_ids),
                    },
                },
                required=(
                    "kind",
                    "id",
                    "requested_fact_id",
                    "required_catalog_choice_input_id",
                ),
            )
        )
    if not variants:
        return None
    return _strict_object(
        {
            "kind": {"enum": ["needs_clarification"]},
            "missing_catalog_inputs": {
                "type": "array",
                "minItems": 1,
                "items": {"oneOf": variants},
            },
        },
        required=("kind", "missing_catalog_inputs"),
    )
