"""Provider schema for source binding."""

from __future__ import annotations


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


def _handle_schema() -> dict[str, object]:
    return {"type": "string", "minLength": 1}


def _population_intent_schema() -> dict[str, object]:
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


def build_source_binding_schema(
    *,
    required_catalog_input_ids: tuple[str, ...] = (),
    required_catalog_choice_input_ids: tuple[str, ...] = (),
    source_candidate_param_decision_ids_by_param: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_required_param_ids: dict[str, tuple[str, ...]],
    source_candidate_finite_choice_values: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_row_predicate_values: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_membership_test_ids: dict[str, tuple[str, ...]],
    source_candidate_normal_instance_test_ids: dict[str, tuple[str, ...]],
    source_candidate_population_roles: dict[str, tuple[dict[str, object], ...]],
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
    source_candidate_requested_fact_ids: dict[str, str] | None = None,
    source_candidate_fulfillment_support_set_ids_by_answer_output: dict[
        str, dict[str, tuple[str, ...]]
    ],
    source_candidate_population_binding_ids: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    fulfillment_support_set_ids_by_answer_output = (
        source_candidate_fulfillment_support_set_ids_by_answer_output
    )
    population_binding_ids = source_candidate_population_binding_ids or {}
    outcome_schema = _source_binding_outcome_schema(
        required_catalog_input_ids=required_catalog_input_ids,
        required_catalog_choice_input_ids=required_catalog_choice_input_ids,
        source_candidate_param_decision_ids_by_param=(
            source_candidate_param_decision_ids_by_param
        ),
        source_candidate_required_param_ids=source_candidate_required_param_ids,
        source_candidate_finite_choice_values=source_candidate_finite_choice_values,
        source_candidate_row_predicate_values=source_candidate_row_predicate_values,
        source_candidate_membership_test_ids=source_candidate_membership_test_ids,
        source_candidate_normal_instance_test_ids=(
            source_candidate_normal_instance_test_ids
        ),
        source_candidate_population_roles=source_candidate_population_roles,
        metric_evidence_ids_by_requested_fact=metric_evidence_ids_by_requested_fact,
        source_candidate_requested_fact_ids=source_candidate_requested_fact_ids or {},
        source_candidate_fulfillment_support_set_ids_by_answer_output=(
            fulfillment_support_set_ids_by_answer_output
        ),
        source_candidate_population_binding_ids=population_binding_ids,
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"outcome": outcome_schema},
        "required": ["outcome"],
        "modelSchemas": {"outcome": outcome_schema},
    }


def _source_binding_outcome_schema(
    *,
    required_catalog_input_ids: tuple[str, ...],
    required_catalog_choice_input_ids: tuple[str, ...],
    source_candidate_param_decision_ids_by_param: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_required_param_ids: dict[str, tuple[str, ...]],
    source_candidate_finite_choice_values: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_row_predicate_values: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_membership_test_ids: dict[str, tuple[str, ...]],
    source_candidate_normal_instance_test_ids: dict[str, tuple[str, ...]],
    source_candidate_population_roles: dict[str, tuple[dict[str, object], ...]],
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
    source_candidate_requested_fact_ids: dict[str, str],
    source_candidate_fulfillment_support_set_ids_by_answer_output: dict[
        str, dict[str, tuple[str, ...]]
    ],
    source_candidate_population_binding_ids: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    variants: list[dict[str, object]] = []
    answer_source_candidate_ids = tuple(
        source_candidate_id
        for source_candidate_id in source_candidate_param_decision_ids_by_param
        if source_candidate_fulfillment_support_set_ids_by_answer_output.get(
            source_candidate_id
        )
    )
    requested_fact_ids = tuple(
        dict.fromkeys(source_candidate_requested_fact_ids.values())
    )
    if answer_source_candidate_ids:
        variants.append(
            _source_binding_plan_schema(
                source_candidate_param_decision_ids_by_param={
                    source_candidate_id: source_candidate_param_decision_ids_by_param[
                        source_candidate_id
                    ]
                    for source_candidate_id in answer_source_candidate_ids
                },
                source_candidate_required_param_ids=source_candidate_required_param_ids,
                source_candidate_finite_choice_values=source_candidate_finite_choice_values,
                source_candidate_row_predicate_values=source_candidate_row_predicate_values,
                source_candidate_membership_test_ids=source_candidate_membership_test_ids,
                source_candidate_normal_instance_test_ids=(
                    source_candidate_normal_instance_test_ids
                ),
                source_candidate_population_roles=source_candidate_population_roles,
                metric_evidence_ids_by_requested_fact=metric_evidence_ids_by_requested_fact,
                source_candidate_requested_fact_ids=source_candidate_requested_fact_ids,
                source_candidate_fulfillment_support_set_ids_by_answer_output=(
                    source_candidate_fulfillment_support_set_ids_by_answer_output
                ),
                source_candidate_population_binding_ids=source_candidate_population_binding_ids,
            )
        )
        variants.append(
            _impossible_schema(
                allowed_bases=(BlockedFactBasis.POLICY_ACCESS.value,),
                requested_fact_ids=requested_fact_ids,
            )
        )
    clarification = _clarification_schema(
        required_catalog_input_ids=required_catalog_input_ids,
        required_catalog_choice_input_ids=required_catalog_choice_input_ids,
    )
    if (
        not answer_source_candidate_ids
        and not required_catalog_input_ids
        and not required_catalog_choice_input_ids
    ):
        variants.append(_impossible_schema(requested_fact_ids=requested_fact_ids))
    if clarification is not None:
        variants.append(clarification)
    return {"oneOf": variants}


def _source_binding_plan_schema(
    *,
    source_candidate_param_decision_ids_by_param: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_required_param_ids: dict[str, tuple[str, ...]],
    source_candidate_finite_choice_values: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_row_predicate_values: dict[str, dict[str, tuple[str, ...]]],
    source_candidate_membership_test_ids: dict[str, tuple[str, ...]],
    source_candidate_normal_instance_test_ids: dict[str, tuple[str, ...]],
    source_candidate_population_roles: dict[str, tuple[dict[str, object], ...]],
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
    source_candidate_requested_fact_ids: dict[str, str],
    source_candidate_fulfillment_support_set_ids_by_answer_output: dict[
        str, dict[str, tuple[str, ...]]
    ],
    source_candidate_population_binding_ids: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    source_invocations_schema = {
        "type": "array",
        "minItems": 1,
        "items": {
            "oneOf": [
                _source_binding_item_schema(
                    source_candidate_id=source_candidate_id,
                    requested_fact_id=source_candidate_requested_fact_ids.get(
                        source_candidate_id, ""
                    ),
                    param_decision_ids_by_param=param_decision_ids_by_param,
                    required_param_ids=source_candidate_required_param_ids.get(
                        source_candidate_id, ()
                    ),
                    finite_choice_values=source_candidate_finite_choice_values.get(
                        source_candidate_id, {}
                    ),
                    row_predicate_values=source_candidate_row_predicate_values.get(
                        source_candidate_id, {}
                    ),
                    membership_test_ids=source_candidate_membership_test_ids.get(
                        source_candidate_id, ()
                    ),
                    normal_instance_test_ids=(
                        source_candidate_normal_instance_test_ids.get(
                            source_candidate_id, ()
                        )
                    ),
                    population_roles=source_candidate_population_roles.get(
                        source_candidate_id,
                        (),
                    ),
                    fulfillment_support_set_ids_by_answer_output=source_candidate_fulfillment_support_set_ids_by_answer_output.get(
                        source_candidate_id, {}
                    ),
                    population_binding_ids=source_candidate_population_binding_ids.get(
                        source_candidate_id, ()
                    ),
                )
                for source_candidate_id, param_decision_ids_by_param in (
                    source_candidate_param_decision_ids_by_param.items()
                )
            ]
        },
    }
    return _strict_object(
        {
            "kind": {"enum": ["source_bindings"]},
            "metric_fit_bases": _metric_fit_bases_schema(
                metric_evidence_ids_by_requested_fact
            ),
            "fit_basis_interpretations": _fit_basis_interpretations_schema(
                metric_evidence_ids_by_requested_fact
            ),
            "source_invocations": source_invocations_schema,
        },
        required=(
            "kind",
            "metric_fit_bases",
            "fit_basis_interpretations",
            "source_invocations",
        ),
    )


def _source_binding_item_schema(
    *,
    source_candidate_id: str,
    requested_fact_id: str,
    param_decision_ids_by_param: dict[str, tuple[str, ...]],
    required_param_ids: tuple[str, ...],
    finite_choice_values: dict[str, tuple[str, ...]],
    row_predicate_values: dict[str, tuple[str, ...]],
    membership_test_ids: tuple[str, ...],
    normal_instance_test_ids: tuple[str, ...],
    population_roles: tuple[dict[str, object], ...],
    fulfillment_support_set_ids_by_answer_output: dict[str, tuple[str, ...]],
    population_binding_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            "requested_fact_id": (
                {"enum": [requested_fact_id]} if requested_fact_id else _handle_schema()
            ),
            "source_candidate_id": {"enum": [source_candidate_id]},
            "answer_population": _answer_population_schema(population_binding_ids),
            "fulfillment_decisions": _fulfillment_decisions_schema(
                fulfillment_support_set_ids_by_answer_output,
                enforce_enum=True,
            ),
            "param_decisions": _param_decisions_schema(
                param_decision_ids_by_param,
                required_param_ids=required_param_ids,
            ),
            "row_predicate_reviews": _row_predicate_reviews_schema(
                row_predicate_values,
                membership_test_ids=membership_test_ids,
            ),
            "finite_choice_param_reviews": _finite_choice_param_reviews_schema(
                finite_choice_values,
                membership_test_ids=membership_test_ids,
                normal_instance_test_ids=normal_instance_test_ids,
                population_roles=population_roles,
            ),
            "source_binding_decision": {"enum": ["USE_SOURCE"]},
        },
        required=(
            "requested_fact_id",
            "source_candidate_id",
            "answer_population",
            "fulfillment_decisions",
            "param_decisions",
            "row_predicate_reviews",
            "finite_choice_param_reviews",
            "source_binding_decision",
        ),
    )


def _finite_choice_param_reviews_schema(
    finite_choice_values: dict[str, tuple[str, ...]],
    *,
    membership_test_ids: tuple[str, ...],
    normal_instance_test_ids: tuple[str, ...],
    population_roles: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if not finite_choice_values:
        return _empty_object_schema()
    return _strict_object(
        {
            param_id: _finite_choice_param_review_schema(
                choices,
                membership_test_ids=membership_test_ids,
                normal_instance_test_ids=normal_instance_test_ids,
                population_roles=population_roles,
            )
            for param_id, choices in finite_choice_values.items()
        },
        required=tuple(finite_choice_values),
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
    if len(role_variants) == 1:
        return role_variants[0]
    if role_variants:
        return {"oneOf": list(role_variants)}
    raise ValueError("finite choice param requires population roles")


def _finite_choice_param_role_review_schema(
    choices: tuple[str, ...],
    *,
    role: dict[str, object],
    membership_test_ids: tuple[str, ...],
    normal_instance_test_ids: tuple[str, ...],
) -> dict[str, object]:
    role_id = str(role.get("role_id") or "")
    return _strict_object(
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
        },
        required=(
            "controlled_population_role_id",
            "role_selection_basis",
            "population_test_basis",
            "choice_reviews",
        ),
    )


def _population_test_basis_schema(
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            test_id: _strict_object(
                {
                    "test_question": _handle_schema(),
                    "role_scoped_test_question": _handle_schema(),
                },
                required=("test_question", "role_scoped_test_question"),
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
    return _strict_object(
        {
            "choice_option_id": {"enum": list(choices)},
            "choice_domain_meaning": _handle_schema(),
            "choice_inclusion_basis": _handle_schema(),
            "choice_inclusion": {"enum": ["INCLUDE", "EXCLUDE"]},
            "population_test_results": _population_test_results_schema(
                membership_test_ids,
                normal_instance_test_ids=normal_instance_test_ids,
            ),
        },
        required=(
            "choice_option_id",
            "choice_domain_meaning",
            "choice_inclusion_basis",
            "choice_inclusion",
            "population_test_results",
        ),
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
    return _standard_population_test_result_schema()


def _standard_population_test_result_schema() -> dict[str, object]:
    return _strict_object(
        {
            "test_basis": _handle_schema(),
            "population_consequence": _handle_schema(),
            "test_effect": {
                "enum": [
                    "SATISFIES_TEST",
                    "CONFLICTS_WITH_TEST",
                    "DOES_NOT_DECIDE_TEST",
                    "UNKNOWN_TEST_EFFECT",
                ]
            },
        },
        required=(
            "test_basis",
            "population_consequence",
            "test_effect",
        ),
    )


def _normal_instance_test_result_schema() -> dict[str, object]:
    return _strict_object(
        {
            "role_match_basis": _handle_schema(),
            "explicit_user_override_evidence": _normal_instance_override_schema(),
            "explicit_user_override_applies": {"type": "boolean"},
            "population_consequence": _handle_schema(),
            "disposition": _strict_object(
                {
                    "matched_excluded_role": _normal_instance_role_schema(),
                    "test_effect": {
                        "enum": [
                            "SATISFIES_TEST",
                            "CONFLICTS_WITH_TEST",
                            "DOES_NOT_DECIDE_TEST",
                            "UNKNOWN_TEST_EFFECT",
                        ]
                    },
                },
                required=(
                    "matched_excluded_role",
                    "test_effect",
                ),
            ),
        },
        required=(
            "role_match_basis",
            "explicit_user_override_evidence",
            "explicit_user_override_applies",
            "population_consequence",
            "disposition",
        ),
    )


def _normal_instance_override_schema() -> dict[str, object]:
    return {
        "type": "array",
        "items": _strict_object(
            {
                "source_text": _handle_schema(),
                "reason": {
                    "enum": [
                        reason.value for reason in NormalInstanceExplicitOverrideReason
                    ],
                },
            },
            required=("source_text", "reason"),
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
    return _strict_object(
        {
            "population_binding_id": population_binding_id_schema,
            "intent_text": _handle_schema(),
            "match_basis_explanation": _handle_schema(),
        },
        required=("population_binding_id", "intent_text", "match_basis_explanation"),
    )


def _fulfillment_decisions_schema(
    fulfillment_support_set_ids_by_answer_output: dict[str, tuple[str, ...]],
    *,
    enforce_enum: bool,
) -> dict[str, object]:
    properties = {
        answer_output_id: _fulfillment_decision_item_schema(
            support_set_ids,
            enforce_enum=enforce_enum,
        )
        for answer_output_id, support_set_ids in (
            fulfillment_support_set_ids_by_answer_output.items()
        )
    }
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": [],
    }
    if properties:
        schema["anyOf"] = [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": [answer_output_id],
            }
            for answer_output_id in properties
        ]
    return schema


def _fulfillment_decision_item_schema(
    fulfillment_choice_ids: tuple[str, ...],
    *,
    enforce_enum: bool,
) -> dict[str, object]:
    choice_id_schema = _handle_schema()
    if fulfillment_choice_ids and enforce_enum:
        choice_id_schema = {
            "type": "string",
            "enum": list(fulfillment_choice_ids),
        }
    return _strict_object(
        {
            "match_basis_explanation": {"type": "string", "minLength": 1},
            "fulfillment_choice_id": choice_id_schema,
        },
        required=(
            "match_basis_explanation",
            "fulfillment_choice_id",
        ),
    )


def _metric_fit_bases_schema(
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    properties = {
        requested_fact_id: _metric_fit_bases_for_fact_schema(metric_evidence_ids)
        for requested_fact_id, metric_evidence_ids in (
            metric_evidence_ids_by_requested_fact.items()
        )
        if metric_evidence_ids
    }
    if not properties:
        return _empty_object_schema()
    return _strict_object(properties, required=tuple(properties))


def _metric_fit_bases_for_fact_schema(
    metric_evidence_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            metric_evidence_id: _metric_fit_basis_schema()
            for metric_evidence_id in metric_evidence_ids
        },
        required=metric_evidence_ids,
    )


def _metric_fit_basis_schema() -> dict[str, object]:
    return _strict_object(
        {
            "metric_meaning": {"type": "string", "minLength": 1},
            "fit_basis": {"type": "string", "minLength": 1},
        },
        required=(
            "metric_meaning",
            "fit_basis",
        ),
    )


def _fit_basis_interpretations_schema(
    metric_evidence_ids_by_requested_fact: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    properties = {
        requested_fact_id: _fit_basis_interpretations_for_fact_schema(
            metric_evidence_ids
        )
        for requested_fact_id, metric_evidence_ids in (
            metric_evidence_ids_by_requested_fact.items()
        )
        if metric_evidence_ids
    }
    if not properties:
        return _empty_object_schema()
    return _strict_object(properties, required=tuple(properties))


def _fit_basis_interpretations_for_fact_schema(
    metric_evidence_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            metric_evidence_id: _fit_basis_interpretation_schema()
            for metric_evidence_id in metric_evidence_ids
        },
        required=metric_evidence_ids,
    )


def _fit_basis_interpretation_schema() -> dict[str, object]:
    return _strict_object(
        {
            "interpretation": {
                "type": "string",
                "enum": list(METRIC_FIT_DECISIONS),
            },
        },
        required=("interpretation",),
    )


def _param_decisions_schema(
    param_decision_ids_by_param: dict[str, tuple[str, ...]],
    *,
    required_param_ids: tuple[str, ...],
    enforce_enum: bool = True,
) -> dict[str, object]:
    if not param_decision_ids_by_param:
        return _empty_object_schema()
    param_ids = tuple(param_decision_ids_by_param)
    return _strict_object(
        {
            param_id: _param_decision_item_schema(
                param_decision_ids_by_param[param_id],
                enforce_enum=enforce_enum,
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
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    if not row_predicate_values:
        return _empty_object_schema()
    return _strict_object(
        {
            predicate_id: _row_predicate_review_schema(
                values,
                membership_test_ids=membership_test_ids,
            )
            for predicate_id, values in row_predicate_values.items()
        },
        required=tuple(row_predicate_values.keys()),
    )


def _row_predicate_review_schema(
    values: tuple[str, ...],
    *,
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
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
        },
        required=("choice_reviews",),
    )


def _row_predicate_choice_review_schema(
    values: tuple[str, ...],
    *,
    membership_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return _strict_object(
        {
            "choice_option_id": {"enum": list(values)},
            "choice_domain_meaning": _handle_schema(),
            "population_test_results": _row_predicate_population_test_results_schema(
                membership_test_ids,
            ),
        },
        required=(
            "choice_option_id",
            "choice_domain_meaning",
            "population_test_results",
        ),
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
    return _strict_object(
        {
            "test_id": {"enum": [test_id]},
            "test_question": _handle_schema(),
            "role_scoped_test_question": _handle_schema(),
            "because": _handle_schema(),
            "test_effect": {
                "enum": [
                    "SATISFIES_TEST",
                    "CONFLICTS_WITH_TEST",
                    "DOES_NOT_DECIDE_TEST",
                    "UNKNOWN_TEST_EFFECT",
                ]
            },
        },
        required=(
            "test_id",
            "test_question",
            "role_scoped_test_question",
            "because",
            "test_effect",
        ),
    )


def _empty_object_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }


def _param_decision_item_schema(
    param_decision_ids: tuple[str, ...],
    *,
    enforce_enum: bool = True,
    include_param_id: bool = False,
) -> dict[str, object]:
    param_decision_id_schema: dict[str, object] = {"type": "string"}
    if param_decision_ids and enforce_enum:
        param_decision_id_schema["enum"] = list(param_decision_ids)
    properties = {
        "population_intent": _population_intent_schema(),
        "match_basis_explanation": _handle_schema(),
        "param_decision_id": param_decision_id_schema,
    }
    required = [
        "population_intent",
        "match_basis_explanation",
        "param_decision_id",
    ]
    if include_param_id:
        properties = {"param_id": _handle_schema(), **properties}
        required = ["param_id", *required]
    return _strict_object(
        properties,
        required=tuple(required),
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
