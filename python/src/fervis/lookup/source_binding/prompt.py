"""Prompt for source binding."""

from __future__ import annotations

from typing import Any

from fervis.lookup.fact_planning.available_relations import (
    operation_input_values_payload,
)
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections import source_binding_candidates_xml
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    requested_fact_evidence_ref,
)
from fervis.lookup.source_binding.candidates import (
    source_binding_candidate_payload,
    source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output,
    source_binding_prompt_candidate_population_binding_ids,
    source_binding_prompt_candidate_requested_fact_ids,
    source_candidate_required_param_decision_ids,
    source_candidate_registry,
)
from fervis.lookup.source_binding.memory_context import (
    source_binding_memory_context_payload as source_binding_memory_context_input_payload,
)
from fervis.lookup.operation_families.source_binding_registry import (
    source_binding_metric_evidence_ids_by_requested_fact,
    source_binding_metric_fit_surface_payload,
)
from fervis.lookup.source_binding.model import (
    SourceBindingRequest,
)
from fervis.lookup.source_binding.membership_tests import (
    membership_test_key,
)
from fervis.lookup.source_binding.param_surface import (
    choice_values_for_effective_params,
    param_decision_ids_by_effective_param,
    param_requires_finite_choice_review,
)
from fervis.lookup.source_binding.schema import (
    build_source_binding_schema,
)
from fervis.lookup.source_binding.terminal_outcomes import (
    source_binding_clarification_input_ids,
)
from fervis.model_io.structured_output.specs import required_tool_spec


SOURCE_BINDING_TOOL_NAME = "submit_source_binding"


def source_binding_requested_facts_payload(
    request: SourceBindingRequest,
) -> dict[str, object]:
    return {
        "requested_facts": [
            {
                "requested_fact_id": fact.id,
                "evidence_ref": requested_fact_evidence_ref(fact.id),
                "description": fact.description,
                "answer_request": fact.answer_request_model_dict(),
                "answer_outputs": [
                    {
                        "id": output.id,
                        "description": output.description,
                    }
                    for output in fact.answer_outputs
                ],
            }
            for fact in request.requested_facts
        ]
    }


def source_binding_grounded_values_payload(
    request: SourceBindingRequest,
) -> dict[str, object]:
    return operation_input_values_payload(
        available_values=request.available_values,
        available_value_uses=(),
    )


def source_binding_memory_context_payload(
    request: SourceBindingRequest,
) -> dict[str, object]:
    return source_binding_memory_context_input_payload(
        request.memory_inputs,
        active_memory_ids=tuple(request.active_memory_ids),
    )


def source_binding_transport_context_payload(
    request: SourceBindingRequest,
) -> dict[str, object]:
    registry = source_candidate_registry(request)
    return {
        "current_question": request.question,
        "requested_facts": source_binding_requested_facts_payload(request),
        "grounded_values": source_binding_grounded_values_payload(request),
        "memory_context": source_binding_memory_context_payload(request),
        "source_candidate_requested_fact_ids_by_id": {
            candidate_id: candidate.requested_fact_id
            for candidate_id, candidate in registry.candidates_by_id.items()
            if candidate_id in registry.prompt_candidate_ids
        },
        "source_candidates_by_id": {
            candidate_id: candidate.payload or {}
            for candidate_id, candidate in registry.candidates_by_id.items()
            if candidate_id in registry.prompt_candidate_ids
        },
    }


class SourceBindingTurnPrompt(TurnPromptBase):
    turn_name = "source binding"
    turn_task = (
        "choose source invocations, bind params, and assess finite choice membership"
    )

    def __init__(self, request: SourceBindingRequest) -> None:
        self.request = request

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.json_section("Requested facts:", self.requested_facts_payload()),
            builder.json_section("Grounded values:", self.grounded_values_payload()),
            builder.json_section("Memory context:", self.memory_context_payload()),
            builder.text_section(
                "Candidate evidence sources:",
                source_binding_candidates_xml(
                    self.source_invocation_candidate_payload()
                ),
            ),
            builder.json_section(
                "Metric fit candidates:",
                self.metric_fit_surface_payload(),
            ),
        )

    def instruction_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Source Binding",
                (
                    "For each requested fact, select the least number of source invocations required to answer its outputs.",
                    "A source invocation chooses the source candidate, answer population, fulfillment choice, scalar param bindings, and finite-choice population reviews.",
                    "Do not bind backup alternatives or invocations that are not required for an answer output.",
                ),
            ),
            builder.instruction_block(
                "Source Population And Fulfillment",
                (
                    "answer_population.population_binding_id is the source candidate's population handle that this invocation uses.",
                    "applied_filters are backend-derived grounded-value filters already attached to this source candidate.",
                    "Do not reinterpret, rematch, remove, or rewrite applied_filters.",
                    "For each fulfilled answer output key, choose fulfillment_choice_id from the selected source candidate's fulfillment_choices.",
                    "Before choosing fulfillment, write metric_fit_bases for every metric candidate.",
                    "Metric contexts are read-only interpretation context for metric candidates.",
                    "Use metric_contexts to understand what a metric field is likely measuring from its row path, same-row sibling fields, and scope fields.",
                    "Do not copy metric_context_id, same_row_field_paths, or scope_field_paths into output.",
                    "Do not treat metric context fields as selectable fulfillment evidence.",
                    "metric_fit_bases is keyed by requested_fact_id, then metric_evidence_id from Metric fit candidates.",
                    "metric_meaning states what the reviewed metric_evidence_id appears to measure from field_path, field_type, resource_names, and the referenced metric_context.",
                    "fit_basis evaluates whether the metric is the row-level or scalar measure that should be aggregated, ranked, compared, or otherwise computed to determine the requested answer output.",
                    "fit_basis must not reject a metric merely because it is an input to a later computation rather than the final answer value.",
                    "After all fit_basis entries, write fit_basis_interpretations with the same keys, using only the already-written fit_basis text.",
                    "Use interpretation=FITS_REQUESTED_ANSWER when the written fit_basis says the metric fits that role; otherwise use DOES_NOT_FIT_REQUESTED_ANSWER.",
                    "A fulfillment_choice_id is valid only when its metric_measure_evidence and row_count_basis_evidence ids have interpretation=FITS_REQUESTED_ANSWER.",
                    "row_count_basis_evidence represents count_rows over a response row population; review it as a metric candidate.",
                    "Decide final source population, row grain, fulfillment, and param membership from the source-binding candidate data and the metric_fit_bases interpretations.",
                    "When multiple source candidates can fulfill the same answer output, choose the invocation whose fitted metric evidence set most directly measures or computes the requested fact.",
                    "If you choose a fitted metric evidence set that is less direct than another available fitted set, the source population or fulfillment basis must explain why that invocation is still stronger.",
                ),
            ),
            builder.instruction_block(
                "Param Binding",
                (
                    "Bind every catalog-required param in param_decisions unless that param appears in binding_params with population_contract.",
                    "You may bind an optional non-choice param in param_decisions only when the requested fact needs that param value.",
                    "For params with decision_surface=single_decision and no population_contract, choose exactly one decision_options item.",
                    "Do not author param_decisions for params with population_contract; write finite_choice_param_reviews keyed by param_id instead.",
                ),
            ),
            builder.instruction_block(
                "Row Predicates",
                (
                    "row_predicates filter response rows when the requested population is not fully controlled by query params.",
                    "For every shown row_predicate on a selected source candidate, write row_predicate_reviews keyed by predicate_id.",
                    "Review every shown predicate value exactly once and independently against the requested answer population tests.",
                    "The backend derives executable row filters from row_predicate_reviews.",
                ),
            ),
            builder.instruction_block(
                "Finite Choice Review Shape",
                (
                    "For every binding_params param with population_contract, write finite_choice_param_reviews[param_id].",
                    "Review every shown choice exactly once and independently as one member of a choice set.",
                    "Use the XML response rows to understand the source row grains before applying membership tests.",
                    "Write each finite-choice param review in this order: controlled_population_role_id, role_selection_basis, population_test_basis, choice_reviews.",
                    "Choose controlled_population_role_id from population_roles for the source row population whose rows this param controls.",
                    "role_selection_basis explains why this param controls that selected role.",
                ),
            ),
            builder.instruction_block(
                "Population Test Basis",
                (
                    "In population_test_basis, write one entry for every shown answer_population.membership_tests key.",
                    "Each population_test_basis item copies test_question.",
                    "Each population_test_basis item writes role_scoped_test_question by applying test_question to the selected role's role_text.",
                    "Use population_test_basis as local context before reviewing choices.",
                    "Do not repeat test_question or role_scoped_test_question inside each choice.",
                ),
            ),
            builder.instruction_block(
                "Choice Test Results",
                (
                    "For each choice, write choice_domain_meaning, choice_inclusion_basis, then choice_inclusion, then population_test_results.",
                    "choice_domain_meaning states what the source returns when this choice value is applied, read against the source candidate description and the answer subject. Do not paraphrase the choice token alone.",
                    "choice_inclusion_basis explains why rows with this choice value should be included or excluded from the answer computation, read against the requested fact and the source candidate description.",
                    "choice_inclusion states whether rows with this choice value should be included in the answer computation: INCLUDE or EXCLUDE.",
                    "population_test_results is keyed by answer_population.membership_tests; write every shown key exactly once.",
                    "Do not add population_test_results keys that are not shown in answer_population.membership_tests.",
                    "For non-NORMAL_INSTANCE_GUARD tests, write test_basis, population_consequence, then test_effect.",
                    "test_basis compares this choice value to the matching population_test_basis item.",
                    "population_consequence states whether and how this choice affects the requested answer population for that test.",
                    "Write test_effect last for every population_test_results item.",
                ),
            ),
            builder.instruction_block(
                "Normal Instance Guard",
                (
                    "When the membership test has kind=NORMAL_INSTANCE_GUARD, write role_match_basis, explicit_user_override_evidence, explicit_user_override_applies, population_consequence, then disposition.",
                    "For NORMAL_INSTANCE_GUARD, use the param's normal_instance_role_profiles input.",
                    "Read all excluded_state_roles in the matching normal_instance_role_profiles item before writing disposition.",
                    "role_match_basis must compare the choice domain meaning to the excluded state roles.",
                    "population_consequence states whether and how this choice affects the requested answer population for the normal-instance guard.",
                    "Based on the role_match_basis you wrote, write disposition.matched_excluded_role, then derive disposition.test_effect mechanically.",
                    "disposition.matched_excluded_role must follow from role_match_basis.",
                    "disposition.matched_excluded_role must be one role from the profile, NONE, or UNKNOWN.",
                    "Use disposition.matched_excluded_role=NONE only after considering all excluded_state_roles and finding that none applies.",
                    "Use disposition.matched_excluded_role=UNKNOWN only when the prompt data is insufficient to classify the choice against those roles.",
                    "If disposition.matched_excluded_role=UNKNOWN, use disposition.test_effect=UNKNOWN_TEST_EFFECT.",
                    "If disposition.matched_excluded_role is an excluded role and explicit_user_override_applies=false, use disposition.test_effect=CONFLICTS_WITH_TEST.",
                    "If disposition.matched_excluded_role is an excluded role and explicit_user_override_applies=true, use disposition.test_effect=SATISFIES_TEST.",
                    "If disposition.matched_excluded_role=NONE, use disposition.test_effect=SATISFIES_TEST only when this choice itself proves normal-instance membership; otherwise use DOES_NOT_DECIDE_TEST.",
                    "If the choice axis does not describe the answer subject, use disposition.matched_excluded_role=NONE and disposition.test_effect=DOES_NOT_DECIDE_TEST for NORMAL_INSTANCE_GUARD, and set choice_inclusion based on whether rows with this choice value should contribute to the requested answer.",
                    "Do not copy role definitions into the output.",
                    "explicit_user_override_evidence contains copied question or conversation-resolution text only when the user explicitly asks for that matched excluded state, raw records, all records, or a non-normal population.",
                    "For NORMAL_INSTANCE_GUARD, explicit_user_override_applies=true requires non-empty explicit_user_override_evidence and disposition.matched_excluded_role must be a role from the profile.",
                ),
            ),
            builder.instruction_block(
                "Test Effect Semantics",
                (
                    "test_effect=SATISFIES_TEST when this choice value itself satisfies that membership test.",
                    "test_effect=CONFLICTS_WITH_TEST when this choice value itself conflicts with that membership test.",
                    "Use DOES_NOT_DECIDE_TEST for a membership test only when this param's choice value does not affect whether rows pass that test.",
                    "Use DOES_NOT_DECIDE_TEST when another param, field, or requested constraint decides the membership test.",
                    "Do not use CONFLICTS_WITH_TEST just because this choice does not decide the test.",
                    "test_effect=UNKNOWN_TEST_EFFECT when the choice-test relationship cannot be safely classified from the prompt data.",
                    "It is valid for one choice to decide a membership test while another choice does not decide that same test.",
                    "A DOES_NOT_DECIDE_TEST result is neutral for that choice; it does not make conflicting choices allowed.",
                ),
            ),
            builder.instruction_block(
                "Finite Choice Guardrails",
                (
                    "For each finite-choice param, at least one reviewed choice must remain allowed after applying the requested answer population tests.",
                    "A param choice may describe evidence rows used to answer the fact rather than the answer_subject itself.",
                    "Do not require the choice to be an instance of answer_subject.subject_text unless the param axis actually describes that subject.",
                    "If the param population_contract has an axis_field returned on the source rows used for the requested fact, assess that choice as controlling which source rows count for the requested fact.",
                    "Do not write SATISFIES_TEST only because the user did not explicitly exclude that choice.",
                    "Do not write SATISFIES_TEST merely because the API can return rows with this choice or because omitting the param includes it.",
                    "Do not write CONFLICTS_WITH_TEST merely because the requested fact does not mention or restrict this param axis; use DOES_NOT_DECIDE_TEST when this choice does not decide that test.",
                    "The backend derives include and exclude choice sets from population_test_results and choice_inclusion.",
                    "Do not output include_values, exclude_values, unresolved_values, keep_choice_argument, remove_choice_argument, argument_comparison, membership_effect, omit, safe_to_omit, applicability_decision, or omission_safety.",
                ),
            ),
            builder.instruction_block(
                "Terminal Outcomes",
                (
                    "This turn binds sources before execution. Candidate source metadata is sufficient to select executable reads; do not require actual returned rows.",
                    "If candidate metadata proves policy access prevents required evidence, return the impossible outcome.",
                    "If required catalog input is missing and a clarification option is shown, return needs_clarification.",
                    "Terminal outcomes are provider-neutral source-binding outcomes.",
                ),
            ),
            builder.instruction_block(
                "Copying And Validity",
                (
                    "Copy requested_fact_id, source_candidate_id, population_binding_id, fulfillment_choice_id, param_decision_id, param_id, and choice_option_id values verbatim from the prompt JSON.",
                    "For metric_fit_bases and fit_basis_interpretations, copy requested_fact_id and metric_evidence_id keys from Metric fit candidates.",
                    "Do not invent endpoints, params, values, fields, formulas, labels, or calculations.",
                ),
            ),
            builder.instruction_block(
                "Output",
                ("Return the submit_source_binding tool call only.",),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(provider_schema=self._schema())

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=SOURCE_BINDING_TOOL_NAME,
                    tool_description="Submit source bindings for requested facts.",
                    input_schema=self._schema(),
                    transport_context={
                        "source_binding": self.transport_context_payload()
                    },
                ),
            )
        )

    def requested_facts_payload(self) -> dict[str, object]:
        return source_binding_requested_facts_payload(self.request)

    def grounded_values_payload(self) -> dict[str, object]:
        return source_binding_grounded_values_payload(self.request)

    def memory_context_payload(self) -> dict[str, object]:
        return source_binding_memory_context_payload(self.request)

    def source_invocation_candidate_payload(self) -> dict[str, object]:
        return source_binding_candidate_payload(self.request)

    def metric_fit_surface_payload(self) -> dict[str, object]:
        return source_binding_metric_fit_surface_payload(self.request)

    def transport_context_payload(self) -> dict[str, object]:
        return source_binding_transport_context_payload(self.request)

    def _schema(self) -> dict[str, Any]:
        registry = source_candidate_registry(self.request)
        candidate_ids = registry.prompt_candidate_ids
        candidate_param_decision_ids_by_param: dict[
            str, dict[str, tuple[str, ...]]
        ] = {}
        candidate_required_param_ids: dict[str, tuple[str, ...]] = {}
        candidate_finite_choice_values: dict[str, dict[str, tuple[str, ...]]] = {}
        candidate_row_predicate_values: dict[str, dict[str, tuple[str, ...]]] = {}
        candidate_membership_test_ids: dict[str, tuple[str, ...]] = {}
        candidate_normal_instance_test_ids: dict[str, tuple[str, ...]] = {}
        candidate_population_roles: dict[str, tuple[dict[str, object], ...]] = {}
        for candidate_id in candidate_ids:
            candidate = registry.candidates_by_id[candidate_id]
            finite_choice_param_ids = set(_finite_choice_review_param_ids(candidate))
            requested_fact_id = candidate.requested_fact_id
            candidate_param_decision_ids_by_param[candidate_id] = {
                param_id: decision_ids
                for param_id, decision_ids in param_decision_ids_by_effective_param(
                    candidate,
                    effective_param_ids=tuple(
                        str(param.get("param_id") or "")
                        for param in candidate.params
                        if isinstance(param, dict)
                    ),
                ).items()
                if param_id not in finite_choice_param_ids
            }
            required_param_ids = source_candidate_required_param_decision_ids(candidate)
            candidate_required_param_ids[candidate_id] = required_param_ids
            candidate_finite_choice_values[candidate_id] = {
                param_id: choices
                for param_id, choices in choice_values_for_effective_params(
                    candidate,
                    effective_param_ids=tuple(finite_choice_param_ids),
                ).items()
            }
            candidate_row_predicate_values[candidate_id] = _row_predicate_values(
                candidate.payload
            )
            candidate_membership_test_ids[candidate_id] = _membership_test_ids(
                self.request,
                requested_fact_id=requested_fact_id,
            )
            normal_instance_test_ids = _normal_instance_membership_test_ids(
                self.request,
                requested_fact_id=requested_fact_id,
            )
            candidate_normal_instance_test_ids[candidate_id] = normal_instance_test_ids
            candidate_population_roles[candidate_id] = _population_roles(
                candidate.payload
            )
        return build_source_binding_schema(
            **source_binding_clarification_input_ids(self.request),
            source_candidate_param_decision_ids_by_param=(
                candidate_param_decision_ids_by_param
            ),
            source_candidate_required_param_ids=candidate_required_param_ids,
            source_candidate_finite_choice_values=candidate_finite_choice_values,
            source_candidate_row_predicate_values=candidate_row_predicate_values,
            source_candidate_membership_test_ids=candidate_membership_test_ids,
            source_candidate_normal_instance_test_ids=(
                candidate_normal_instance_test_ids
            ),
            source_candidate_population_roles=candidate_population_roles,
            metric_evidence_ids_by_requested_fact=(
                source_binding_metric_evidence_ids_by_requested_fact(self.request)
            ),
            source_candidate_requested_fact_ids=(
                source_binding_prompt_candidate_requested_fact_ids(self.request)
            ),
            source_candidate_fulfillment_support_set_ids_by_answer_output=(
                source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output(
                    self.request
                )
            ),
            source_candidate_population_binding_ids=(
                source_binding_prompt_candidate_population_binding_ids(self.request)
            ),
        )


def _finite_choice_review_param_ids(candidate: Any) -> tuple[str, ...]:
    return tuple(
        param_id
        for param in candidate.params
        if isinstance(param, dict) and param_requires_finite_choice_review(param)
        for param_id in (str(param.get("param_id") or ""),)
        if param_id
    )


def _row_predicate_values(payload: dict[str, Any] | None) -> dict[str, tuple[str, ...]]:
    if not isinstance(payload, dict):
        return {}
    output: dict[str, tuple[str, ...]] = {}
    for item in payload.get("row_predicates") or ():
        if not isinstance(item, dict):
            continue
        predicate_id = str(item.get("predicate_id") or "")
        values = tuple(
            str(value) for value in item.get("allowed_values") or () if str(value)
        )
        if predicate_id and values:
            output[predicate_id] = values
    return output


def _membership_test_ids(
    request: SourceBindingRequest,
    *,
    requested_fact_id: str,
) -> tuple[str, ...]:
    fact = next(
        (item for item in request.requested_facts if item.id == requested_fact_id),
        None,
    )
    if fact is None or fact.answer_population is None:
        return ()
    return tuple(
        membership_test_key(test) for test in fact.answer_population.membership_tests
    )


def _normal_instance_membership_test_ids(
    request: SourceBindingRequest,
    *,
    requested_fact_id: str,
) -> tuple[str, ...]:
    fact = next(
        (item for item in request.requested_facts if item.id == requested_fact_id),
        None,
    )
    if fact is None or fact.answer_population is None:
        return ()
    return tuple(
        membership_test_key(test)
        for test in fact.answer_population.membership_tests
        if test.kind == AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
    )


def _population_roles(payload: dict[str, Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "role_id": role_id,
        }
        for item in payload.get("population_roles") or ()
        if isinstance(item, dict)
        for role_id in (str(item.get("role_id") or ""),)
        if role_id
    )
