"""Prompt for source binding."""

from __future__ import annotations

from typing import Any, cast

from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections import (
    resolved_inputs_for_requested_fact,
    source_binding_candidates_xml,
)
from fervis.lookup.question_contract import (
    RequestedFactAnswerPopulationMembershipTest,
    requested_fact_evidence_ref,
)
from fervis.lookup.source_binding.candidates import (
    SourceCandidate,
    SourceCandidateRegistry,
    fulfillment_preserves_row_grain,
    source_binding_candidate_payload,
    source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output,
    source_binding_prompt_candidate_population_binding_ids,
    source_candidate_required_param_decision_ids,
    source_candidate_registry,
)
from fervis.lookup.source_binding.candidates.contracts import JsonObject, JsonValue
from fervis.lookup.source_binding.closed_key_params import (
    ClosedKeyParamBindingIndex,
    closed_key_param_binding_index,
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
from fervis.lookup.source_binding.input_applications import (
    ResolvedInputApplicationSurface,
    resolved_input_application_surfaces,
)
from fervis.lookup.source_binding.membership_tests import membership_tests_by_key
from fervis.lookup.source_binding.population_effects import (
    population_test_basis_payload,
)
from fervis.lookup.source_binding.param_surface import (
    param_decision_ids_by_effective_param,
)
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingPlanFamily,
    SourceBindingTarget,
    SourceBindingTargetIndex,
    source_binding_plan_families,
    source_binding_target_index,
)
from fervis.lookup.source_binding.review_scope import (
    SourceBindingReviewScope,
    source_binding_review_scope,
)
from fervis.lookup.source_binding.review_surface import source_binding_review_surface
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
                "answer_request": fact.answer_request_model_dict(),
                "answer_outputs": [
                    {
                        "id": output.id,
                        "description": output.description,
                    }
                    for output in fact.support_answer_outputs
                ],
                "resolved_inputs": list(
                    resolved_inputs_for_requested_fact(
                        fact,
                        available_values=request.available_values,
                    )
                ),
            }
            for fact in request.requested_facts
        ]
    }


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
    target_index = source_binding_target_index(request)
    targets = _prompt_binding_targets(
        request, registry=registry, target_index=target_index
    )
    closed_key_bindings = closed_key_param_binding_index(
        request,
        targets=targets,
        candidates_by_id=registry.candidates_by_id,
    )
    input_application_surfaces = resolved_input_application_surfaces(
        request,
        targets=targets,
        candidates_by_id=registry.candidates_by_id,
        closed_key_bindings=closed_key_bindings,
    )
    review_scope = source_binding_review_scope(
        request,
        candidates_by_id=registry.candidates_by_id,
        target_index=target_index,
    )
    return {
        "current_question": request.question,
        "requested_facts": source_binding_requested_facts_payload(request),
        "memory_context": source_binding_memory_context_payload(request),
        "binding_plan_families": _plan_families_payload(
            source_binding_plan_families(
                request,
                target_index=target_index,
                visible_targets=targets,
            ),
            target_payload=lambda target: _binding_target_payload(
                target,
                request=request,
                review_scope=review_scope,
                closed_key_bindings=closed_key_bindings,
                input_application_surfaces=input_application_surfaces,
            ),
        ),
        "source_candidates_by_id": _prompt_candidates_by_id(registry.prompt_payload),
    }


def _prompt_candidates_by_id(cards: JsonObject) -> dict[str, JsonObject]:
    output: dict[str, JsonObject] = {}
    for fact_sources in _json_objects(cards.get("requested_fact_sources")):
        for context in _json_objects(fact_sources.get("source_contexts")):
            _index_prompt_cards(context.get("source_options"), output=output)
    for key in ("utility_source_candidates", "value_source_candidates"):
        _index_prompt_cards(cards.get(key), output=output)
    return output


def _index_prompt_cards(
    value: JsonValue | None,
    *,
    output: dict[str, JsonObject],
) -> None:
    for card in _json_objects(value):
        candidate_id = card.get("source_candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            output[candidate_id] = card


def _json_objects(value: JsonValue | None) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _prompt_binding_targets(
    request: SourceBindingRequest,
    *,
    registry: Any,
    target_index: SourceBindingTargetIndex | None = None,
) -> tuple[SourceBindingTarget, ...]:
    targets = target_index or source_binding_target_index(request)
    return tuple(
        target
        for target in targets.targets
        if target.source_candidate_id in registry.prompt_candidate_ids
    )


def _binding_target_payload(
    target: SourceBindingTarget,
    *,
    request: SourceBindingRequest,
    review_scope: SourceBindingReviewScope,
    closed_key_bindings: ClosedKeyParamBindingIndex,
    input_application_surfaces: dict[str, ResolvedInputApplicationSurface],
) -> dict[str, object]:
    payload = closed_key_bindings.model_visible_target_payload(target)
    surface = input_application_surfaces.get(target.binding_target_id)
    if surface is not None and (
        surface.parameter_targets_by_id
        or surface.identity_targets_by_id
        or surface.returned_field_targets_by_id
    ):
        payload["resolved_input_application"] = surface.prompt_payload()
    population_tests = _population_binding_tests(
        request,
        target=target,
        review_scope=review_scope,
    )
    if population_tests:
        payload["answer_population_test_basis"] = population_test_basis_payload(
            population_tests,
            role_text=target.requirement_id,
        )
    return payload


def _population_binding_tests(
    request: SourceBindingRequest,
    *,
    target: SourceBindingTarget,
    review_scope: SourceBindingReviewScope,
) -> tuple[RequestedFactAnswerPopulationMembershipTest, ...]:
    fact = next(
        item for item in request.requested_facts if item.id == target.requested_fact_id
    )
    if fact.answer_population is None:
        return ()
    tests_by_key = membership_tests_by_key(fact.answer_population.membership_tests)
    return tuple(
        tests_by_key[test_id]
        for test_id in review_scope.population_binding_test_ids(
            target.binding_target_id
        )
    )


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
            builder.json_section(
                "Binding plan families:",
                self.binding_plan_families_payload(),
            ),
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
                    "For each requested fact, complete its bindings_for_<requested_fact_id> object: choose one shown plan_shape and bind every role shown for that shape exactly once, including roles with no answer outputs.",
                    "Choose each binding_target_id only from its enclosing role and obey the shape's member_constraint.",
                    "The role binding also chooses its answer population, required fulfillment, params, and population reviews.",
                    "Bind sources under the fixed meanings in resolved_inputs.",
                    "Apply resolved question inputs through resolved_input_applications on the selected source invocation.",
                ),
            ),
            builder.instruction_block(
                "Source Population And Fulfillment",
                (
                    "binding_target_id identifies the selected plan member and source candidate role for this invocation.",
                    "answer_population.population_binding_id is the source candidate's population handle that this invocation uses.",
                    "When answer_population_test_basis is shown for the selected binding target, answer answer_population.population_test_results for every shown key exactly once.",
                    "Each answer-population result copies test_id, test_question, and role_scoped_test_question from answer_population_test_basis, then states because and test_effect for the selected source population.",
                    "Use DOES_NOT_DECIDE_TEST or UNKNOWN_TEST_EFFECT when the selected source population does not establish the test. Do not infer satisfaction merely because the source can return rows.",
                    "applied_filters are backend-derived grounded-value filters already attached to this source candidate.",
                    "Do not reinterpret, rematch, remove, or rewrite applied_filters.",
                    "For each answer output shown under the selected binding target, choose fulfillment_choice_id from that source candidate's fulfillment_choices. Entity outputs use a declared source candidate key or entity reference. Scalar group outputs use a declared factual value. Context labels are not selectable computation evidence.",
                    "Before choosing fulfillment, write metric_fit_bases for every metric candidate.",
                    "Metric contexts are read-only interpretation context for metric candidates.",
                    "Use metric_contexts to understand what a metric field is likely measuring from its row path, same-row sibling fields, and scope fields.",
                    "Do not copy metric_context_id, same_row_field_paths, or scope_field_paths into output.",
                    "Do not treat metric context fields as selectable fulfillment evidence.",
                    "metric_fit_bases is keyed by requested_fact_id, then metric_evidence_id from Metric fit candidates.",
                    "metric_meaning states what the reviewed metric_evidence_id appears to measure from field_path, field_type, resource_names, and the referenced metric_context.",
                    "fit_basis evaluates whether the metric is the row-level or scalar measure that should be aggregated, used to order results, compared, or otherwise computed to determine the requested answer.",
                    "A metric fits when it is the correct measure input to the requested computation. Do not reject it merely because aggregation or another later operation produces the final answer value.",
                    "After all fit_basis entries, write fit_basis_interpretations with the same keys, using only the already-written fit_basis text.",
                    "Use interpretation=FITS_REQUESTED_ANSWER when the written fit_basis says the metric fits that role; otherwise use DOES_NOT_FIT_REQUESTED_ANSWER.",
                    "A fulfillment_choice_id is valid only when its metric_measure_evidence and row_count_basis_evidence ids have interpretation=FITS_REQUESTED_ANSWER.",
                    "For metric_operation=count_rows, evaluate the count of rows in that population, not the raw population object.",
                    "Decide final source population, row grain, fulfillment, and param membership from the source-binding candidate data and the metric_fit_bases interpretations.",
                    "When multiple source candidates can fulfill the same answer output, choose the invocation whose fitted metric evidence set most directly measures or computes the requested fact.",
                    "If you choose a fitted metric evidence set that is less direct than another available fitted set, the source population or fulfillment basis must explain why that invocation is still stronger.",
                ),
            ),
            builder.instruction_block(
                "Predicate Applications",
                (
                    "resolved_input_applications is inside one role binding. Each item selects one resolved_values entry by value_id and lists its physical target applications under applications.",
                    "An application target with target_kind=request_parameter applies the resolved input to its shown request parameter.",
                    "An application target with target_kind=returned_identity keeps only returned rows whose shown candidate-key or entity-reference target equals the resolved input value.",
                    "For each shown predicate requirement, select the shown target or targets whose combined mechanics apply that predicate.",
                    "A request_parameter application means the source applies the shown predicate before returning rows.",
                    "A returned_field application means Fervis keeps returned rows satisfying the shown field, operator, and value predicate.",
                    "For threshold_value, the requested fact already fixes the value and comparison operator; choose only where that predicate is applied.",
                    "For predicate_value, select the shown application_target_id whose target and, when present, finite-choice value apply the fixed predicate.",
                    "Do not infer predicate meaning from parameter or field names.",
                    "Each selected role binding must apply every fact-local resolved input that owns an explicit population constraint and exposes a compatible component and target kind on that role.",
                    "Copy value_component from the selected value's components_by_target_kind entry for the application target's target_kind.",
                    "For returned identities, use value_component=canonical_key; the backend maps every declared key component to its declared returned field.",
                    "Resolved values with the same request_parameter_alternative_group are alternative operands. Apply every value in that group to the same request_parameter target; the backend creates one source invocation per value. Otherwise, use each target at most once.",
                    "Each application.match_basis_explanation states how that target applies the resolved input.",
                    "Write population_test_results once for the resolved value, using the combined effect of all selected applications in that item.",
                ),
            ),
            builder.instruction_block(
                "Param Binding",
                (
                    "For every shown parameter with decision_surface=single_decision and no population_contract, choose exactly one shown decision_options item when that parameter is required or has catalog choices.",
                    "Do not author param_decisions for backend_owned_param_bindings; the backend compiles those keyed bindings.",
                    "For params with decision_surface=single_decision and no population_contract, choose exactly one decision_options item.",
                    "Do not author param_decisions for params with population_contract; write finite_choice_param_reviews keyed by param_id instead.",
                ),
            ),
            builder.instruction_block(
                "Row Predicates",
                (
                    "row_predicates filter response rows when the requested population is not fully controlled by query params.",
                    "For every shown row_predicate review entry, write row_predicate_reviews keyed by predicate_id.",
                    "Review every shown predicate value exactly once against the membership tests exposed for that predicate.",
                    "The backend derives executable row filters from row_predicate_reviews.",
                ),
            ),
            builder.instruction_block(
                "Finite Choice Review Shape",
                (
                    "For every shown finite-choice review entry, write finite_choice_param_reviews[param_id].",
                    "Review every shown choice exactly once as one member of a choice set.",
                    "Use the XML response rows to understand the source row grains before applying membership tests.",
                    "Write each finite-choice param review in this order: controlled_population_role_id, role_selection_basis, population_test_basis, choice_reviews.",
                    "Choose controlled_population_role_id from population_roles for the source row population whose rows this param controls.",
                    "role_selection_basis explains why this param controls that selected role.",
                ),
            ),
            builder.instruction_block(
                "Population Test Basis",
                (
                    "In population_test_basis, write one entry for every membership-test key exposed by that review entry.",
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
                    "choice_inclusion_basis explains why rows with this choice value independently belong or do not belong in the answer computation.",
                    "choice_inclusion is row-set membership for this one choice, not selection of one parameter value. Multiple choices may be INCLUDE.",
                    "Do not exclude a choice because selecting only that value would omit rows belonging to another included choice.",
                    "population_test_results is keyed only by the membership-test keys exposed for this review entry; write every exposed key exactly once.",
                    "Do not add population_test_results keys that are not exposed for this review entry.",
                    "For non-NORMAL_INSTANCE_GUARD tests, write test_basis, population_consequence, then test_effect.",
                    "test_basis compares this choice value to the matching population_test_basis item.",
                    "population_consequence states whether and how this choice affects the requested answer population for that test.",
                    "Write test_effect last for every population_test_results item.",
                ),
            ),
            builder.instruction_block(
                "Normal Instance Guard",
                (
                    "When the membership test has kind=NORMAL_INSTANCE_GUARD, write role_match_basis, population_consequence, then disposition.",
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
                    "If disposition.matched_excluded_role is an excluded role, use disposition.test_effect=CONFLICTS_WITH_TEST.",
                    "If disposition.matched_excluded_role=NONE, use disposition.test_effect=SATISFIES_TEST only when this choice itself proves normal-instance membership; otherwise use DOES_NOT_DECIDE_TEST.",
                    "If the choice axis does not describe the answer subject, use disposition.matched_excluded_role=NONE and disposition.test_effect=DOES_NOT_DECIDE_TEST for NORMAL_INSTANCE_GUARD, and set choice_inclusion based on whether rows with this choice value should contribute to the requested answer.",
                    "Do not copy role definitions into the output.",
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
                    "Do not narrow the requested population for an unstated preference for cleaner, safer, validated, finalized, or higher-quality evidence. Exclude a choice only when the requested fact, supplied semantic authority, or a population test supports excluding its rows.",
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
                    "Copy binding_target_id, population_binding_id, fulfillment_choice_id, param_decision_id, param_id, value_id, target_id, and choice_option_id values verbatim from the prompt data.",
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

    def memory_context_payload(self) -> dict[str, object]:
        return source_binding_memory_context_payload(self.request)

    def source_invocation_candidate_payload(self) -> dict[str, object]:
        registry = source_candidate_registry(self.request)
        targets = _prompt_binding_targets(self.request, registry=registry)
        closed_key_bindings = self._closed_key_bindings(targets, registry=registry)
        payload = source_binding_candidate_payload(self.request)
        return closed_key_bindings.model_visible_candidate_payload(payload)

    def binding_plan_families_payload(self) -> dict[str, object]:
        registry = source_candidate_registry(self.request)
        target_index = source_binding_target_index(self.request)
        targets = _prompt_binding_targets(
            self.request,
            registry=registry,
            target_index=target_index,
        )
        closed_key_bindings = self._closed_key_bindings(targets, registry=registry)
        input_application_surfaces = resolved_input_application_surfaces(
            self.request,
            targets=targets,
            candidates_by_id=registry.candidates_by_id,
            closed_key_bindings=closed_key_bindings,
        )
        review_scope = source_binding_review_scope(
            self.request,
            candidates_by_id=registry.candidates_by_id,
            target_index=target_index,
        )
        families = source_binding_plan_families(
            self.request,
            target_index=target_index,
            visible_targets=targets,
        )
        return _plan_families_payload(
            families,
            target_payload=lambda target: _binding_target_payload(
                target,
                request=self.request,
                review_scope=review_scope,
                closed_key_bindings=closed_key_bindings,
                input_application_surfaces=input_application_surfaces,
            ),
        )

    def metric_fit_surface_payload(self) -> dict[str, object]:
        return source_binding_metric_fit_surface_payload(self.request)

    def transport_context_payload(self) -> dict[str, object]:
        return source_binding_transport_context_payload(self.request)

    def _schema(self) -> dict[str, Any]:
        registry = source_candidate_registry(self.request)
        target_index = source_binding_target_index(self.request)
        targets = _prompt_binding_targets(
            self.request,
            registry=registry,
            target_index=target_index,
        )
        review_scope = source_binding_review_scope(
            self.request,
            candidates_by_id=registry.candidates_by_id,
            target_index=target_index,
        )
        closed_key_bindings = self._closed_key_bindings(targets, registry=registry)
        input_application_surfaces = resolved_input_application_surfaces(
            self.request,
            targets=targets,
            candidates_by_id=registry.candidates_by_id,
            closed_key_bindings=closed_key_bindings,
        )
        target_param_decision_ids_by_param: dict[str, dict[str, tuple[str, ...]]] = {}
        target_required_param_decision_ids: dict[str, tuple[str, ...]] = {}
        target_finite_choice_values: dict[str, dict[str, tuple[str, ...]]] = {}
        target_row_predicate_values: dict[str, dict[str, tuple[str, ...]]] = {}
        target_finite_choice_test_ids: dict[str, dict[str, tuple[str, ...]]] = {}
        target_finite_choice_normal_test_ids: dict[
            str,
            dict[str, tuple[str, ...]],
        ] = {}
        target_row_predicate_test_ids: dict[str, dict[str, tuple[str, ...]]] = {}
        target_population_roles: dict[str, tuple[dict[str, object], ...]] = {}
        target_requested_fact_ids: dict[str, str] = {}
        candidate_fulfillment_supports = source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output(
            self.request
        )
        candidate_population_bindings = (
            source_binding_prompt_candidate_population_binding_ids(self.request)
        )
        target_fulfillment_supports: dict[str, dict[str, tuple[str, ...]]] = {}
        target_required_fulfillment_output_ids: dict[str, tuple[str, ...]] = {}
        target_population_binding_ids: dict[str, tuple[str, ...]] = {}
        target_population_binding_test_ids: dict[str, tuple[str, ...]] = {}
        for target in targets:
            candidate = registry.candidates_by_id[target.source_candidate_id]
            review_surface = source_binding_review_surface(candidate)
            target_id = target.binding_target_id
            finite_choice_param_ids = set(review_surface.finite_choice_params)
            requested_fact_id = target.requested_fact_id
            target_requested_fact_ids[target_id] = requested_fact_id
            target_param_decision_ids_by_param[target_id] = (
                closed_key_bindings.model_visible_param_map(
                    target_id,
                    {
                        param_id: decision_ids
                        for param_id, decision_ids in param_decision_ids_by_effective_param(
                            candidate,
                            effective_param_ids=tuple(
                                param.id for param in candidate.params
                            ),
                        ).items()
                        if param_id not in finite_choice_param_ids
                    },
                )
            )
            required_param_ids = source_candidate_required_param_decision_ids(
                candidate
            )
            visible_required_params = closed_key_bindings.model_visible_param_map(
                target_id,
                {param_id: param_id for param_id in required_param_ids},
            )
            target_required_param_decision_ids[target_id] = tuple(
                visible_required_params
            )
            target_finite_choice_values[target_id] = {
                param_id: axis.choices
                for param_id, axis in review_surface.finite_choice_params.items()
            }
            target_finite_choice_test_ids[target_id] = {
                param_id: review_scope.finite_choice_param_test_ids(
                    target_id,
                    param_id,
                )
                for param_id in target_finite_choice_values[target_id]
            }
            target_finite_choice_normal_test_ids[target_id] = {
                param_id: review_scope.finite_choice_param_normal_instance_test_ids(
                    target_id,
                    param_id,
                )
                for param_id in target_finite_choice_values[target_id]
            }
            target_row_predicate_values[target_id] = {
                predicate_id: axis.allowed_values
                for predicate_id, axis in review_surface.row_predicates.items()
            }
            target_row_predicate_test_ids[target_id] = {
                predicate_id: review_scope.row_predicate_test_ids(
                    target_id,
                    predicate_id,
                )
                for predicate_id in target_row_predicate_values[target_id]
            }
            target_population_roles[target_id] = tuple(
                {"role_id": role.role_id} for role in review_surface.population_roles
            )
            visible_fulfillment_supports = (
                closed_key_bindings.model_visible_fulfillment_supports(
                    candidate,
                    target=target,
                    candidate_fulfillment_supports=(
                        candidate_fulfillment_supports.get(candidate.id, {})
                    ),
                )
            )
            target_fulfillment_supports[target_id] = _grain_safe_fulfillment_supports(
                candidate,
                target=target,
                ordered=_target_fact_is_ordered(
                    self.request,
                    requested_fact_id=target.requested_fact_id,
                ),
                fulfillment_supports=visible_fulfillment_supports,
            )
            target_required_fulfillment_output_ids[target_id] = (
                target.required_answer_output_ids
            )
            target_population_binding_ids[target_id] = (
                candidate_population_bindings.get(
                    candidate.id,
                    (),
                )
            )
            target_population_binding_test_ids[target_id] = (
                review_scope.population_binding_test_ids(target_id)
            )
        return build_source_binding_schema(
            **source_binding_clarification_input_ids(self.request),
            target_param_decision_ids_by_param=target_param_decision_ids_by_param,
            target_required_param_decision_ids=(
                target_required_param_decision_ids
            ),
            target_resolved_input_application_schemas={
                target_id: surface.provider_schema()
                for target_id, surface in input_application_surfaces.items()
            },
            target_finite_choice_values=target_finite_choice_values,
            target_row_predicate_values=target_row_predicate_values,
            target_finite_choice_test_ids=target_finite_choice_test_ids,
            target_finite_choice_normal_instance_test_ids=(
                target_finite_choice_normal_test_ids
            ),
            target_row_predicate_test_ids=target_row_predicate_test_ids,
            target_population_roles=target_population_roles,
            target_requested_fact_ids=target_requested_fact_ids,
            metric_evidence_ids_by_requested_fact=(
                source_binding_metric_evidence_ids_by_requested_fact(self.request)
            ),
            target_fulfillment_support_set_ids_by_answer_output=target_fulfillment_supports,
            target_required_fulfillment_answer_output_ids=(
                target_required_fulfillment_output_ids
            ),
            target_population_binding_ids=target_population_binding_ids,
            target_population_binding_test_ids=(
                target_population_binding_test_ids
            ),
            plan_families=source_binding_plan_families(
                self.request,
                target_index=target_index,
                visible_targets=targets,
            ),
        )

    def _closed_key_bindings(
        self,
        targets: tuple[SourceBindingTarget, ...],
        *,
        registry: SourceCandidateRegistry,
    ) -> ClosedKeyParamBindingIndex:
        return closed_key_param_binding_index(
            self.request,
            targets=targets,
            candidates_by_id=registry.candidates_by_id,
        )


def _grain_safe_fulfillment_supports(
    candidate: SourceCandidate,
    *,
    target: SourceBindingTarget,
    ordered: bool,
    fulfillment_supports: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    if target.plan_shape != "list_rows" or not ordered:
        return fulfillment_supports
    return {
        answer_output_id: tuple(
            support_set_id
            for support_set_id in support_set_ids
            if fulfillment_preserves_row_grain(candidate, support_set_id)
        )
        for answer_output_id, support_set_ids in fulfillment_supports.items()
    }


def _target_fact_is_ordered(
    request: SourceBindingRequest,
    *,
    requested_fact_id: str,
) -> bool:
    fact = next(
        (item for item in request.requested_facts if item.id == requested_fact_id),
        None,
    )
    return bool(
        fact is not None
        and fact.answer_expression is not None
        and fact.answer_expression.is_ordered
    )


def _plan_families_payload(
    families: tuple[SourceBindingPlanFamily, ...],
    *,
    target_payload: Any | None = None,
) -> dict[str, object]:
    render_target = target_payload or (lambda target: target.to_payload())
    facts: dict[str, dict[str, object]] = {}
    for family in families:
        fact = facts.setdefault(family.requested_fact_id, {"plan_shapes": {}})
        shapes = cast(dict[str, object], fact["plan_shapes"])
        shapes[family.plan_shape] = family.payload(target_payload=render_target)
    return {"bindings_by_requested_fact": facts}
