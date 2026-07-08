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
from fervis.lookup.question_contract import requested_fact_evidence_ref
from fervis.lookup.source_binding.candidates import (
    SourceCandidateRegistry,
    source_binding_candidate_payload,
    source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output,
    source_binding_prompt_candidate_population_binding_ids,
    source_candidate_registry,
)
from fervis.lookup.source_binding.candidates.candidate_tree import (
    CandidateTreeContext,
    map_source_candidate_tree,
)
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
from fervis.lookup.source_binding.param_surface import (
    param_decision_ids_by_effective_param,
)
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
    SourceBindingTargetIndex,
    source_binding_target_index,
)
from fervis.lookup.source_binding.review_scope import source_binding_review_scope
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
                "description": fact.description,
                "answer_request": fact.answer_request_model_dict(),
                "answer_outputs": [
                    {
                        "id": output.id,
                        "description": output.description,
                    }
                    for output in fact.support_answer_outputs
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
    targets = _prompt_binding_targets(request, registry=registry)
    return {
        "current_question": request.question,
        "requested_facts": source_binding_requested_facts_payload(request),
        "grounded_values": source_binding_grounded_values_payload(request),
        "memory_context": source_binding_memory_context_payload(request),
        "binding_targets": [target.to_payload() for target in targets],
        "source_candidates_by_id": {
            candidate_id: candidate.payload or {}
            for candidate_id, candidate in registry.candidates_by_id.items()
            if candidate_id in registry.prompt_candidate_ids
        },
    }


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
                "Binding targets:",
                self.binding_targets_payload(),
            ),
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
                    "For each requested fact, select one complete compatible set of source invocations required by the selected operation plan.",
                    "Bind every operation-required role target in that set, including targets with no answer outputs.",
                    "A source invocation chooses one binding_target, answer population, fulfillment choice when required by that target, scalar param bindings, and finite-choice population reviews.",
                    "Do not bind backup alternatives or invocations outside the selected operation plan.",
                ),
            ),
            builder.instruction_block(
                "Source Population And Fulfillment",
                (
                    "binding_target_id identifies the selected plan member and source candidate role for this invocation.",
                    "answer_population.population_binding_id is the source candidate's population handle that this invocation uses.",
                    "applied_filters are backend-derived grounded-value filters already attached to this source candidate.",
                    "Do not reinterpret, rematch, remove, or rewrite applied_filters.",
                    "For each answer output shown under the selected binding target, choose fulfillment_choice_id from that source candidate's fulfillment_choices.",
                    "Before choosing fulfillment, write metric_fit_bases for every metric candidate.",
                    "Metric contexts are read-only interpretation context for metric candidates.",
                    "Use metric_contexts to understand what a metric field is likely measuring from its row path, same-row sibling fields, and scope fields.",
                    "Do not copy metric_context_id, same_row_field_paths, or scope_field_paths into output.",
                    "Do not treat metric context fields as selectable fulfillment evidence.",
                    "Consider each api_read selection_note before choosing an invocation, but treat it as advisory guidance to evaluate against the read's evidence, not as source truth.",
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
                    "Do not author param_decisions for backend_owned_param_bindings; the backend compiles those keyed bindings.",
                    "You may bind an optional non-choice param in param_decisions only when the requested fact needs that param value.",
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
                    "choice_inclusion_basis explains why rows with this choice value should be included or excluded from the answer computation, read against the requested fact and the source candidate description.",
                    "choice_inclusion states whether rows with this choice value should be included in the answer computation: INCLUDE or EXCLUDE.",
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
                    "Copy binding_target_id, population_binding_id, fulfillment_choice_id, param_decision_id, param_id, and choice_option_id values verbatim from the prompt JSON.",
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
        registry = source_candidate_registry(self.request)
        targets = _prompt_binding_targets(self.request, registry=registry)
        closed_key_bindings = self._closed_key_bindings(targets, registry=registry)
        payload = _candidate_payload_with_selection_notes(
            source_binding_candidate_payload(self.request),
            self.request,
        )
        return closed_key_bindings.model_visible_candidate_payload(
            payload
        )

    def binding_targets_payload(self) -> dict[str, object]:
        registry = source_candidate_registry(self.request)
        target_index = source_binding_target_index(self.request)
        targets = _prompt_binding_targets(
            self.request,
            registry=registry,
            target_index=target_index,
        )
        closed_key_bindings = self._closed_key_bindings(targets, registry=registry)
        return {
            "binding_targets": [
                closed_key_bindings.model_visible_target_payload(target)
                for target in targets
            ]
        }

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
        target_param_decision_ids_by_param: dict[str, dict[str, tuple[str, ...]]] = {}
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
                                str(param.get("param_id") or "")
                                for param in candidate.params
                                if isinstance(param, dict)
                            ),
                        ).items()
                        if param_id not in finite_choice_param_ids
                    },
                )
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
            target_fulfillment_supports[target_id] = (
                closed_key_bindings.model_visible_fulfillment_supports(
                    candidate,
                    target=target,
                    candidate_fulfillment_supports=candidate_fulfillment_supports.get(
                        candidate.id,
                        {},
                    ),
                )
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
        return build_source_binding_schema(
            **source_binding_clarification_input_ids(self.request),
            target_param_decision_ids_by_param=target_param_decision_ids_by_param,
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
            source_invocations_max_items=_source_invocations_max_items(
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


def _candidate_payload_with_selection_notes(
    payload: dict[str, object],
    request: SourceBindingRequest,
) -> dict[str, object]:
    notes = _selection_notes_by_fact_source(request)
    if not notes:
        return payload

    def apply_note(
        candidate: dict[str, Any],
        context: CandidateTreeContext,
    ) -> dict[str, Any]:
        note = notes.get(
            (
                context.requested_fact_id,
                str(candidate.get("source_candidate_id") or ""),
            ),
            "",
        )
        return {**candidate, "selection_note": note} if note else candidate

    return map_source_candidate_tree(
        payload,
        apply_note,
        top_level_keys=(),
    )


def _selection_notes_by_fact_source(
    request: SourceBindingRequest,
) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for plan in request.plan_selection.plan_selections:
        basis = plan.basis.strip()
        if not basis:
            continue
        for member in plan.source_members:
            key = (plan.requested_fact_id, member.source_candidate_id)
            notes = grouped.setdefault(key, [])
            if basis not in notes:
                notes.append(basis)
    return {
        key: " | ".join(notes)
        for key, notes in grouped.items()
        if notes
    }


def _source_invocations_max_items(
    *,
    target_index: SourceBindingTargetIndex,
    visible_targets: tuple[SourceBindingTarget, ...],
) -> int:
    visible_target_ids = {target.binding_target_id for target in visible_targets}
    plan_target_ids = tuple(
        target_ids
        for target_ids in target_index.target_ids_by_plan().values()
        if target_ids <= visible_target_ids
    )
    if not plan_target_ids:
        return len(visible_targets)
    target_requested_fact_ids = {
        target.binding_target_id: target.requested_fact_id for target in visible_targets
    }
    max_items_by_fact: dict[str, int] = {}
    for target_ids in plan_target_ids:
        requested_fact_ids = {
            target_requested_fact_ids[target_id] for target_id in target_ids
        }
        if len(requested_fact_ids) != 1:
            continue
        requested_fact_id = next(iter(requested_fact_ids))
        max_items_by_fact[requested_fact_id] = max(
            max_items_by_fact.get(requested_fact_id, 0),
            len(target_ids),
        )
    return sum(max_items_by_fact.values()) or len(visible_targets)
