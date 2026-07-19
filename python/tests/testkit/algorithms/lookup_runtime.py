from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from fervis.lookup.orchestration.pipeline import run_lookup_question
from fervis.lineage.enums import ProgramInvocationKind
from fervis.lookup.answer_program.codec import answer_program_id, decode_answer_program
from fervis.lookup.answer_program.persistence import (
    ProgramInvocation,
    StoredProgramInvocation,
    program_invocation,
)
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupRuntimePorts,
)
from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    CompletenessPolicy,
    EndpointRead,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
    FieldRequirement,
    PaginationMetadata,
    PaginationMode,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    CONVERSATION_RESOLUTION_TOOL_NAMES,
)
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.question_contract import (
    KnownInputKind,
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerSubject,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
    default_answer_population,
)
from fervis.memory.addresses import fact_address_from_payload
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)
from tests.lookup.orchestrator._payloads import (
    ReadEligibilityRetentionSpec,
    _answer_subject_payload,
    read_eligibility_response_from_fact_plan,
    read_eligibility_response_from_prompt,
)
from tests.lookup.orchestrator._runtime_ports import (
    _grounding_payload_from_prompt,
    _grounding_review_for_task,
    _time_resolution_payload_from_prompt,
)
from tests.lookup.prompt_sections import prompt_section_payload
from tests.lookup.source_binding_helpers import (
    bound_fact_plan_payload_from_fact_plan,
    plan_selection_payload_from_fact_plan,
    resolved_input_applications_for_target,
    source_binding_payload_from_fact_plan,
    source_binding_payload_from_fact_plan_with_invocation_overrides,
    source_binding_payload_for_one_call,
    source_binding_target_id_for_candidate,
    source_fulfills_keys_for_candidate,
)
from fervis.lookup.clarification import clarification_payload
from tests.testkit.assertions import subset_mismatches
from tests.testkit.catalog import catalog_from_payload
from tests.testkit.answer_program_contracts import binding_set_from_payload
from tests.testkit.question_contract_provider import (
    provider_membership_tests,
    provider_question_input_ownership,
)


def run_lookup_runtime_case(payload: dict[str, Any]) -> list[str]:
    scenario = str(payload["input"]["scenario"])
    if scenario == "grounded_identity_endpoint_variant":
        return _run_grounded_identity_endpoint_variant(payload)
    if scenario == "scripted_pattern":
        return _run_scripted_pattern(payload)
    if scenario == "callable_prior_frame":
        return _run_callable_prior_frame(payload)
    return [f"unsupported lookup runtime scenario: {scenario}"]


def _run_scripted_pattern(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    fact_plan = dict(input_payload["fact_plan"])
    planner = _ScriptedPatternPlannerPort(
        question_contract=input_payload["question_contract"],
        fact_plan=fact_plan,
        conversation_resolution=input_payload.get("conversation_resolution"),
        read_eligibility_retention_specs=_retention_specs(
            input_payload.get("read_eligibility") or (),
        ),
        grounding=input_payload.get("grounding"),
        source_binding_invocation_overrides=_source_binding_invocation_overrides(
            input_payload.get("source_binding"),
        ),
    )
    data_access = _DataAccessPort(dict(input_payload["responses"]))
    result = run_lookup_question(
        LookupRequest(
            question=str(input_payload["question"]),
            conversation_context=_conversation_context(input_payload),
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
            run_id=str(
                input_payload.get("run_id") or "run_conformance_scripted_pattern"
            ),
            runtime_values=_runtime_values(input_payload.get("runtime_values")),
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                catalog_from_payload(input_payload["catalog"])
            ),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )
    rendered_rows = _portable_rows(
        list(result.rendered_fact.rows if result.rendered_fact else ())
    )
    rendered_scalars = (
        _portable_rows(dict(result.rendered_fact.scalars))
        if result.rendered_fact and result.rendered_fact.scalars
        else {}
    )
    outcome = (
        getattr(result.fact_result, "outcome", None) if result.fact_result else None
    )
    clarifications = tuple(getattr(outcome, "clarifications", ()) or ())
    return subset_mismatches(
        actual={
            "status": result.status,
            "error": result.error,
            "answer": result.answer,
            "outcome_kind": getattr(getattr(outcome, "kind", ""), "value", ""),
            "clarifications": [clarification_payload(item) for item in clarifications],
            "rendered_rows": rendered_rows,
            "rendered_scalars": rendered_scalars,
            "proof_refs": list(
                result.rendered_fact.proof_refs
                if result.rendered_fact is not None
                else ()
            ),
            "endpoint_args": {
                request["endpointName"]: request["args"]
                for request in data_access.requests
            },
            "tool_names": planner.tool_names,
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _run_callable_prior_frame(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    program = decode_answer_program(input_payload["base_program"])
    base_bindings = binding_set_from_payload(input_payload["base_invocation"])
    base = StoredProgramInvocation(
        invocation=program_invocation(
            run_id=str(input_payload["base_run_id"]),
            program_id=answer_program_id(program),
            bindings=base_bindings,
            kind=ProgramInvocationKind.COMPILED_QUESTION,
        ),
        program=program,
    )
    planner = _ScriptedPatternPlannerPort(
        question_contract=input_payload["question_contract"],
        fact_plan={},
        conversation_resolution=input_payload["conversation_resolution"],
        grounding=input_payload.get("grounding"),
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="sales",
                known_input_resolver_results=(
                    ("pivot_mall_qi_1", "store:primary_key"),
                ),
            ),
        ),
    )
    data_access = _DataAccessPort(dict(input_payload["responses"]))
    invocation_binding = _RecordingInvocationBinding(
        run_id=str(input_payload.get("run_id") or "run_callable_frame")
    )
    conversation_id = str(input_payload.get("conversation_id") or "conversation_1")
    tenant_id = str(input_payload.get("tenant_id") or "tenant_1")
    result = run_lookup_question(
        LookupRequest(
            question=str(input_payload["question"]),
            conversation_context=_conversation_context(input_payload),
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
            run_id=invocation_binding.run_id,
            tenant_id=tenant_id,
            user_context={"conversationId": conversation_id},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                catalog_from_payload(input_payload["catalog"])
            ),
            data_access_port=data_access,
            planner_model_port=planner,
            prior_program_invocations=_PriorInvocationReader(base),
            program_invocation_binding=invocation_binding,
        ),
    )
    invocation = invocation_binding.invocation
    return subset_mismatches(
        actual={
            "status": result.status,
            "error": result.error,
            "answer": result.answer,
            "endpoint_names": [
                request["endpointName"] for request in data_access.requests
            ],
            "endpoint_count": len(data_access.requests),
            "endpoint_args": {
                request["endpointName"]: request["args"]
                for request in data_access.requests
            },
            "tool_names": planner.tool_names,
            "invocation_kind": invocation.kind.value if invocation else "",
            "base_invocation_id": (
                invocation.base_invocation_id if invocation else None
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


@dataclass(frozen=True)
class _PriorInvocationReader:
    stored: StoredProgramInvocation

    def load_prior_answered_invocation(
        self,
        *,
        run_id: str,
        conversation_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None:
        del conversation_id, tenant_id
        return self.stored if run_id == self.stored.invocation.run_id else None


@dataclass
class _RecordingInvocationBinding:
    run_id: str
    invocation: ProgramInvocation | None = None

    def bind(self, execution, *, kind, base_invocation_id):
        self.invocation = program_invocation(
            run_id=self.run_id,
            program_id=answer_program_id(execution.answer),
            bindings=execution.bindings,
            kind=kind,
            base_invocation_id=base_invocation_id,
        )
        return self.invocation


def _run_grounded_identity_endpoint_variant(payload: dict[str, Any]) -> list[str]:
    planner = _VariantGroundingPlannerPort()
    data_access = _DataAccessPort(
        {
            "list_staff_list": {
                "data": [
                    {
                        "staff_id": "staff-alice",
                        "full_name": "Alice Smith",
                        "first_name": "Alice",
                    }
                ]
            },
            "list_sale_list": {
                "data": [
                    {
                        "items": [
                            {
                                "sale_id": "sale-1",
                                "merch_shade_id": "merch-shade-lipstick",
                                "snapshot_merch_name": "Lipstick",
                            },
                            {
                                "sale_id": "sale-1",
                                "merch_shade_id": "merch-shade-mascara",
                                "snapshot_merch_name": "Mascara",
                            },
                        ]
                    }
                ]
            },
        }
    )
    result = run_lookup_question(
        LookupRequest(
            question="Which products did Alice sell today? Group them by sale.",
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-09",
                timezone="Africa/London",
            ),
            run_id="run_conformance_grounded_identity_variant",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_variant_staff_sales_catalog()),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )
    rendered_rows = _portable_rows(
        list(result.rendered_fact.rows if result.rendered_fact else ())
    )
    endpoint_args = {
        request["endpointName"]: request["args"] for request in data_access.requests
    }
    return subset_mismatches(
        actual={
            "status": result.status,
            "error": result.error,
            "rendered_rows": rendered_rows,
            "endpoint_args": endpoint_args,
            "tool_names": planner.tool_names,
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _portable_rows(rows: list[Any]) -> list[Any]:
    return json.loads(json.dumps(rows, default=str))


@dataclass
class _VariantGroundingPlannerPort:
    tool_names: list[str] = field(default_factory=list)

    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode: Any = None,
        tool_specs: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        del provider, max_thinking_tokens, system_prompt, output_mode
        tool_name = tool_specs[0].name if tool_specs else ""
        self.tool_names.append(tool_name)
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _question_contract_decisions_payload()
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload()
        elif tool_name == "submit_grounding":
            arguments = self._grounding_arguments(prompt)
        elif tool_name == "submit_read_eligibility":
            arguments = read_eligibility_response_from_prompt(
                prompt,
                retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="list_sale_list",
                        answer_value_fields=(
                            "merch_shade_id",
                            "data.items.sale_id",
                        ),
                        known_input_resolver_results=(
                            ("alice_qi_1", "staff:staff_key"),
                        ),
                    ),
                ),
            )
            return arguments
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                _pattern_sale_items_answer_plan(read_id="list_sale_list"),
                prompt=prompt,
            )
        elif tool_name == "submit_source_binding":
            arguments = source_binding_payload_for_one_call(
                self._source_binding_arguments(prompt),
                prompt=prompt,
            )
        elif tool_name == "submit_pattern_fact_plan":
            arguments = bound_fact_plan_payload_from_fact_plan(
                _pattern_sale_items_answer_plan(read_id="list_sale_list"),
                prompt=prompt,
            )
        elif tool_name == CONVERSATION_RESOLUTION_TOOL_NAME:
            arguments = {
                "outcome": {
                    "kind": "standalone_question",
                    "resolution_basis": "No prior memory is needed.",
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
        return _tool_output(tool_name=tool_name, arguments=arguments)

    def _grounding_arguments(self, prompt: str) -> dict[str, Any]:
        tasks = prompt_section_payload(prompt, "Known input binding tasks")[
            "known_input_binding_tasks"
        ]
        reviews = {
            task["known_input_id"]: _grounding_review_for_task(
                task,
                compatible_option_ids={
                    str(option["binding_option_id"])
                    for option in task["binding_options"]
                    if option["api_read"]["read_id"] == "list_staff_list"
                },
            )
            for task in tasks
        }
        return {
            "known_time_resolutions": _time_resolution_payload_from_prompt(prompt),
            "known_input_binding_reviews": reviews,
        }

    def _source_binding_arguments(self, prompt: str) -> dict[str, Any]:
        fact_sources = prompt_section_payload(prompt, "Candidate evidence sources")[
            "requested_fact_sources"
        ][0]
        relation = next(
            item
            for item in _source_options_for_fact_sources(fact_sources)
            if _candidate_has_field(item, "merch_shade_id")
        )
        binding_target_id = source_binding_target_id_for_candidate(
            prompt,
            requested_fact_id="fact_1",
            source_candidate_id=str(relation["source_candidate_id"]),
            plan_shape="list_rows",
        )
        return {
            "outcome": {
                "kind": "source_bindings",
                "bindings_for_fact_1": {
                    "plan_shape": "list_rows",
                    "primary": {
                        "binding_target_id": binding_target_id,
                        "answer_population": {
                            "population_binding_id": _candidate_binding_surface(
                                relation
                            )["population_bindings"][0]["population_binding_id"],
                            "intent_text": "products did Alice sell today",
                            "match_basis_explanation": (
                                "The question asks for sale item rows for Alice."
                            ),
                        },
                        "fulfillment_decisions": source_fulfills_keys_for_candidate(
                            relation,
                            key_ids_by_answer_output={
                                "answer_1": "merch_shade_key",
                                "answer_2": "sale_key",
                            },
                            row_path_ids_by_answer_output={"answer_2": "items"},
                        ),
                        "param_decisions": {},
                        "resolved_input_applications": (
                            resolved_input_applications_for_target(
                                prompt,
                                binding_target_id=binding_target_id,
                                selections=(
                                    {
                                        "value_id": (
                                            "grounded_input_1_staff_staff_key_"
                                            "staff_id_staff_alice"
                                        ),
                                        "value_component": "canonical_key",
                                        "target_kind": "request_parameter",
                                        "target_id": "staff_id",
                                    },
                                    {
                                        "value_id": "grounded_input_2",
                                        "value_component": "instant",
                                        "target_kind": "request_parameter",
                                        "target_id": "start_date",
                                    },
                                    {
                                        "value_id": "grounded_input_2",
                                        "value_component": "instant",
                                        "target_kind": "request_parameter",
                                        "target_id": "end_date",
                                    },
                                ),
                            )
                        ),
                        "row_predicate_reviews": {},
                        "finite_choice_param_reviews": {},
                    },
                },
            }
        }


@dataclass
class _ScriptedPatternPlannerPort:
    question_contract: dict[str, Any]
    fact_plan: dict[str, Any]
    conversation_resolution: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
    read_eligibility_retention_specs: tuple[ReadEligibilityRetentionSpec, ...] = ()
    source_binding_invocation_overrides: tuple[dict[str, Any], ...] = ()
    tool_names: list[str] = field(default_factory=list)

    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode: Any = None,
        tool_specs: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        del provider, max_thinking_tokens, system_prompt, output_mode
        tool_name = tool_specs[0].name if tool_specs else ""
        self.tool_names.append(tool_name)
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _scripted_conversation_resolution_payload(
                prompt,
                payload=self.conversation_resolution,
            )
        elif tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _scripted_question_contract_payload(self.question_contract)
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _scripted_query_enrichment_payload(
                self.question_contract,
            )
        elif tool_name == "submit_grounding":
            arguments = _scripted_grounding_payload(prompt, payload=self.grounding)
        elif tool_name == "submit_read_eligibility":
            if self.read_eligibility_retention_specs:
                return read_eligibility_response_from_prompt(
                    prompt,
                    retention_specs=self.read_eligibility_retention_specs,
                )
            return read_eligibility_response_from_fact_plan(prompt, self.fact_plan)
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                self.fact_plan,
                prompt=prompt,
            )
        elif tool_name == "submit_source_binding":
            if self.source_binding_invocation_overrides:
                arguments = (
                    source_binding_payload_from_fact_plan_with_invocation_overrides(
                        self.fact_plan,
                        prompt=prompt,
                        invocation_overrides=self.source_binding_invocation_overrides,
                    )
                )
            else:
                arguments = source_binding_payload_from_fact_plan(
                    self.fact_plan,
                    prompt=prompt,
                )
        elif tool_name == "submit_pattern_fact_plan":
            arguments = bound_fact_plan_payload_from_fact_plan(
                self.fact_plan,
                prompt=prompt,
            )
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
        return _tool_output(tool_name=tool_name, arguments=arguments)


def _scripted_grounding_payload(
    prompt: str,
    *,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("compatible_resolver_reads"):
        return _grounding_payload_for_compatible_resolver_reads(prompt, payload=payload)
    return _grounding_payload_from_prompt(prompt)


def _grounding_payload_for_compatible_resolver_reads(
    prompt: str,
    *,
    payload: dict[str, Any],
) -> dict[str, Any]:
    tasks = prompt_section_payload(prompt, "Known input binding tasks")[
        "known_input_binding_tasks"
    ]
    compatible_by_input = _compatible_resolver_reads(payload)
    compatible_surfaces_by_input = _compatible_result_surfaces(payload)
    return {
        "known_time_resolutions": _time_resolution_payload_from_prompt(prompt),
        "known_input_binding_reviews": {
            task["known_input_id"]: _grounding_review_for_task(
                task,
                compatible_option_ids={
                    str(option["binding_option_id"])
                    for option in task["binding_options"]
                    if _grounding_option_is_compatible(
                        option,
                        known_input_id=str(task["known_input_id"]),
                        compatible_by_input=compatible_by_input,
                        compatible_surfaces_by_input=compatible_surfaces_by_input,
                    )
                },
                request_values_for_option=_scripted_resolver_request_values,
                match_fields_for_option=_scripted_resolver_match_fields,
            )
            for task in tasks
        },
    }


def _grounding_option_is_compatible(
    option: dict[str, Any],
    *,
    known_input_id: str,
    compatible_by_input: dict[str, set[str]],
    compatible_surfaces_by_input: dict[str, set[str]],
) -> bool:
    if option["api_read"]["read_id"] not in compatible_by_input.get(
        known_input_id, set()
    ):
        return False
    compatible_surfaces = compatible_surfaces_by_input.get(known_input_id)
    if compatible_surfaces is None:
        return True
    result = option["canonical_result"]
    result_surface = f"entity {result['entity_kind']}:{result['key_id']}"
    return result_surface in compatible_surfaces


def _scripted_resolver_request_values(
    option: dict[str, Any],
    *,
    lookup_text: str,
) -> dict[str, Any]:
    params = option["api_read"]["input_params"]
    if not params:
        return {}
    return {str(params[0]["param_ref"]): lookup_text}


def _scripted_resolver_match_fields(option: dict[str, Any]) -> list[str]:
    fields = [
        field for row in option["api_read"]["response_rows"] for field in row["fields"]
    ]
    named = [str(field["path"]) for field in fields if "name" in str(field["field_id"])]
    if named:
        return named
    return [
        str(component["field_path"])
        for component in option["canonical_result"]["components"]
    ]


def _compatible_result_surfaces(payload: dict[str, Any]) -> dict[str, set[str]]:
    raw = payload.get("compatible_result_surfaces")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AssertionError("grounding.compatible_result_surfaces must be an object")
    output: dict[str, set[str]] = {}
    for known_input_id, surfaces in raw.items():
        if not isinstance(surfaces, list):
            raise AssertionError(
                "grounding.compatible_result_surfaces values must be lists"
            )
        output[str(known_input_id)] = {str(surface) for surface in surfaces}
    return output


def _compatible_resolver_reads(payload: dict[str, Any]) -> dict[str, set[str]]:
    raw = payload.get("compatible_resolver_reads")
    if not isinstance(raw, dict):
        raise AssertionError("grounding.compatible_resolver_reads must be an object")
    output: dict[str, set[str]] = {}
    for known_input_id, read_ids in raw.items():
        if not isinstance(read_ids, list):
            raise AssertionError(
                f"grounding.compatible_resolver_reads.{known_input_id} must be a list"
            )
        output[str(known_input_id)] = {str(read_id) for read_id in read_ids}
    return output


def _scripted_question_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    answer_outputs = tuple(payload.get("answer_outputs") or ("answer",))
    subject_text = str(
        payload.get("subject_text") or payload.get("fact_description") or "records"
    )
    fact_description = str(payload.get("fact_description") or subject_text)
    known_inputs = tuple(
        item for item in payload.get("known_inputs") or () if isinstance(item, dict)
    )
    question_inputs = [
        _scripted_question_input(index=index, payload=item)
        for index, item in enumerate(known_inputs, start=1)
    ]
    answer_expression = _scripted_answer_expression(payload)
    group_key = answer_expression.get("group_key")
    group_refs = tuple(
        str(ref)
        for ref in (
            group_key.pop("question_input_refs", ())
            if isinstance(group_key, dict)
            else ()
        )
    )
    result_limit_refs = tuple(
        str(item["input_ref"])
        for item in question_inputs
        if item.get("role") == LiteralInputRole.RESULT_LIMIT.value
    )
    population_refs = tuple(
        str(item["input_ref"])
        for item in question_inputs
        if str(item["input_ref"]) not in {*group_refs, *result_limit_refs}
    )
    ownership = provider_question_input_ownership(
        group_key_input_refs=group_refs,
        population_input_refs_by_test_id=(
            {
                f"input_constraint_{index}": (input_ref,)
                for index, input_ref in enumerate(population_refs, start=1)
            }
        ),
        result_limit_input_ref=result_limit_refs[0] if result_limit_refs else "",
    )
    population = default_answer_population(
        description=fact_description,
        subject_text=subject_text,
        instance_interpretation=RequestedFactAnswerSubject(
            subject_text=subject_text
        ).instance_interpretation,
    ).to_question_contract_dict()
    population["membership_tests"] = provider_membership_tests(
        population["membership_tests"],
        ownership=ownership,
    )
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": question_inputs,
        "answer_requests": [
            {
                "answer_fact": fact_description,
                "answer_expression": answer_expression,
                "question_input_uses": list(ownership.question_input_uses),
                "answer_subject": _answer_subject_payload(subject_text),
                "answer_population": population,
                "answer_outputs": [
                    _scripted_answer_output(
                        answer_output,
                        default_role=str(payload.get("support_role") or "ROW_COUNT"),
                    )
                    for answer_output in answer_outputs
                ],
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }


def _question_contract_decision(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_basis": "The current wording supports the selected outcome.",
        "outcome": outcome,
    }


def _scripted_answer_expression(payload: dict[str, Any]) -> dict[str, Any]:
    family = str(payload.get("answer_expression_family") or "list_rows")
    output = {"family": family}
    if isinstance(payload.get("group_key"), dict):
        output["group_key"] = dict(payload["group_key"])
    return output


def _scripted_answer_output(
    payload: object,
    *,
    default_role: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"description": str(payload), "role": default_role}
    return {
        "description": str(payload.get("description") or ""),
        "role": str(payload.get("role") or default_role),
    }


def _scripted_answer_output_description(payload: object) -> str:
    if isinstance(payload, dict):
        return str(payload.get("description") or "")
    return str(payload)


def _scripted_answer_output_support_role(payload: object, *, default: str) -> str:
    if isinstance(payload, dict):
        return str(payload.get("support_role") or payload.get("role") or default)
    return default


def _scripted_question_input(*, index: int, payload: dict[str, Any]) -> dict[str, Any]:
    kind = KnownInputKind(str(payload["kind"]))
    text = str(payload["text"])
    source = str(payload.get("source") or "question_context")
    output: dict[str, Any] = {
        "input_ref": str(payload.get("input_ref") or f"input_{index}"),
        "source": source,
        "kind": kind.value,
        "inventory_check": {
            "why_this_is_an_input": f"{text} is a declared question input",
        },
    }
    if kind == KnownInputKind.LITERAL:
        role = LiteralInputRole(str(payload["role"]))
        output["value_source_text"] = text
        output["operand_text"] = str(
            payload.get("operand_text") or payload.get("resolved_value_text") or text
        )
        output["role"] = role.value
        if payload.get("value_meaning_hint"):
            output["value_meaning_hint"] = str(payload["value_meaning_hint"])
        if payload.get("field_label_text"):
            output["field_label_text"] = str(payload["field_label_text"])
        if source == KnownInputSource.CONVERSATION_RESOLUTION.value:
            output["occurrence"] = int(payload.get("occurrence") or 1)
            output["resolved_input_ref"] = str(
                payload.get("resolved_input_ref") or f"cr_input_{index}"
            )
    if kind == KnownInputKind.ROW_SET_REFERENCE:
        output["reference_text"] = text
        output["source"] = "conversation_resolution"
        output["occurrence"] = int(payload.get("occurrence") or 1)
        output["resolved_input_ref"] = str(
            payload.get("resolved_input_ref") or f"cr_input_{index}"
        )
    return output


def _scripted_query_enrichment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    resource_terms = list(payload.get("resource_terms") or ())
    return {
        "requested_fact_resource_name_matches": [
            {
                "requested_fact_id": "fact_1",
                "answer_output_resource_lineage": [
                    {
                        "answer_output_id": f"answer_{index}",
                        "support_role": _scripted_answer_output_support_role(
                            answer_output,
                            default=str(payload.get("support_role") or "ROW_COUNT"),
                        ),
                        "source_text": _scripted_answer_output_description(
                            answer_output
                        ),
                        "matching_resource_names": resource_terms,
                    }
                    for index, answer_output in enumerate(
                        payload.get("answer_outputs") or ("answer",),
                        start=1,
                    )
                    if resource_terms
                ],
            }
        ],
        "entity_target_catalog_search_terms": list(
            payload.get("entity_target_catalog_search_terms") or ()
        ),
    }


def _retention_specs(items: Any) -> tuple[ReadEligibilityRetentionSpec, ...]:
    if not isinstance(items, list):
        return ()
    return tuple(
        ReadEligibilityRetentionSpec(
            requested_fact_id=str(item.get("requested_fact_id") or "fact_1"),
            read_id=str(item.get("read_id") or ""),
            source_candidate_id=str(item.get("source_candidate_id") or ""),
            row_path_ids=tuple(item.get("row_path_ids") or ()),
            answer_value_fields=tuple(item.get("answer_value_fields") or ()),
            measured_value_fields=tuple(item.get("measured_value_fields") or ()),
            group_key_fields=tuple(item.get("group_key_fields") or ()),
            population_scope_fields=tuple(item.get("population_scope_fields") or ()),
            known_input_resolver_results=tuple(
                (
                    str(binding.get("known_input_id") or ""),
                    str(binding.get("resolver_result") or ""),
                )
                for binding in item.get("known_input_bindings") or ()
                if isinstance(binding, dict)
            ),
        )
        for item in items
        if isinstance(item, dict)
    )


def _source_binding_invocation_overrides(items: Any) -> tuple[dict[str, Any], ...]:
    if items is None:
        return ()
    if not isinstance(items, list):
        raise AssertionError("source_binding must be a list")
    output = []
    for item in items:
        if not isinstance(item, dict):
            raise AssertionError("source_binding items must be objects")
        output.append(_source_binding_invocation_override(item))
    return tuple(output)


def _source_binding_invocation_override(item: dict[str, Any]) -> dict[str, Any]:
    requested_fact_id = str(item.get("requested_fact_id") or "")
    if not requested_fact_id:
        raise AssertionError("source_binding item requires requested_fact_id")
    choices = item.get("param_value_choices") or {}
    default_param_ids = item.get("default_params") or []
    row_predicate_choices = item.get("row_predicate_choices") or {}
    resolved_input_applications = item.get("resolved_input_applications") or []
    if not isinstance(choices, dict):
        raise AssertionError("source_binding param_value_choices must be an object")
    if not isinstance(default_param_ids, list):
        raise AssertionError("source_binding default_params must be a list")
    if not isinstance(row_predicate_choices, dict):
        raise AssertionError("source_binding row_predicate_choices must be an object")
    if not isinstance(resolved_input_applications, list):
        raise AssertionError(
            "source_binding resolved_input_applications must be a list"
        )
    if not any(
        (
            choices,
            default_param_ids,
            row_predicate_choices,
            resolved_input_applications,
        )
    ):
        raise AssertionError(
            "source_binding item requires param_value_choices, default_params, "
            "row_predicate_choices, or resolved_input_applications"
        )
    return {
        "requested_fact_id": requested_fact_id,
        "use_default_param_ids": tuple(str(item) for item in default_param_ids),
        "row_predicate_choices": {
            str(field_id): tuple(str(value) for value in values)
            for field_id, values in row_predicate_choices.items()
        },
        "resolved_input_applications": tuple(resolved_input_applications),
        "param_decisions": {
            str(param_id): _param_value_choice_decision(
                param_id=str(param_id),
                payload=choice,
            )
            for param_id, choice in choices.items()
        },
    }


def _param_value_choice_decision(
    *,
    param_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AssertionError(f"{param_id} choice must be an object")
    if not isinstance(payload.get("selected_values"), list):
        raise AssertionError(f"{param_id} requires selected_values")
    if not isinstance(payload.get("all_values"), list):
        raise AssertionError(f"{param_id} requires all_values")
    selected_values = [str(value) for value in payload["selected_values"]]
    all_values = [str(value) for value in payload["all_values"]]
    return {
        "population_intent": str(
            payload.get("population_intent") or f"selected {param_id} values"
        ),
        "match_basis_explanation": str(
            payload.get("basis") or f"{param_id} values are selected for this source."
        ),
        "population_choice_set": {
            "include_values": selected_values,
            "exclude_values": [
                value for value in all_values if value not in selected_values
            ],
        },
    }


def _scripted_conversation_resolution_payload(
    prompt: str,
    *,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("mode") == "use_visible_memory":
        return _conversation_resolution_using_visible_memory(
            prompt,
            contextualized_question=str(payload.get("contextualized_question") or ""),
        )
    if isinstance(payload, dict) and payload.get("mode") == "select_visible_memory":
        return _conversation_resolution_selecting_visible_memory(
            prompt,
            memory_id=str(payload.get("memory_id") or ""),
            anchor_text=str(payload.get("anchor_text") or ""),
            contextualized_question=str(payload.get("contextualized_question") or ""),
            resolved_text=str(payload.get("resolved_text") or ""),
        )
    if payload:
        return dict(payload)
    current_question = _current_question_from_prompt(prompt)
    return {
        "kind": "conversation_resolution",
        "current_question_text": current_question,
        "outcome": {
            "kind": "resolved",
            "resolution_basis": "The current question is context-free.",
            "contextualized_question": current_question,
            "clauses": [
                {
                    "current_clause_text": current_question,
                    "occurrence": 1,
                    "resolved_text": current_question,
                    "retained_frame_parts": [],
                    "values": [],
                }
            ],
        },
    }


def _conversation_resolution_using_visible_memory(
    prompt: str,
    *,
    contextualized_question: str,
) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    context_sources = (
        prompt_section_payload(prompt, "Context sources").get("context_sources") or []
    )
    components = [
        component
        for source in context_sources
        if isinstance(source, dict)
        for component in _meaning_components_for_source(source)
    ]
    resolved_question = contextualized_question or current_question
    return {
        "kind": "conversation_resolution",
        "current_question_text": current_question,
        "outcome": {
            "kind": "resolved",
            "resolution_basis": (
                "Visible prior meaning supplies the context omitted by the current "
                "question."
            ),
            "contextualized_question": resolved_question,
            "clauses": [
                {
                    "current_clause_text": current_question,
                    "occurrence": 1,
                    "resolved_text": resolved_question,
                    "retained_frame_parts": [],
                    "values": [
                        {
                            "value_id": f"memory_value_{index}",
                            "resolved_text": component["resolved_text"],
                            "frame_parameter": {"kind": "none"},
                            "sources": [_context_anchor_source(component)],
                        }
                        for index, component in enumerate(components, start=1)
                    ],
                }
            ],
        },
    }


def _conversation_resolution_selecting_visible_memory(
    prompt: str,
    *,
    memory_id: str,
    anchor_text: str,
    contextualized_question: str,
    resolved_text: str,
) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    anchor = anchor_text or current_question
    if not memory_id:
        raise AssertionError("select_visible_memory requires memory_id")
    component = _meaning_component_for_memory_id(
        prompt,
        memory_id=memory_id,
        resolved_text=resolved_text,
    )
    resolved = resolved_text or component["resolved_text"]
    resolved_question = contextualized_question or current_question
    return {
        "kind": "conversation_resolution",
        "current_question_text": current_question,
        "outcome": {
            "kind": "resolved",
            "resolution_basis": (
                "The selected visible memory supplies the referent omitted by the "
                "current question."
            ),
            "contextualized_question": resolved_question,
            "clauses": [
                {
                    "current_clause_text": current_question,
                    "occurrence": 1,
                    "resolved_text": resolved_question,
                    "retained_frame_parts": [],
                    "values": [
                        {
                            "value_id": "selected_memory_value",
                            "resolved_text": resolved,
                            "frame_parameter": {"kind": "none"},
                            "sources": [
                                {
                                    "kind": "current_span",
                                    "text": anchor,
                                    "occurrence": 1,
                                },
                                _context_anchor_source(component),
                            ],
                        }
                    ],
                }
            ],
        },
    }


def _meaning_component_for_memory_id(
    prompt: str,
    *,
    memory_id: str,
    resolved_text: str,
) -> dict[str, str]:
    context_sources = (
        prompt_section_payload(prompt, "Context sources").get("context_sources") or []
    )
    for source in context_sources:
        if not isinstance(source, dict):
            continue
        for component in _meaning_components_for_source(source):
            if component.get("anchor_id") == memory_id:
                return {
                    **component,
                    "resolved_text": resolved_text or component["resolved_text"],
                }
    raise AssertionError(f"visible memory not found: {memory_id}")


def _meaning_components_for_source(source: dict[str, Any]) -> list[dict[str, str]]:
    components: list[dict[str, str]] = []
    for anchor in source.get("meaning_anchors") or ():
        if not isinstance(anchor, dict):
            continue
        anchor_id = str(anchor.get("anchor_id") or "")
        source_text = str(anchor.get("text") or "")
        if not anchor_id or not source_text:
            continue
        components.append(
            {
                "kind": _meaning_component_kind(str(anchor.get("kind") or "")),
                "source_id": str(source.get("source_id") or ""),
                "source_text": source_text,
                "anchor_id": anchor_id,
                "resolved_text": source_text,
            }
        )
    return components


def _meaning_component_kind(anchor_kind: str) -> str:
    if anchor_kind == "entity_identity":
        return "entity"
    if anchor_kind == "time_scope":
        return "scope"
    if anchor_kind == "row_set":
        return "row_set"
    if anchor_kind == "scalar_value":
        return "value"
    return "other"


def _context_anchor_source(component: dict[str, str]) -> dict[str, str]:
    return {
        "kind": "context_anchor",
        "source_id": component["source_id"],
        "anchor_id": component["anchor_id"],
    }


def _current_question_from_prompt(prompt: str) -> str:
    marker = "Current question:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def _conversation_context(payload: dict[str, Any]) -> dict[str, Any]:
    context = dict(payload.get("conversation_context") or {})
    artifacts = tuple(
        _fact_artifact(item) for item in payload.get("memory_artifacts") or ()
    )
    if artifacts:
        context["factArtifacts"] = [item.to_dict() for item in artifacts]
    return context


def _fact_artifact(payload: dict[str, Any]) -> Any:
    return build_fact_artifact(
        artifact_id=str(payload["artifact_id"]),
        outcome=FactOutcome(str(payload.get("outcome") or FactOutcome.ANSWERED)),
        source_question=str(payload.get("source_question") or ""),
        source_answer=str(payload.get("source_answer") or ""),
        provenance=dict(payload.get("provenance") or {}),
        addresses=tuple(
            fact_address_from_payload(item) for item in payload.get("addresses") or ()
        ),
    )


def _runtime_values(payload: Any) -> RuntimeValueContext | None:
    if not isinstance(payload, dict):
        return None
    return RuntimeValueContext(
        runtime_date=str(payload["runtime_date"]),
        timezone=str(payload["timezone"]),
    )


@dataclass
class _CatalogPort:
    catalog: RelationCatalog

    def build_relation_catalog(self) -> RelationCatalog:
        return self.catalog


@dataclass
class _DataAccessPort:
    responses: dict[str, Any]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        body, truncated, page_count = _response_body_and_meta(
            self.responses[endpoint_name]
        )
        return {
            "endpointName": endpoint_name,
            "responseStatus": 200,
            "responseBody": body,
            "truncated": truncated,
            "pageCount": page_count,
        }


def _response_body_and_meta(response: Any) -> tuple[Any, bool, int]:
    if isinstance(response, dict) and "response_body" in response:
        return (
            response["response_body"],
            bool(response.get("truncated") is True),
            int(response.get("page_count") or 1),
        )
    return response, False, 1


def _variant_staff_sales_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="list_sale_list",
                endpoint_name="list_sale_list",
                path="/v1/sales/",
                resource_names=("sale",),
                params=(
                    CatalogParam(
                        ref="list_sale_list.query.staff_id",
                        name="staff_id",
                        source=ParamSource.QUERY,
                        type="string",
                        entity_target=EntityKeyComponentTarget(
                            entity_kind="staff",
                            key_id="staff_key",
                            component_id="staff_id",
                        ),
                    ),
                    CatalogParam(
                        ref="list_sale_list.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                    CatalogParam(
                        ref="list_sale_list.query.end_date",
                        name="end_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                    CatalogParam(
                        ref="list_sale_list.query.include_items",
                        name="include_items",
                        source=ParamSource.QUERY,
                        type="boolean",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                    RowPath(
                        id="items",
                        path="data.items",
                        cardinality=RowCardinality.MANY,
                        parent_path="data",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.data.sale_id",
                        path="data.sale_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.data.items.sale_id",
                        path="data.items.sale_id",
                        row_path_id="items",
                        type="string",
                        requirements=(
                            FieldRequirement(
                                param_ref="list_sale_list.query.include_items",
                                value=True,
                            ),
                        ),
                    ),
                    CatalogField(
                        ref="field.data.items.merch_shade_id",
                        path="data.items.merch_shade_id",
                        row_path_id="items",
                        type="string",
                        requirements=(
                            FieldRequirement(
                                param_ref="list_sale_list.query.include_items",
                                value=True,
                            ),
                        ),
                    ),
                    CatalogField(
                        ref="field.data.items.snapshot_merch_name",
                        path="data.items.snapshot_merch_name",
                        row_path_id="items",
                        type="string",
                        requirements=(
                            FieldRequirement(
                                param_ref="list_sale_list.query.include_items",
                                value=True,
                            ),
                        ),
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="sale_key",
                        entity_kind="sale",
                        components=(
                            CandidateKeyComponent(
                                id="sale_id",
                                field_ref="field.data.sale_id",
                            ),
                        ),
                        primary=True,
                        stable=True,
                    ),
                ),
                entity_references=(
                    EntityReference(
                        id="item_merch_shade",
                        target_entity_kind="merch_shade",
                        target_key_id="merch_shade_key",
                        components=(
                            EntityReferenceComponent(
                                target_component_id="merch_shade_id",
                                local_field_ref="field.data.items.merch_shade_id",
                            ),
                        ),
                    ),
                    EntityReference(
                        id="item_sale",
                        target_entity_kind="sale",
                        target_key_id="sale_key",
                        components=(
                            EntityReferenceComponent(
                                target_component_id="sale_id",
                                local_field_ref="field.data.items.sale_id",
                            ),
                        ),
                    ),
                ),
                source_metadata={
                    "description": "List sales with optional nested sale items."
                },
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
            EndpointRead(
                id="list_staff_list",
                endpoint_name="list_staff_list",
                path="/v1/staff/",
                resource_names=("staff",),
                params=(
                    CatalogParam(
                        ref="list_staff_list.query.name",
                        name="name",
                        source=ParamSource.QUERY,
                        type="string",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.staff.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.staff.full_name",
                        path="data.full_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.staff.first_name",
                        path="data.first_name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="staff_key",
                        entity_kind="staff",
                        components=(
                            CandidateKeyComponent(
                                id="staff_id",
                                field_ref="field.staff.staff_id",
                            ),
                        ),
                        primary=True,
                        context_field_refs=(
                            "field.staff.full_name",
                            "field.staff.first_name",
                        ),
                    ),
                ),
                source_metadata={"description": "List staff people by name."},
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
            EndpointRead(
                id="get_merch_shade",
                endpoint_name="get_merch_shade",
                resource_names=("merch shade authority",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.ONE),
                ),
                fields=(
                    CatalogField(
                        ref="field.merch_shade.merch_shade_id",
                        path="data.merch_shade_id",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="merch_shade_key",
                        entity_kind="merch_shade",
                        components=(
                            CandidateKeyComponent(
                                id="merch_shade_id",
                                field_ref="field.merch_shade.merch_shade_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
            ),
        )
    )


def _question_contract_decisions_payload() -> dict[str, Any]:
    fact = _variant_grounding_question_contract().requested_facts[0]
    ownership = provider_question_input_ownership(
        population_input_refs_by_test_id={
            "input_constraint_1": ("input_1",),
            "input_constraint_2": ("input_2",),
        }
    )
    population = default_answer_population(
        description=fact.description,
        subject_text="products",
        instance_interpretation=RequestedFactAnswerSubject(
            subject_text="products"
        ).instance_interpretation,
    ).to_question_contract_dict()
    population["membership_tests"] = provider_membership_tests(
        population["membership_tests"],
        ownership=ownership,
    )
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [
            {
                "input_ref": "input_1",
                "source": "question_context",
                "kind": KnownInputKind.LITERAL.value,
                "value_source_text": "Alice",
                "operand_text": "Alice",
                "value_meaning_hint": "staff member",
                "role": LiteralInputRole.REFERENCE_VALUE.value,
                "inventory_check": {
                    "why_this_is_an_input": "Alice is a declared question input"
                },
            },
            {
                "input_ref": "input_2",
                "source": "question_context",
                "kind": KnownInputKind.LITERAL.value,
                "value_source_text": "today",
                "operand_text": "today",
                "role": LiteralInputRole.TIME_VALUE.value,
                "inventory_check": {
                    "why_this_is_an_input": "today is a declared question input"
                },
            },
        ],
        "answer_requests": [
            {
                "answer_fact": fact.description,
                "answer_expression": {"family": "list_rows"},
                "question_input_uses": list(ownership.question_input_uses),
                "answer_subject": _answer_subject_payload("products"),
                "answer_population": population,
                "answer_outputs": [
                    {"description": "products sold", "role": "ANSWER_VALUE"},
                    {
                        "description": "sale containing each product",
                        "role": "ANSWER_VALUE",
                    },
                ],
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }


def _variant_grounding_question_contract() -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description=(
                    "Identify the products sold by Alice today, grouped by each "
                    "individual sale."
                ),
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer_1",
                        role="ANSWER_VALUE",
                        description="products sold by sale",
                    ),
                    RequestedFactAnswerOutput(
                        id="answer_2",
                        role="ANSWER_VALUE",
                        description="sale grouping",
                    ),
                ),
                known_inputs=(
                    RequestedFactLiteralInput(
                        id="fact_1_input_1",
                        source=KnownInputSource.QUESTION_CONTEXT,
                        text="Alice",
                        resolved_value_text="Alice",
                        value_meaning_hint="staff member",
                        role=LiteralInputRole.REFERENCE_VALUE,
                    ),
                    RequestedFactLiteralInput(
                        id="fact_1_input_2",
                        source=KnownInputSource.QUESTION_CONTEXT,
                        text="today",
                        resolved_value_text="today",
                        role=LiteralInputRole.TIME_VALUE,
                    ),
                ),
            ),
        )
    )


def _query_enrichment_payload() -> dict[str, Any]:
    return {
        "requested_fact_resource_name_matches": [
            {
                "requested_fact_id": "fact_1",
                "answer_output_resource_lineage": [
                    {
                        "answer_output_id": "answer_1",
                        "support_role": "ROW_COUNT",
                        "source_text": "sales",
                        "matching_resource_names": ["sale"],
                    }
                ],
            }
        ],
        "entity_target_catalog_search_terms": [
            {
                "target_id": "input_1",
                "catalog_search_terms": [
                    {
                        "basis": "staff can identify Alice.",
                        "term": "staff",
                    }
                ],
            }
        ],
    }


def _param_decisions(
    relation: dict[str, Any],
    *,
    bindings: dict[str, str],
) -> dict[str, dict[str, str]]:
    return {
        param_id: {
            "population_intent": intent_text,
            "match_basis_explanation": (
                "The selected option matches the requested source argument scope."
            ),
            "param_decision_id": _bind_param_decision_option(
                relation,
                param_id,
                intent_text,
            )["param_decision_id"],
        }
        for param_id, intent_text in bindings.items()
    }


def _bind_param_decision_option(
    relation: dict[str, Any],
    param_id: str,
    intent_text: str,
) -> dict[str, str]:
    for param in _candidate_binding_surface(relation).get("params") or ():
        if not isinstance(param, dict) or param.get("param_id") != param_id:
            continue
        matching_options = tuple(
            option
            for option in param.get("decision_options") or ()
            if isinstance(option, dict)
            and option.get("decision") == "bind"
            and _binding_value_matches_intent(
                param_id=param_id,
                intent_text=intent_text,
                label=str(option.get("value") or ""),
                value_component=str(option.get("value_component") or ""),
            )
        )
        if len(matching_options) == 1:
            return {
                "param_decision_id": str(
                    matching_options[0]["param_decision_id"]
                )
            }
    raise AssertionError(f"missing bind option for {param_id}")


def _binding_value_matches_intent(
    *,
    param_id: str,
    intent_text: str,
    label: str,
    value_component: str,
) -> bool:
    del intent_text
    if param_id == "start_date":
        return value_component == "start"
    if param_id == "end_date":
        return value_component == "end"
    if param_id == "include_items":
        return label.lower() == "true"
    return True


def _source_options_for_fact_sources(
    fact_sources: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        candidate
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
    )


def _candidate_has_field(candidate: dict[str, Any], field_id: str) -> bool:
    surface = _candidate_binding_surface(candidate)
    return any(
        isinstance(item, dict) and item.get("field_id") == field_id
        for field_source in (
            surface.get("evidence_items") or (),
            surface.get("fields") or (),
            candidate.get("fields") or (),
        )
        for item in field_source
    ) or any(
        isinstance(field, dict) and field.get("field_id") == field_id
        for row in candidate.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
    )


def _candidate_binding_surface(candidate: dict[str, Any]) -> dict[str, Any]:
    surface = candidate.get("binding_surface")
    if isinstance(surface, dict):
        return surface
    output = {
        key: candidate[key]
        for key in (
            "population_bindings",
            "params",
            "evidence_items",
            "fulfillment_choices",
        )
        if key in candidate
    }
    if "fulfillment_choices" in output:
        output["fulfillment_support_sets"] = output.pop("fulfillment_choices")
    fields = [
        field
        for row in candidate.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    ]
    if fields:
        output["fields"] = fields
    return output


def _pattern_sale_items_answer_plan(*, read_id: str) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1", "answer_2"],
                    "pattern": "list_rows",
                    "source": {"kind": "read", "read_id": read_id},
                    "output_fields": [
                        {
                            "field_id": "merch_shade_id",
                        },
                        {
                            "field_id": "sale_id",
                        },
                    ],
                }
            ],
        }
    }


def _tool_output(*, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": json.dumps(
            {"tool": tool_name, "arguments": arguments},
            default=str,
        ),
        "usage": {
            "inputTokens": 1,
            "outputTokens": 1,
            "thinkingTokens": 0,
            "costUsd": 0,
        },
    }
