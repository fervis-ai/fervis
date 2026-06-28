from __future__ import annotations

from collections.abc import Callable

from tests.testkit.algorithms.business_time import run_business_time_case
from tests.testkit.algorithms.catalog_selection import (
    run_catalog_selection_case,
    run_resolver_catalog_selection_case,
)
from tests.testkit.algorithms.capabilities import run_capabilities_case
from tests.testkit.algorithms.conversation_resolution import (
    run_conversation_resolution_parse_case,
    run_conversation_resolution_schema_case,
)
from tests.testkit.algorithms.fact_requirements import (
    run_fact_requirements_case,
)
from tests.testkit.algorithms.fact_plan_schema import (
    run_fact_plan_schema_case,
)
from tests.testkit.algorithms.grouped_ranked_choices import (
    run_grouped_ranked_choices_case,
)
from tests.testkit.algorithms.endpoint_response import (
    run_endpoint_response_case,
)
from tests.testkit.algorithms.execution import run_execution_proof_graph_case
from tests.testkit.algorithms.identity_set_binding import (
    run_identity_set_binding_case,
)
from tests.testkit.algorithms.lineage import (
    run_lineage_explain_case,
    run_lineage_input_lineage_case,
)
from tests.testkit.algorithms.lookup_runtime import run_lookup_runtime_case
from tests.testkit.algorithms.outcomes import run_outcomes_classify_case
from tests.testkit.algorithms.relation_engine import (
    run_calendar_relation_case,
    run_relation_engine_case,
)
from tests.testkit.algorithms.relation_catalog import (
    run_relation_catalog_case,
)
from tests.testkit.algorithms.question_run_lifecycle import (
    run_question_run_lifecycle_case,
)
from tests.testkit.algorithms.relation_contract import (
    run_relation_contract_case,
)
from tests.testkit.algorithms.value_uses import (
    run_value_contract_case,
    run_value_uses_case,
)
from tests.testkit.algorithms.memory import (
    run_conversation_memory_card_projection_case,
    run_conversation_memory_expand_activated_case,
    run_memory_activate_case,
    run_memory_answer_addresses_case,
    run_memory_available_values_case,
    run_memory_build_artifact_case,
    run_memory_identity_projection_case,
    run_memory_lineage_memory_artifacts_case,
    run_memory_lookup_projection_case,
    run_memory_outcome_address_case,
    run_memory_planner_contract_case,
    run_memory_prior_answer_request_case,
    run_memory_project_conversation_case,
)
from tests.testkit.algorithms.question_contract import (
    run_question_contract_prompt_case,
    run_question_contract_parse_case,
    run_question_contract_schema_case,
    run_question_contract_schema_validate_case,
)
from tests.testkit.algorithms.query_enrichment import (
    run_query_enrichment_parse_case,
    run_query_enrichment_prompt_case,
    run_query_enrichment_schema_case,
)
from tests.testkit.algorithms.read_eligibility import (
    run_read_eligibility_cards_case,
    run_read_eligibility_parse_case,
    run_read_eligibility_prepare_recall_case,
    run_read_eligibility_prompt_case,
    run_read_eligibility_schema_validate_case,
)
from tests.testkit.algorithms.source_binding import (
    run_source_binding_bound_params_case,
    run_source_binding_finite_choice_parse_case,
    run_source_binding_fulfillment_support_case,
    run_source_binding_metric_fit_surface_case,
    run_source_binding_metric_fit_parse_case,
    run_source_binding_parse_case,
    run_source_binding_prompt_surface_case,
    run_source_binding_row_predicate_parse_case,
    run_source_binding_row_predicates_case,
    run_source_binding_row_predicate_schema_case,
    run_source_binding_schema_surface_case,
)
from tests.testkit.adapters.host_api import run_host_api_projection_case
from tests.testkit.case_loader import ConformanceCase

CaseRunner = Callable[[dict], list[str]]

_RUNNERS: dict[tuple[str, str], CaseRunner] = {
    ("adapter", "host_api.endpoint_contract_projection"): run_host_api_projection_case,
    ("algorithm", "business_time.resolve"): run_business_time_case,
    (
        "algorithm",
        "catalog_selection.resolver_select",
    ): run_resolver_catalog_selection_case,
    ("algorithm", "catalog_selection.select"): run_catalog_selection_case,
    ("algorithm", "conversation_memory.card_projection"): (
        run_conversation_memory_card_projection_case
    ),
    ("algorithm", "conversation_memory.expand_activated"): (
        run_conversation_memory_expand_activated_case
    ),
    (
        "algorithm",
        "conversation_resolution.parse",
    ): run_conversation_resolution_parse_case,
    (
        "algorithm",
        "conversation_resolution.schema",
    ): run_conversation_resolution_schema_case,
    ("algorithm", "core.capabilities_from_schema"): run_capabilities_case,
    ("algorithm", "execution.calendar_relation"): run_calendar_relation_case,
    ("algorithm", "execution.endpoint_response"): run_endpoint_response_case,
    ("algorithm", "execution.proof_graph"): run_execution_proof_graph_case,
    ("algorithm", "execution.identity_set_binding"): run_identity_set_binding_case,
    ("algorithm", "execution.relation_engine"): run_relation_engine_case,
    ("algorithm", "lineage.explain"): run_lineage_explain_case,
    ("algorithm", "lineage.input_lineage"): run_lineage_input_lineage_case,
    ("algorithm", "lookup.orchestration"): run_lookup_runtime_case,
    ("algorithm", "memory.activate"): run_memory_activate_case,
    ("algorithm", "memory.answer_addresses"): run_memory_answer_addresses_case,
    ("algorithm", "memory.available_values"): run_memory_available_values_case,
    ("algorithm", "memory.build_artifact"): run_memory_build_artifact_case,
    ("algorithm", "memory.identity_projection"): run_memory_identity_projection_case,
    ("algorithm", "memory.lineage_memory_artifacts"): (
        run_memory_lineage_memory_artifacts_case
    ),
    ("algorithm", "memory.lookup_projection"): run_memory_lookup_projection_case,
    ("algorithm", "memory.outcome_address"): run_memory_outcome_address_case,
    ("algorithm", "memory.planner_contract"): run_memory_planner_contract_case,
    ("algorithm", "memory.prior_answer_request"): run_memory_prior_answer_request_case,
    ("algorithm", "memory.project_conversation"): run_memory_project_conversation_case,
    ("algorithm", "outcomes.classify"): run_outcomes_classify_case,
    ("algorithm", "planning.fact_endpoint_requirements"): run_fact_requirements_case,
    ("algorithm", "planning.fact_plan_schema"): run_fact_plan_schema_case,
    ("algorithm", "planning.grouped_ranked_choices"): run_grouped_ranked_choices_case,
    ("algorithm", "planning.relation_contract"): run_relation_contract_case,
    ("algorithm", "planning.value_contract"): run_value_contract_case,
    ("algorithm", "planning.value_uses"): run_value_uses_case,
    ("algorithm", "query_enrichment.parse"): run_query_enrichment_parse_case,
    ("algorithm", "query_enrichment.prompt"): run_query_enrichment_prompt_case,
    ("algorithm", "query_enrichment.schema"): run_query_enrichment_schema_case,
    ("algorithm", "question_contract.parse"): run_question_contract_parse_case,
    ("algorithm", "question_contract.prompt"): run_question_contract_prompt_case,
    ("algorithm", "question_contract.schema"): run_question_contract_schema_case,
    ("algorithm", "question_contract.schema_validate"): (
        run_question_contract_schema_validate_case
    ),
    ("algorithm", "read_eligibility.cards"): run_read_eligibility_cards_case,
    ("algorithm", "read_eligibility.parse"): run_read_eligibility_parse_case,
    ("algorithm", "read_eligibility.prepare_recall"): (
        run_read_eligibility_prepare_recall_case
    ),
    ("algorithm", "read_eligibility.prompt"): run_read_eligibility_prompt_case,
    ("algorithm", "read_eligibility.schema_validate"): (
        run_read_eligibility_schema_validate_case
    ),
    ("algorithm", "relation_catalog.row_source_projection"): run_relation_catalog_case,
    ("algorithm", "questions.lifecycle"): run_question_run_lifecycle_case,
    ("algorithm", "source_binding.bound_params"): (
        run_source_binding_bound_params_case
    ),
    ("algorithm", "source_binding.finite_choice_parse"): (
        run_source_binding_finite_choice_parse_case
    ),
    ("algorithm", "source_binding.fulfillment_support"): (
        run_source_binding_fulfillment_support_case
    ),
    ("algorithm", "source_binding.metric_fit_parse"): (
        run_source_binding_metric_fit_parse_case
    ),
    ("algorithm", "source_binding.metric_fit_surface"): (
        run_source_binding_metric_fit_surface_case
    ),
    ("algorithm", "source_binding.parse"): run_source_binding_parse_case,
    ("algorithm", "source_binding.prompt_surface"): (
        run_source_binding_prompt_surface_case
    ),
    ("algorithm", "source_binding.row_predicates"): (
        run_source_binding_row_predicates_case
    ),
    ("algorithm", "source_binding.row_predicate_parse"): (
        run_source_binding_row_predicate_parse_case
    ),
    ("algorithm", "source_binding.row_predicate_schema"): (
        run_source_binding_row_predicate_schema_case
    ),
    ("algorithm", "source_binding.schema_surface"): (
        run_source_binding_schema_surface_case
    ),
}


def run_case(case: ConformanceCase) -> list[str]:
    payload = case.payload
    runner = _RUNNERS.get(_case_key(payload))
    if runner is not None:
        return runner(payload)
    return [f"{case.path}: unsupported conformance case"]


def _case_key(payload: dict) -> tuple[str, str]:
    kind = str(payload["kind"])
    if kind == "algorithm":
        return kind, str(payload["algorithm"])
    if kind == "adapter":
        return kind, str(payload["adapter"])
    return kind, ""
