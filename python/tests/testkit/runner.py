from __future__ import annotations

from collections.abc import Callable

from tests.testkit.adapters.host_api import run_host_api_projection_case
from tests.testkit.algorithms.answer_program import (
    run_answer_program_apply_capability_case,
    run_answer_program_canonicalize_case,
    run_answer_program_decode_case,
    run_answer_program_instantiate_case,
    run_answer_program_invoke_case,
    run_answer_program_order_take_case,
    run_answer_program_patch_case,
    run_answer_program_projected_operation_inputs_case,
    run_answer_program_relation_case,
)
from tests.testkit.algorithms.business_time import run_business_time_case
from tests.testkit.algorithms.capabilities import run_capabilities_case
from tests.testkit.algorithms.catalog_selection import (
    run_catalog_selection_case,
    run_resolver_catalog_selection_case,
)
from tests.testkit.algorithms.conversation_resolution import (
    run_conversation_resolution_compile_case,
    run_conversation_resolution_parse_case,
    run_conversation_resolution_schema_case,
)
from tests.testkit.algorithms.endpoint_response import run_endpoint_response_case
from tests.testkit.algorithms.identity_set_binding import run_identity_set_binding_case
from tests.testkit.algorithms.lineage import (
    run_lineage_explain_case,
    run_lineage_input_lineage_case,
)
from tests.testkit.algorithms.memory import (
    run_conversation_memory_card_projection_case,
    run_conversation_memory_expand_activated_case,
    run_memory_answer_addresses_case,
    run_memory_available_values_case,
    run_memory_build_artifact_case,
    run_memory_identity_projection_case,
    run_memory_lineage_memory_artifacts_case,
    run_memory_lookup_projection_case,
    run_memory_outcome_address_case,
    run_memory_project_conversation_case,
    run_memory_prior_answer_request_case,
)
from tests.testkit.algorithms.outcomes import run_outcomes_classify_case
from tests.testkit.algorithms.parameter_bindings import (
    run_parameter_binding_alternatives_case,
)
from tests.testkit.algorithms.question_run_lifecycle import (
    run_question_run_lifecycle_case,
)
from tests.testkit.algorithms.questions_projection import (
    run_questions_memory_projection_case,
    run_questions_projection_case,
)
from tests.testkit.algorithms.relation_engine import (
    run_api_read_completeness_case,
    run_calendar_relation_case,
    run_relation_engine_case,
)
from tests.testkit.algorithms.relation_catalog import run_relation_catalog_case
from tests.testkit.algorithms.semantic_kernel import run_semantic_kernel_case
from tests.testkit.algorithms.semantic_query_enrichment import (
    run_semantic_query_enrichment_case,
)
from tests.testkit.algorithms.semantic_question_contract import (
    run_semantic_question_contract_case,
)
from tests.testkit.algorithms.semantic_source_binding import (
    run_semantic_source_binding_case,
)
from tests.testkit.case_loader import ConformanceCase

CaseRunner = Callable[[dict], list[str]]

_RUNNERS: dict[tuple[str, str], CaseRunner] = {
    ("adapter", "host_api.endpoint_contract_projection"): run_host_api_projection_case,
    (
        "algorithm",
        "answer_program.apply_capability",
    ): run_answer_program_apply_capability_case,
    ("algorithm", "answer_program.canonicalize"): run_answer_program_canonicalize_case,
    ("algorithm", "answer_program.decode"): run_answer_program_decode_case,
    ("algorithm", "answer_program.instantiate"): run_answer_program_instantiate_case,
    ("algorithm", "answer_program.invoke"): run_answer_program_invoke_case,
    ("algorithm", "answer_program.order_take"): run_answer_program_order_take_case,
    ("algorithm", "answer_program.patch"): run_answer_program_patch_case,
    (
        "algorithm",
        "answer_program.projected_operation_inputs",
    ): run_answer_program_projected_operation_inputs_case,
    ("algorithm", "answer_program.relation"): run_answer_program_relation_case,
    ("algorithm", "business_time.resolve"): run_business_time_case,
    (
        "algorithm",
        "catalog_selection.resolver_select",
    ): run_resolver_catalog_selection_case,
    ("algorithm", "catalog_selection.select"): run_catalog_selection_case,
    (
        "algorithm",
        "conversation_memory.card_projection",
    ): run_conversation_memory_card_projection_case,
    (
        "algorithm",
        "conversation_memory.expand_activated",
    ): run_conversation_memory_expand_activated_case,
    (
        "algorithm",
        "conversation_resolution.compile",
    ): run_conversation_resolution_compile_case,
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
    ("algorithm", "execution.api_read_completeness"): run_api_read_completeness_case,
    ("algorithm", "execution.endpoint_response"): run_endpoint_response_case,
    ("algorithm", "execution.identity_set_binding"): run_identity_set_binding_case,
    ("algorithm", "execution.relation_engine"): run_relation_engine_case,
    ("algorithm", "relation_catalog.row_source_projection"): run_relation_catalog_case,
    ("algorithm", "lineage.explain"): run_lineage_explain_case,
    ("algorithm", "lineage.input_lineage"): run_lineage_input_lineage_case,
    ("algorithm", "memory.answer_addresses"): run_memory_answer_addresses_case,
    ("algorithm", "memory.available_values"): run_memory_available_values_case,
    ("algorithm", "memory.build_artifact"): run_memory_build_artifact_case,
    ("algorithm", "memory.identity_projection"): run_memory_identity_projection_case,
    (
        "algorithm",
        "memory.lineage_memory_artifacts",
    ): run_memory_lineage_memory_artifacts_case,
    ("algorithm", "memory.lookup_projection"): run_memory_lookup_projection_case,
    ("algorithm", "memory.outcome_address"): run_memory_outcome_address_case,
    ("algorithm", "memory.project_conversation"): run_memory_project_conversation_case,
    ("algorithm", "memory.prior_answer_request"): run_memory_prior_answer_request_case,
    ("algorithm", "outcomes.classify"): run_outcomes_classify_case,
    ("algorithm", "questions.lifecycle"): run_question_run_lifecycle_case,
    ("algorithm", "questions.memory_projection"): run_questions_memory_projection_case,
    ("algorithm", "questions.projection"): run_questions_projection_case,
    ("algorithm", "semantic.kernel"): run_semantic_kernel_case,
    ("algorithm", "semantic.query_enrichment"): run_semantic_query_enrichment_case,
    ("algorithm", "semantic.question_contract"): run_semantic_question_contract_case,
    ("algorithm", "semantic.source_binding"): run_semantic_source_binding_case,
    (
        "algorithm",
        "source_binding.parameter_alternatives",
    ): run_parameter_binding_alternatives_case,
}


def run_case(case: ConformanceCase) -> list[str]:
    runner = _RUNNERS.get(_case_key(case.payload))
    if runner is None:
        return [f"{case.path}: unsupported conformance case"]
    return runner(case.payload)


def _case_key(payload: dict) -> tuple[str, str]:
    kind = str(payload["kind"])
    if kind == "algorithm":
        return kind, str(payload["algorithm"])
    if kind == "adapter":
        return kind, str(payload["adapter"])
    return kind, ""
