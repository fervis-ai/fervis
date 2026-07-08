from __future__ import annotations

from enum import StrEnum


def choices(enum_type: type[StrEnum]) -> tuple[tuple[str, str], ...]:
    return tuple((item.value, item.value) for item in enum_type)


class ConversationOriginKind(StrEnum):
    INITIAL = "initial"
    FORK = "fork"


class RunTriggerKind(StrEnum):
    INITIAL = "initial"
    CLARIFICATION_RESPONSE = "clarification_response"
    RETRY = "retry"
    RERUN = "rerun"
    REPLAY = "replay"


class RunResultKind(StrEnum):
    ANSWERED = "answered"
    FACTUAL_TERMINAL = "factual_terminal"
    RUNTIME_ERROR = "runtime_error"


class FactResultKind(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    IMPOSSIBLE = "impossible"
    NO_DATA = "no_data"
    UNDEFINED = "undefined"


class AnswerValueKind(StrEnum):
    ENTITY = "entity"
    NUMBER = "number"
    MONEY = "money"
    BOOLEAN = "boolean"
    TEXT = "text"
    DATE = "date"
    DATETIME = "datetime"
    TABLE = "table"
    LIST = "list"
    OBJECT = "object"


class PresentationKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    TEMPLATE = "template"
    FORMATTED_NUMBER = "formatted_number"


class PresentationClientKey(StrEnum):
    DEFAULT = "default"
    API = "api"
    CLI = "cli"
    WEB = "web"
    SLACK = "slack"


class ContributionOrigin(StrEnum):
    EXPLICIT = "explicit"
    DERIVED = "derived"
    CONTEXTUAL = "contextual"


class MemoryArtifactSourceKind(StrEnum):
    REQUESTED_FACT = "requested_fact"
    KNOWN_INPUT = "known_input"
    FACT_RESULT = "fact_result"
    RUN_TERMINAL = "run_terminal"


class ProofNodeKind(StrEnum):
    ANSWER_OUTPUT = "answer_output"
    ENDPOINT_ARG = "endpoint_arg"
    OPERATION = "operation"
    OPERATION_INPUT = "operation_input"
    POPULATION_CHOICE = "population_choice"
    RELATION = "relation"
    ROW_FILTER = "row_filter"
    SCALAR = "scalar"


class ProofEdgeRole(StrEnum):
    INPUT = "input"
    NARROWS = "narrows"
    PRODUCES = "produces"
    RANK_LIMIT = "rank_limit"
    SCOPES = "scopes"


class RuntimeErrorKind(StrEnum):
    PLANNING_FAILED = "planning_failed"
    FACT_PLAN_PARSE_FAILED = "fact_plan_parse_failed"
    PLAN_VALIDATION_FAILED = "plan_validation_failed"
    FACT_PLAN_EXECUTION_FAILED = "fact_plan_execution_failed"
    FACT_PLAN_VERIFICATION_FAILED = "fact_plan_verification_failed"
    COMPILER_INVARIANT_FAILED = "compiler_invariant_failed"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    PAGE_CAP_TRUNCATION = "page_cap_truncation"
    PROVIDER_RUNTIME_FAILED = "provider_runtime_failed"
    POLICY_LIMIT_EXCEEDED = "policy_limit_exceeded"
    FRAMEWORK_ADAPTER_FAILED = "framework_adapter_failed"
    LINEAGE_PERSISTENCE_FAILED = "lineage_persistence_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


class RunStepKind(StrEnum):
    MODEL_TURN = "model_turn"
    DETERMINISTIC = "deterministic"


class RunStepKey(StrEnum):
    CONVERSATION_RESOLUTION = "conversation_resolution"
    QUESTION_CONTRACT = "question_contract"
    QUERY_ENRICHMENT = "query_enrichment"
    CATALOG_SELECTION = "catalog_selection"
    GROUNDING = "grounding"
    READ_ELIGIBILITY = "read_eligibility"
    PLAN_SELECTION = "plan_selection"
    SOURCE_BINDING = "source_binding"
    FACT_PLANNING = "fact_planning"
    VERIFY = "verify"
    COMPILE = "compile"
    EXECUTE = "execute"
    CLASSIFY = "classify"
    RENDER = "render"
    ANSWER_SYNTHESIS = "answer_synthesis"


class RunStepScopeType(StrEnum):
    REQUESTED_FACT = "requested_fact"
    SOURCE = "source"
    PAGE = "page"


class ModelCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelUsageKind(StrEnum):
    INPUT_TOKENS = "input_tokens"
    CACHED_INPUT_TOKENS = "cached_input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    REASONING_TOKENS = "reasoning_tokens"
    THINKING_TOKENS = "thinking_tokens"
    TOOL_TOKENS = "tool_tokens"


class ModelUsageUnit(StrEnum):
    TOKENS = "tokens"
    CHARS = "chars"
    CALLS = "calls"


class SourceReadStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ArtifactKind(StrEnum):
    SYSTEM_PROMPT = "system_prompt"
    PROMPT = "prompt"
    SCHEMA = "schema"
    TOOL_SPEC = "tool_spec"
    SUBMITTED_PAYLOAD = "submitted_payload"
    RAW_OUTPUT = "raw_output"
    PARSED_PAYLOAD = "parsed_payload"
    DETERMINISTIC_INPUT = "deterministic_input"
    DETERMINISTIC_OUTPUT = "deterministic_output"
    SOURCE_RESPONSE = "source_response"
    ROW_CONTEXT = "row_context"
    COMPILED_EXECUTION = "compiled_execution"


ARTIFACT_KINDS_REQUIRING_MODEL_CALL = (
    ArtifactKind.SYSTEM_PROMPT.value,
    ArtifactKind.PROMPT.value,
    ArtifactKind.SCHEMA.value,
    ArtifactKind.TOOL_SPEC.value,
    ArtifactKind.SUBMITTED_PAYLOAD.value,
    ArtifactKind.RAW_OUTPUT.value,
    ArtifactKind.PARSED_PAYLOAD.value,
)
