from __future__ import annotations

from django.db import models

from fervis.lineage import records
from fervis.lineage.enums import (
    AnswerValueKind,
    ArtifactKind,
    ClarificationBasis,
    ConversationOriginKind,
    FactResultKind,
    MemoryArtifactSourceKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    PresentationClientKey,
    PresentationKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunStepScopeType,
    RunTriggerKind,
    RuntimeErrorKind,
    SourceReadStatus,
    choices,
)


def default_conversation_read_context_ref() -> dict[str, str | None]:
    return {
        "scheme": "anonymous",
        "key": None,
        "tenant_key": None,
    }


class Conversation(models.Model):
    lineage_record_key = records.CONVERSATION.key

    conversation_id = models.CharField(max_length=128, primary_key=True)
    tenant_id = models.CharField(max_length=128, db_index=True)
    read_context_ref = models.JSONField(default=default_conversation_read_context_ref)
    origin_kind = models.CharField(
        max_length=32,
        choices=choices(ConversationOriginKind),
        default=ConversationOriginKind.INITIAL.value,
    )
    parent_conversation = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="forks",
    )
    forked_after_question = models.ForeignKey(
        "Question",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="conversation_forks",
    )
    forked_after_run = models.ForeignKey(
        "QuestionRun",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="conversation_forks",
    )
    origin_ref = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_conversation"
        indexes = [
            models.Index(
                fields=["tenant_id", "created_at"], name="fervis_conv_tenant_idx"
            ),
            models.Index(fields=["parent_conversation"], name="fervis_conv_parent_idx"),
        ]


class Question(models.Model):
    lineage_record_key = records.QUESTION.key

    question_id = models.CharField(max_length=128, primary_key=True)
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.PROTECT,
        related_name="questions",
    )
    conversation_sequence = models.PositiveIntegerField()
    origin_message_ref = models.CharField(max_length=255, blank=True, default="")
    original_question = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_question"
        constraints = [
            models.UniqueConstraint(
                fields=["conversation", "conversation_sequence"],
                name="fervis_question_conv_seq_uq",
            ),
            models.UniqueConstraint(
                fields=["conversation", "question_id"],
                name="fervis_question_conv_id_uq",
            ),
        ]
        indexes = [
            models.Index(
                fields=["conversation", "conversation_sequence"],
                name="fervis_question_conv_seq_idx",
            ),
        ]


class QuestionRun(models.Model):
    lineage_record_key = records.QUESTION_RUN.key

    run_id = models.CharField(max_length=128, primary_key=True)
    question = models.ForeignKey(
        Question,
        on_delete=models.PROTECT,
        related_name="runs",
    )
    run_number = models.PositiveIntegerField()
    trigger_kind = models.CharField(
        max_length=32,
        choices=choices(RunTriggerKind),
        default=RunTriggerKind.INITIAL.value,
    )
    previous_run = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="triggered_runs",
    )
    trigger_clarification_response_run = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="clarification_triggered_runs",
    )
    trigger_clarification_response_id = models.CharField(
        max_length=128,
        blank=True,
        default="",
    )
    integrated_question = models.TextField()
    adapter_ref = models.CharField(max_length=128)
    runtime_version = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_question_run"
        constraints = [
            models.UniqueConstraint(
                fields=["question", "run_number"],
                name="fervis_run_question_num_uq",
            ),
            models.UniqueConstraint(
                fields=["question", "run_id"],
                name="fervis_run_question_id_unique",
            ),
        ]
        indexes = [
            models.Index(
                fields=["question", "run_number"], name="fervis_run_question_num_idx"
            ),
            models.Index(fields=["previous_run"], name="fervis_run_previous_idx"),
            models.Index(
                fields=["trigger_clarification_response_run"],
                name="fervis_run_clarify_idx",
            ),
        ]


class RunStep(models.Model):
    lineage_record_key = records.RUN_STEP.key

    step_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="steps",
    )
    sequence = models.PositiveIntegerField()
    step_key = models.CharField(max_length=64, choices=choices(RunStepKey))
    attempt = models.PositiveIntegerField(null=True, blank=True)
    scope_type = models.CharField(
        max_length=64,
        choices=choices(RunStepScopeType),
        blank=True,
        default="",
    )
    scope_id = models.CharField(max_length=128, blank=True, default="")
    kind = models.CharField(max_length=32, choices=choices(RunStepKind))
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    input_summary_json = models.JSONField(default=dict, blank=True)
    output_summary_json = models.JSONField(default=dict, blank=True)
    error_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_run_step"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "sequence"],
                name="fervis_step_run_seq_uq",
            ),
            models.UniqueConstraint(
                fields=["run", "step_id"],
                name="fervis_step_run_id_unique",
            ),
        ]
        indexes = [
            models.Index(fields=["run", "step_key"], name="fervis_step_run_key_idx"),
            models.Index(
                fields=["run", "step_key", "scope_type", "scope_id"],
                name="fervis_step_scope_idx",
            ),
        ]


class ModelCall(models.Model):
    lineage_record_key = records.MODEL_CALL.key

    model_call_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="model_calls",
    )
    step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="model_calls",
    )
    call_index = models.PositiveIntegerField()
    provider = models.CharField(max_length=64)
    model_key = models.CharField(max_length=128)
    provider_request_id = models.CharField(max_length=128, blank=True, default="")
    status = models.CharField(max_length=32, choices=choices(ModelCallStatus))
    finish_reason = models.CharField(max_length=64, blank=True, default="")
    duration_ms = models.PositiveIntegerField(null=True, blank=True)
    reasoning_effort = models.CharField(max_length=64, blank=True, default="")
    reasoning_budget_tokens = models.PositiveIntegerField(null=True, blank=True)
    max_output_tokens = models.PositiveIntegerField(null=True, blank=True)
    prompt_chars = models.PositiveIntegerField(default=0)
    schema_chars = models.PositiveIntegerField(default=0)
    tool_spec_chars = models.PositiveIntegerField(default=0)
    submitted_payload_chars = models.PositiveIntegerField(null=True, blank=True)
    raw_output_chars = models.PositiveIntegerField(null=True, blank=True)
    model_subcalls_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_model_call"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "model_call_id"],
                name="fervis_call_run_id_uq",
            ),
            models.UniqueConstraint(
                fields=["run", "step", "model_call_id"],
                name="fervis_call_step_id_uq",
            ),
            models.UniqueConstraint(
                fields=["run", "step", "call_index"],
                name="fervis_call_step_idx_uq",
            ),
        ]
        indexes = [
            models.Index(fields=["run", "step"], name="fervis_model_call_step_idx"),
            models.Index(
                fields=["provider", "model_key"], name="fervis_model_call_provider_idx"
            ),
        ]


class ModelCallUsage(models.Model):
    lineage_record_key = records.MODEL_CALL_USAGE.key

    usage_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="model_call_usages",
    )
    model_call = models.ForeignKey(
        ModelCall,
        on_delete=models.PROTECT,
        related_name="usage_rows",
    )
    usage_kind = models.CharField(max_length=64, choices=choices(ModelUsageKind))
    quantity = models.PositiveBigIntegerField()
    unit = models.CharField(max_length=32, choices=choices(ModelUsageUnit))
    provider_usage_key = models.CharField(max_length=128)
    cost_micros = models.BigIntegerField(null=True, blank=True)
    currency = models.CharField(max_length=16, blank=True, default="")
    price_basis_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_model_call_usage"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "usage_id"],
                name="fervis_usage_run_id_unique",
            )
        ]
        indexes = [
            models.Index(fields=["run", "model_call"], name="fervis_usage_call_idx"),
            models.Index(fields=["usage_kind"], name="fervis_usage_kind_idx"),
        ]


class RunArtifact(models.Model):
    lineage_record_key = records.RUN_ARTIFACT.key

    artifact_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="artifacts",
    )
    step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="artifacts",
    )
    model_call = models.ForeignKey(
        ModelCall,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="artifacts",
    )
    artifact_kind = models.CharField(max_length=64, choices=choices(ArtifactKind))
    content_hash = models.CharField(max_length=128)
    content = models.TextField(null=True, blank=True)
    storage_ref = models.CharField(max_length=512, null=True, blank=True)
    content_type = models.CharField(max_length=128)
    size_bytes = models.PositiveBigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_run_artifact"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "artifact_id"],
                name="fervis_artifact_run_id_unique",
            ),
            models.UniqueConstraint(
                fields=["run", "step", "artifact_id"],
                name="fervis_artifact_step_id_uq",
            ),
            models.CheckConstraint(
                name="fervis_artifact_one_body",
                condition=(
                    models.Q(content__isnull=False, storage_ref__isnull=True)
                    | (
                        models.Q(content__isnull=True, storage_ref__isnull=False)
                        & ~models.Q(storage_ref="")
                    )
                ),
            ),
        ]
        indexes = [
            models.Index(fields=["run", "step"], name="fervis_artifact_step_idx"),
            models.Index(
                fields=["model_call", "artifact_kind"],
                name="fervis_artifact_call_kind_idx",
            ),
        ]


class CatalogEndpoint(models.Model):
    lineage_record_key = records.CATALOG_ENDPOINT.key

    catalog_endpoint_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="catalog_endpoints",
    )
    catalog_endpoint_key = models.CharField(max_length=128)
    endpoint_name = models.CharField(max_length=128, db_index=True)
    framework_kind = models.CharField(max_length=64)
    source_namespace_kind = models.CharField(max_length=64)
    source_namespace_path_json = models.JSONField(default=list, blank=True)
    route_method = models.CharField(max_length=16)
    route_path_template = models.CharField(max_length=512)
    route_name = models.CharField(max_length=128, blank=True, default="")
    api_schema_operation_id = models.CharField(max_length=128, blank=True, default="")
    handler_ref = models.CharField(max_length=512)
    domain_resource_names_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_catalog_endpoint"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "catalog_endpoint_id"],
                name="fervis_cat_endpoint_run_id_uq",
            ),
            models.UniqueConstraint(
                fields=["run", "catalog_endpoint_key"],
                name="fervis_cat_endpoint_run_key_uq",
            ),
        ]
        indexes = [
            models.Index(fields=["run"], name="fervis_cat_endpoint_run_idx"),
            models.Index(
                fields=["framework_kind", "source_namespace_kind"],
                name="fervis_cat_endpoint_fw_idx",
            ),
            models.Index(
                fields=["endpoint_name"],
                name="fervis_cat_endpoint_name_idx",
            ),
        ]


class SourceRead(models.Model):
    lineage_record_key = records.SOURCE_READ.key

    source_read_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="source_reads",
    )
    step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="source_reads",
    )
    catalog_endpoint = models.ForeignKey(
        CatalogEndpoint,
        on_delete=models.PROTECT,
        related_name="source_reads",
    )
    args_json = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=32, choices=choices(SourceReadStatus))
    row_count = models.PositiveIntegerField(null=True, blank=True)
    completeness_json = models.JSONField(default=dict, blank=True)
    response_hash = models.CharField(max_length=128, blank=True, default="")
    artifact = models.ForeignKey(
        RunArtifact,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="source_reads",
    )
    error_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_source_read"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "source_read_id"],
                name="fervis_src_read_run_id_uq",
            ),
        ]
        indexes = [
            models.Index(fields=["run", "step"], name="fervis_source_read_step_idx"),
            models.Index(
                fields=["catalog_endpoint"],
                name="fervis_src_read_catalog_idx",
            ),
        ]


class RunResult(models.Model):
    lineage_record_key = records.RUN_RESULT.key

    run_result_id = models.CharField(max_length=128, primary_key=True)
    run = models.OneToOneField(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="run_result",
    )
    result_kind = models.CharField(max_length=32, choices=choices(RunResultKind))
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_run_result"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "run_result_id"],
                name="fervis_run_result_run_id_uq",
            )
        ]


class RuntimeErrorDetail(models.Model):
    lineage_record_key = records.RUNTIME_ERROR.key

    runtime_error_detail_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="runtime_errors",
    )
    run_result = models.OneToOneField(
        RunResult,
        on_delete=models.PROTECT,
        related_name="runtime_error_detail",
    )
    failed_step = models.ForeignKey(
        RunStep,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="runtime_errors",
    )
    error_kind = models.CharField(max_length=64, choices=choices(RuntimeErrorKind))
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_runtime_error_detail"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "runtime_error_detail_id"],
                name="fervis_runtime_err_run_id_uq",
            )
        ]
        indexes = [
            models.Index(
                fields=["run", "failed_step"], name="fervis_runtime_error_step_idx"
            ),
        ]


class RequestedFact(models.Model):
    lineage_record_key = records.REQUESTED_FACT.key

    requested_fact_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="requested_facts",
    )
    produced_by_step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="requested_facts",
    )
    fact_key = models.CharField(max_length=128)
    description = models.TextField(blank=True, default="")
    answer_expression_family = models.CharField(max_length=64)
    requested_fact_json = models.JSONField(default=dict, blank=True)
    answer_requests_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_requested_fact"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "requested_fact_id"],
                name="fervis_req_fact_run_id_uq",
            ),
            models.UniqueConstraint(
                fields=["run", "produced_by_step", "fact_key"],
                name="fervis_req_fact_step_key_uq",
            ),
        ]
        indexes = [
            models.Index(fields=["run", "fact_key"], name="fervis_req_fact_key_idx"),
        ]


class FactResult(models.Model):
    lineage_record_key = records.FACT_RESULT.key

    fact_result_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="fact_results",
    )
    requested_fact = models.ForeignKey(
        RequestedFact,
        on_delete=models.PROTECT,
        related_name="fact_results",
    )
    produced_by_step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="fact_results",
    )
    result_kind = models.CharField(max_length=64, choices=choices(FactResultKind))
    evidence_refs_json = models.JSONField(default=list, blank=True)
    payload_schema = models.CharField(max_length=128, blank=True, default="")
    payload_schema_rev = models.PositiveIntegerField(null=True, blank=True)
    payload_json = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_fact_result"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "fact_result_id"],
                name="fervis_fact_result_run_id_uq",
            ),
        ]
        indexes = [
            models.Index(
                fields=["run", "requested_fact"], name="fervis_fact_result_fact_idx"
            ),
            models.Index(
                fields=["run", "result_kind"], name="fervis_fact_result_kind_idx"
            ),
        ]


class MemoryArtifact(models.Model):
    lineage_record_key = records.MEMORY_ARTIFACT.key

    memory_artifact_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="memory_artifacts",
    )
    produced_by_step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="memory_artifacts",
    )
    source_kind = models.CharField(
        max_length=64,
        choices=choices(MemoryArtifactSourceKind),
    )
    requested_fact = models.ForeignKey(
        RequestedFact,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="memory_artifacts",
    )
    fact_result = models.ForeignKey(
        FactResult,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="memory_artifacts",
    )
    payload_schema = models.CharField(max_length=128)
    payload_schema_rev = models.PositiveIntegerField()
    payload_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_memory_artifact"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "memory_artifact_id"],
                name="fervis_memory_art_run_id_uq",
            ),
        ]
        indexes = [
            models.Index(
                fields=["run", "source_kind"], name="fervis_memory_art_source_idx"
            ),
            models.Index(
                fields=["run", "requested_fact"], name="fervis_memory_art_req_idx"
            ),
            models.Index(
                fields=["run", "fact_result"], name="fervis_memory_art_result_idx"
            ),
        ]


class ClarificationRequest(models.Model):
    lineage_record_key = records.CLARIFICATION_REQUEST.key

    clarification_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="clarification_requests",
    )
    fact_result = models.ForeignKey(
        FactResult,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="clarification_requests",
    )
    step = models.ForeignKey(
        RunStep,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="clarification_requests",
    )
    basis = models.CharField(max_length=64, choices=choices(ClarificationBasis))
    question_text = models.TextField()
    options_json = models.JSONField(default=list, blank=True)
    evidence_refs_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_clarification_request"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "clarification_id"],
                name="fervis_clar_req_run_id_uq",
            ),
        ]
        indexes = [
            models.Index(fields=["run", "basis"], name="fervis_clar_req_basis_idx"),
        ]


class ClarificationResponse(models.Model):
    lineage_record_key = records.CLARIFICATION_RESPONSE.key

    response_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="clarification_responses",
    )
    clarification = models.ForeignKey(
        ClarificationRequest,
        on_delete=models.PROTECT,
        related_name="responses",
    )
    source_message_ref = models.CharField(max_length=255, blank=True, default="")
    selected_option_id = models.CharField(max_length=128, blank=True, default="")
    response_text = models.TextField(blank=True, default="")
    evidence_ref = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_clarification_response"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "response_id"],
                name="fervis_clar_resp_run_id_uq",
            )
        ]
        indexes = [
            models.Index(
                fields=["run", "clarification"], name="fervis_clar_resp_req_idx"
            ),
        ]


class Answer(models.Model):
    lineage_record_key = records.ANSWER.key

    answer_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="answers",
    )
    run_result = models.OneToOneField(
        RunResult,
        on_delete=models.PROTECT,
        related_name="answer",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_answer"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "answer_id"],
                name="fervis_answer_run_id_uq",
            )
        ]


class AnswerOutput(models.Model):
    lineage_record_key = records.ANSWER_OUTPUT.key

    answer_output_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="answer_outputs",
    )
    answer = models.ForeignKey(
        Answer,
        on_delete=models.PROTECT,
        related_name="outputs",
    )
    fact_result = models.ForeignKey(
        FactResult,
        on_delete=models.PROTECT,
        related_name="answer_outputs",
    )
    output_key = models.CharField(max_length=128)
    value_kind = models.CharField(max_length=64, choices=choices(AnswerValueKind))
    value_json = models.JSONField(default=dict)
    proof_node_refs_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_answer_output"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "answer_output_id"],
                name="fervis_answer_output_run_id_uq",
            ),
            models.UniqueConstraint(
                fields=["run", "answer", "output_key"],
                name="fervis_answer_output_key_uq",
            ),
        ]
        indexes = [
            models.Index(fields=["run", "answer"], name="fervis_answer_output_ans_idx"),
            models.Index(
                fields=["run", "fact_result"], name="fervis_answer_output_fact_idx"
            ),
        ]


class AnswerPresentation(models.Model):
    lineage_record_key = records.ANSWER_PRESENTATION.key

    presentation_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="answer_presentations",
    )
    answer = models.ForeignKey(
        Answer,
        on_delete=models.PROTECT,
        related_name="presentations",
    )
    client_key = models.CharField(
        max_length=64,
        choices=choices(PresentationClientKey),
        default=PresentationClientKey.DEFAULT.value,
    )
    locale = models.CharField(max_length=32, default="default")
    presentation_kind = models.CharField(
        max_length=64,
        choices=choices(PresentationKind),
    )
    rendered_value = models.TextField(null=True, blank=True)
    render_step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="answer_presentations",
    )
    payload_schema = models.CharField(max_length=128, blank=True, default="")
    payload_schema_rev = models.PositiveIntegerField(null=True, blank=True)
    payload_json = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_answer_presentation"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "presentation_id"],
                name="fervis_answer_pres_run_id_uq",
            ),
            models.UniqueConstraint(
                fields=["run", "answer", "client_key", "locale", "presentation_kind"],
                name="fervis_answer_pres_shape_uq",
            ),
            models.CheckConstraint(
                name="fervis_answer_pres_body_ck",
                condition=(
                    models.Q(rendered_value__isnull=False, payload_json__isnull=True)
                    | models.Q(rendered_value__isnull=True, payload_json__isnull=False)
                ),
            ),
        ]
        indexes = [
            models.Index(fields=["run", "answer"], name="fervis_answer_pres_ans_idx"),
        ]


class ExecutionProofGraph(models.Model):
    lineage_record_key = records.EXECUTION_PROOF_GRAPH.key

    proof_graph_id = models.CharField(max_length=128, primary_key=True)
    run = models.ForeignKey(
        QuestionRun,
        on_delete=models.PROTECT,
        related_name="execution_proof_graphs",
    )
    fact_result = models.OneToOneField(
        FactResult,
        on_delete=models.PROTECT,
        related_name="execution_proof_graph",
    )
    compile_step = models.ForeignKey(
        RunStep,
        on_delete=models.PROTECT,
        related_name="compiled_proof_graphs",
    )
    execute_step = models.ForeignKey(
        RunStep,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="executed_proof_graphs",
    )
    payload_schema = models.CharField(max_length=128)
    payload_schema_rev = models.PositiveIntegerField()
    payload_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fervis_execution_proof_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "proof_graph_id"],
                name="fervis_proof_graph_run_id_uq",
            )
        ]
        indexes = [
            models.Index(fields=["run", "fact_result"], name="fervis_proof_fact_idx"),
        ]
