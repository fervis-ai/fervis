"""Frozen Fervis persistence schema snapshot for revision 0001."""

from __future__ import annotations

import sqlalchemy as sa

metadata = sa.MetaData()

fervis_conversation = sa.Table(
    "fervis_conversation",
    metadata,
    sa.Column("conversation_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column("tenant_id", sa.String(128), nullable=False),
    sa.Column("read_context_ref", sa.JSON(), nullable=False),
    sa.Column("origin_kind", sa.String(32), nullable=False),
    sa.Column(
        "parent_conversation_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_conversation.conversation_id",
            name="fervis_conversation_parent_conversation_id_fk",
        ),
        nullable=True,
    ),
    sa.Column(
        "forked_after_question_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question.question_id",
            name="fervis_conversation_forked_after_question_id_fk",
        ),
        nullable=True,
    ),
    sa.Column(
        "forked_after_run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id",
            name="fervis_conversation_forked_after_run_id_fk",
        ),
        nullable=True,
    ),
    sa.Column("origin_ref", sa.String(255), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)
sa.Index(
    "fervis_conv_tenant_idx",
    fervis_conversation.c.tenant_id,
    fervis_conversation.c.created_at,
)
sa.Index("fervis_conv_parent_idx", fervis_conversation.c.parent_conversation_id)
sa.Index("fervis_conversation_tenant_id_idx", fervis_conversation.c.tenant_id)
sa.Index(
    "fervis_conversation_parent_conversation_id_idx",
    fervis_conversation.c.parent_conversation_id,
)
sa.Index(
    "fervis_conversation_forked_after_question_id_idx",
    fervis_conversation.c.forked_after_question_id,
)
sa.Index(
    "fervis_conversation_forked_after_run_id_idx",
    fervis_conversation.c.forked_after_run_id,
)

fervis_question = sa.Table(
    "fervis_question",
    metadata,
    sa.Column("question_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "conversation_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_conversation.conversation_id",
            name="fervis_question_conversation_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("conversation_sequence", sa.Integer(), nullable=False),
    sa.Column("origin_message_ref", sa.String(255), nullable=False),
    sa.Column("original_question", sa.Text(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "conversation_id", "conversation_sequence", name="fervis_question_conv_seq_uq"
    ),
    sa.UniqueConstraint(
        "conversation_id", "question_id", name="fervis_question_conv_id_uq"
    ),
    sa.CheckConstraint(
        "conversation_sequence" + " >= 0",
        name="fervis_question_conversation_sequence_nonnegative_ck",
    ),
)
sa.Index(
    "fervis_question_conv_seq_idx",
    fervis_question.c.conversation_id,
    fervis_question.c.conversation_sequence,
)
sa.Index("fervis_question_conversation_id_idx", fervis_question.c.conversation_id)

fervis_question_run = sa.Table(
    "fervis_question_run",
    metadata,
    sa.Column("run_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "question_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question.question_id", name="fervis_question_run_question_id_fk"
        ),
        nullable=False,
    ),
    sa.Column("run_number", sa.Integer(), nullable=False),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("trigger_kind", sa.String(32), nullable=False),
    sa.Column(
        "base_run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_question_run_base_run_id_fk"
        ),
        nullable=True,
    ),
    sa.Column("trigger_clarification_response_id", sa.String(128), nullable=False),
    sa.Column("adapter_ref", sa.String(128), nullable=False),
    sa.Column("runtime_version", sa.String(128), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("question_id", "run_number", name="fervis_run_question_num_uq"),
    sa.UniqueConstraint("question_id", "run_id", name="fervis_run_question_id_unique"),
    sa.CheckConstraint(
        "run_number" + " >= 0", name="fervis_question_run_run_number_nonnegative_ck"
    ),
)
sa.Index(
    "fervis_run_question_num_idx",
    fervis_question_run.c.question_id,
    fervis_question_run.c.run_number,
)
sa.Index("fervis_run_base_idx", fervis_question_run.c.base_run_id)
sa.Index("fervis_question_run_question_id_idx", fervis_question_run.c.question_id)
sa.Index(
    "fervis_question_run_base_run_id_idx", fervis_question_run.c.base_run_id
)

fervis_answer_program = sa.Table(
    "fervis_answer_program",
    metadata,
    sa.Column("program_id", sa.String(80), primary_key=True, nullable=False),
    sa.Column("schema_revision", sa.Integer(), nullable=False),
    sa.Column("canonical_json", sa.Text(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

fervis_program_revision = sa.Table(
    "fervis_program_revision",
    metadata,
    sa.Column("revision_id", sa.String(80), primary_key=True, nullable=False),
    sa.Column(
        "base_program_id",
        sa.String(80),
        sa.ForeignKey(
            "fervis_answer_program.program_id",
            name="fervis_program_revision_base_program_id_fk",
        ),
        nullable=False,
    ),
    sa.Column(
        "revised_program_id",
        sa.String(80),
        sa.ForeignKey(
            "fervis_answer_program.program_id",
            name="fervis_program_revision_revised_program_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("capability_id", sa.String(128), nullable=False),
    sa.Column("application_json", sa.Text(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)
sa.Index(
    "fervis_program_rev_base_idx",
    fervis_program_revision.c.base_program_id,
)
sa.Index(
    "fervis_program_rev_revised_idx",
    fervis_program_revision.c.revised_program_id,
)

fervis_program_invocation = sa.Table(
    "fervis_program_invocation",
    metadata,
    sa.Column("invocation_id", sa.String(80), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id",
            name="fervis_program_invocation_run_id_fk",
        ),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "program_id",
        sa.String(80),
        sa.ForeignKey(
            "fervis_answer_program.program_id",
            name="fervis_program_invocation_program_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("bindings_json", sa.Text(), nullable=False),
    sa.Column("patch_id", sa.String(80), nullable=True),
    sa.Column("binding_patch_json", sa.Text(), nullable=True),
    sa.Column(
        "revision_id",
        sa.String(80),
        sa.ForeignKey(
            "fervis_program_revision.revision_id",
            name="fervis_program_invocation_revision_id_fk",
        ),
        nullable=True,
    ),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        "(patch_id IS NULL AND binding_patch_json IS NULL) OR "
        "(patch_id IS NOT NULL AND binding_patch_json IS NOT NULL)",
        name="fervis_program_invocation_patch_pair_ck",
    ),
)
sa.Index(
    "fervis_program_invocation_program_id_idx",
    fervis_program_invocation.c.program_id,
)

fervis_run_step = sa.Table(
    "fervis_run_step",
    metadata,
    sa.Column("step_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey("fervis_question_run.run_id", name="fervis_run_step_run_id_fk"),
        nullable=False,
    ),
    sa.Column("sequence", sa.Integer(), nullable=False),
    sa.Column("step_key", sa.String(64), nullable=False),
    sa.Column("attempt", sa.Integer(), nullable=True),
    sa.Column("scope_type", sa.String(64), nullable=False),
    sa.Column("scope_id", sa.String(128), nullable=False),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("input_summary_json", sa.JSON(), nullable=False),
    sa.Column("output_summary_json", sa.JSON(), nullable=False),
    sa.Column("error_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "sequence", name="fervis_step_run_seq_uq"),
    sa.UniqueConstraint("run_id", "step_id", name="fervis_step_run_id_unique"),
    sa.CheckConstraint(
        "sequence" + " >= 0", name="fervis_run_step_sequence_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "attempt" + " >= 0", name="fervis_run_step_attempt_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "JSON_VALID(input_summary_json)",
        name="fervis_run_step_input_summary_json_json_valid_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(output_summary_json)",
        name="fervis_run_step_output_summary_json_json_valid_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(error_json)", name="fervis_run_step_error_json_json_valid_ck"
    ),
)
sa.Index(
    "fervis_step_run_key_idx", fervis_run_step.c.run_id, fervis_run_step.c.step_key
)
sa.Index(
    "fervis_step_scope_idx",
    fervis_run_step.c.run_id,
    fervis_run_step.c.step_key,
    fervis_run_step.c.scope_type,
    fervis_run_step.c.scope_id,
)
sa.Index("fervis_run_step_run_id_idx", fervis_run_step.c.run_id)

fervis_model_call = sa.Table(
    "fervis_model_call",
    metadata,
    sa.Column("model_call_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey("fervis_question_run.run_id", name="fervis_model_call_run_id_fk"),
        nullable=False,
    ),
    sa.Column(
        "step_id",
        sa.String(128),
        sa.ForeignKey("fervis_run_step.step_id", name="fervis_model_call_step_id_fk"),
        nullable=False,
    ),
    sa.Column("call_index", sa.Integer(), nullable=False),
    sa.Column("provider", sa.String(64), nullable=False),
    sa.Column("model_key", sa.String(128), nullable=False),
    sa.Column("provider_request_id", sa.String(128), nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("finish_reason", sa.String(64), nullable=False),
    sa.Column("duration_ms", sa.Integer(), nullable=True),
    sa.Column("reasoning_effort", sa.String(64), nullable=False),
    sa.Column("reasoning_budget_tokens", sa.Integer(), nullable=True),
    sa.Column("max_output_tokens", sa.Integer(), nullable=True),
    sa.Column("prompt_chars", sa.Integer(), nullable=False),
    sa.Column("schema_chars", sa.Integer(), nullable=False),
    sa.Column("tool_spec_chars", sa.Integer(), nullable=False),
    sa.Column("submitted_payload_chars", sa.Integer(), nullable=True),
    sa.Column("raw_output_chars", sa.Integer(), nullable=True),
    sa.Column("model_subcalls_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "model_call_id", name="fervis_call_run_id_uq"),
    sa.UniqueConstraint(
        "run_id", "step_id", "model_call_id", name="fervis_call_step_id_uq"
    ),
    sa.UniqueConstraint(
        "run_id", "step_id", "call_index", name="fervis_call_step_idx_uq"
    ),
    sa.CheckConstraint(
        "call_index" + " >= 0", name="fervis_model_call_call_index_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "duration_ms" + " >= 0", name="fervis_model_call_duration_ms_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "reasoning_budget_tokens" + " >= 0",
        name="fervis_model_call_reasoning_budget_tokens_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "max_output_tokens" + " >= 0",
        name="fervis_model_call_max_output_tokens_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "prompt_chars" + " >= 0", name="fervis_model_call_prompt_chars_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "schema_chars" + " >= 0", name="fervis_model_call_schema_chars_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "tool_spec_chars" + " >= 0",
        name="fervis_model_call_tool_spec_chars_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "submitted_payload_chars" + " >= 0",
        name="fervis_model_call_submitted_payload_chars_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "raw_output_chars" + " >= 0",
        name="fervis_model_call_raw_output_chars_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(model_subcalls_json)",
        name="fervis_model_call_model_subcalls_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_model_call_step_idx",
    fervis_model_call.c.run_id,
    fervis_model_call.c.step_id,
)
sa.Index(
    "fervis_model_call_provider_idx",
    fervis_model_call.c.provider,
    fervis_model_call.c.model_key,
)
sa.Index("fervis_model_call_run_id_idx", fervis_model_call.c.run_id)
sa.Index("fervis_model_call_step_id_idx", fervis_model_call.c.step_id)

fervis_model_call_usage = sa.Table(
    "fervis_model_call_usage",
    metadata,
    sa.Column("usage_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_model_call_usage_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "model_call_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_model_call.model_call_id",
            name="fervis_model_call_usage_model_call_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("usage_kind", sa.String(64), nullable=False),
    sa.Column("quantity", sa.BigInteger(), nullable=False),
    sa.Column("unit", sa.String(32), nullable=False),
    sa.Column("provider_usage_key", sa.String(128), nullable=False),
    sa.Column("cost_micros", sa.BigInteger(), nullable=True),
    sa.Column("currency", sa.String(16), nullable=False),
    sa.Column("price_basis_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "usage_id", name="fervis_usage_run_id_unique"),
    sa.CheckConstraint(
        "quantity" + " >= 0", name="fervis_model_call_usage_quantity_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "JSON_VALID(price_basis_json)",
        name="fervis_model_call_usage_price_basis_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_usage_call_idx",
    fervis_model_call_usage.c.run_id,
    fervis_model_call_usage.c.model_call_id,
)
sa.Index("fervis_usage_kind_idx", fervis_model_call_usage.c.usage_kind)
sa.Index("fervis_model_call_usage_run_id_idx", fervis_model_call_usage.c.run_id)
sa.Index(
    "fervis_model_call_usage_model_call_id_idx", fervis_model_call_usage.c.model_call_id
)

fervis_run_artifact = sa.Table(
    "fervis_run_artifact",
    metadata,
    sa.Column("artifact_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_run_artifact_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "step_id",
        sa.String(128),
        sa.ForeignKey("fervis_run_step.step_id", name="fervis_run_artifact_step_id_fk"),
        nullable=False,
    ),
    sa.Column(
        "model_call_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_model_call.model_call_id",
            name="fervis_run_artifact_model_call_id_fk",
        ),
        nullable=True,
    ),
    sa.Column("artifact_kind", sa.String(64), nullable=False),
    sa.Column("content_hash", sa.String(128), nullable=False),
    sa.Column("content", sa.Text(), nullable=True),
    sa.Column("storage_ref", sa.String(512), nullable=True),
    sa.Column("content_type", sa.String(128), nullable=False),
    sa.Column("size_bytes", sa.BigInteger(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "artifact_id", name="fervis_artifact_run_id_unique"),
    sa.UniqueConstraint(
        "run_id", "step_id", "artifact_id", name="fervis_artifact_step_id_uq"
    ),
    sa.CheckConstraint(
        "((content IS NOT NULL) AND (storage_ref IS NULL)) OR ((content IS NULL) AND (storage_ref IS NOT NULL) AND (storage_ref != ''))",
        name="fervis_artifact_one_body",
    ),
    sa.CheckConstraint(
        "size_bytes" + " >= 0", name="fervis_run_artifact_size_bytes_nonnegative_ck"
    ),
)
sa.Index(
    "fervis_artifact_step_idx",
    fervis_run_artifact.c.run_id,
    fervis_run_artifact.c.step_id,
)
sa.Index(
    "fervis_artifact_call_kind_idx",
    fervis_run_artifact.c.model_call_id,
    fervis_run_artifact.c.artifact_kind,
)
sa.Index("fervis_run_artifact_run_id_idx", fervis_run_artifact.c.run_id)
sa.Index("fervis_run_artifact_step_id_idx", fervis_run_artifact.c.step_id)
sa.Index("fervis_run_artifact_model_call_id_idx", fervis_run_artifact.c.model_call_id)

fervis_catalog_endpoint = sa.Table(
    "fervis_catalog_endpoint",
    metadata,
    sa.Column("catalog_endpoint_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_catalog_endpoint_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column("catalog_endpoint_key", sa.String(128), nullable=False),
    sa.Column("endpoint_name", sa.String(128), nullable=False),
    sa.Column("framework_kind", sa.String(64), nullable=False),
    sa.Column("source_namespace_kind", sa.String(64), nullable=False),
    sa.Column("source_namespace_path_json", sa.JSON(), nullable=False),
    sa.Column("route_method", sa.String(16), nullable=False),
    sa.Column("route_path_template", sa.String(512), nullable=False),
    sa.Column("route_name", sa.String(128), nullable=False),
    sa.Column("api_schema_operation_id", sa.String(128), nullable=False),
    sa.Column("handler_ref", sa.String(512), nullable=False),
    sa.Column("domain_resource_names_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "catalog_endpoint_id", name="fervis_cat_endpoint_run_id_uq"
    ),
    sa.UniqueConstraint(
        "run_id", "catalog_endpoint_key", name="fervis_cat_endpoint_run_key_uq"
    ),
    sa.CheckConstraint(
        "JSON_VALID(source_namespace_path_json)",
        name="fervis_catalog_endpoint_source_namespace_path_json_json_valid_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(domain_resource_names_json)",
        name="fervis_catalog_endpoint_domain_resource_names_json_json_valid_ck",
    ),
)
sa.Index("fervis_cat_endpoint_run_idx", fervis_catalog_endpoint.c.run_id)
sa.Index(
    "fervis_cat_endpoint_fw_idx",
    fervis_catalog_endpoint.c.framework_kind,
    fervis_catalog_endpoint.c.source_namespace_kind,
)
sa.Index("fervis_cat_endpoint_name_idx", fervis_catalog_endpoint.c.endpoint_name)
sa.Index("fervis_catalog_endpoint_run_id_idx", fervis_catalog_endpoint.c.run_id)
sa.Index(
    "fervis_catalog_endpoint_endpoint_name_idx", fervis_catalog_endpoint.c.endpoint_name
)

fervis_source_read = sa.Table(
    "fervis_source_read",
    metadata,
    sa.Column("source_read_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_source_read_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "step_id",
        sa.String(128),
        sa.ForeignKey("fervis_run_step.step_id", name="fervis_source_read_step_id_fk"),
        nullable=False,
    ),
    sa.Column(
        "catalog_endpoint_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_catalog_endpoint.catalog_endpoint_id",
            name="fervis_source_read_catalog_endpoint_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("args_json", sa.JSON(), nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("row_count", sa.Integer(), nullable=True),
    sa.Column("completeness_json", sa.JSON(), nullable=False),
    sa.Column("response_hash", sa.String(128), nullable=False),
    sa.Column(
        "artifact_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_artifact.artifact_id", name="fervis_source_read_artifact_id_fk"
        ),
        nullable=True,
    ),
    sa.Column("error_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "source_read_id", name="fervis_src_read_run_id_uq"),
    sa.CheckConstraint(
        "row_count" + " >= 0", name="fervis_source_read_row_count_nonnegative_ck"
    ),
    sa.CheckConstraint(
        "JSON_VALID(args_json)", name="fervis_source_read_args_json_json_valid_ck"
    ),
    sa.CheckConstraint(
        "JSON_VALID(completeness_json)",
        name="fervis_source_read_completeness_json_json_valid_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(error_json)", name="fervis_source_read_error_json_json_valid_ck"
    ),
)
sa.Index(
    "fervis_source_read_step_idx",
    fervis_source_read.c.run_id,
    fervis_source_read.c.step_id,
)
sa.Index("fervis_src_read_catalog_idx", fervis_source_read.c.catalog_endpoint_id)
sa.Index("fervis_source_read_run_id_idx", fervis_source_read.c.run_id)
sa.Index("fervis_source_read_step_id_idx", fervis_source_read.c.step_id)
sa.Index(
    "fervis_source_read_catalog_endpoint_id_idx",
    fervis_source_read.c.catalog_endpoint_id,
)
sa.Index("fervis_source_read_artifact_id_idx", fervis_source_read.c.artifact_id)

fervis_run_result = sa.Table(
    "fervis_run_result",
    metadata,
    sa.Column("run_result_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey("fervis_question_run.run_id", name="fervis_run_result_run_id_fk"),
        nullable=False,
        unique=True,
    ),
    sa.Column("result_kind", sa.String(32), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "run_result_id", name="fervis_run_result_run_id_uq"),
)

fervis_runtime_error_detail = sa.Table(
    "fervis_runtime_error_detail",
    metadata,
    sa.Column(
        "runtime_error_detail_id", sa.String(128), primary_key=True, nullable=False
    ),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_runtime_error_detail_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "run_result_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_result.run_result_id",
            name="fervis_runtime_error_detail_run_result_id_fk",
        ),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "failed_step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id",
            name="fervis_runtime_error_detail_failed_step_id_fk",
        ),
        nullable=True,
    ),
    sa.Column("error_kind", sa.String(64), nullable=False),
    sa.Column("message", sa.Text(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "runtime_error_detail_id", name="fervis_runtime_err_run_id_uq"
    ),
)
sa.Index(
    "fervis_runtime_error_step_idx",
    fervis_runtime_error_detail.c.run_id,
    fervis_runtime_error_detail.c.failed_step_id,
)
sa.Index("fervis_runtime_error_detail_run_id_idx", fervis_runtime_error_detail.c.run_id)
sa.Index(
    "fervis_runtime_error_detail_failed_step_id_idx",
    fervis_runtime_error_detail.c.failed_step_id,
)

fervis_requested_fact = sa.Table(
    "fervis_requested_fact",
    metadata,
    sa.Column("requested_fact_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_requested_fact_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "produced_by_step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id",
            name="fervis_requested_fact_produced_by_step_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("fact_key", sa.String(128), nullable=False),
    sa.Column("description", sa.Text(), nullable=False),
    sa.Column("answer_expression_family", sa.String(64), nullable=False),
    sa.Column("requested_fact_json", sa.JSON(), nullable=False),
    sa.Column("answer_requests_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "requested_fact_id", name="fervis_req_fact_run_id_uq"
    ),
    sa.UniqueConstraint(
        "run_id", "produced_by_step_id", "fact_key", name="fervis_req_fact_step_key_uq"
    ),
    sa.CheckConstraint(
        "JSON_VALID(requested_fact_json)",
        name="fervis_requested_fact_requested_fact_json_json_valid_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(answer_requests_json)",
        name="fervis_requested_fact_answer_requests_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_req_fact_key_idx",
    fervis_requested_fact.c.run_id,
    fervis_requested_fact.c.fact_key,
)
sa.Index("fervis_requested_fact_run_id_idx", fervis_requested_fact.c.run_id)
sa.Index(
    "fervis_requested_fact_produced_by_step_id_idx",
    fervis_requested_fact.c.produced_by_step_id,
)

fervis_fact_result = sa.Table(
    "fervis_fact_result",
    metadata,
    sa.Column("fact_result_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_fact_result_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "requested_fact_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_requested_fact.requested_fact_id",
            name="fervis_fact_result_requested_fact_id_fk",
        ),
        nullable=False,
    ),
    sa.Column(
        "produced_by_step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id", name="fervis_fact_result_produced_by_step_id_fk"
        ),
        nullable=False,
    ),
    sa.Column("result_kind", sa.String(64), nullable=False),
    sa.Column("evidence_refs_json", sa.JSON(), nullable=False),
    sa.Column("payload_schema", sa.String(128), nullable=False),
    sa.Column("payload_schema_rev", sa.Integer(), nullable=True),
    sa.Column("payload_json", sa.JSON(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "fact_result_id", name="fervis_fact_result_run_id_uq"
    ),
    sa.CheckConstraint(
        "payload_schema_rev" + " >= 0",
        name="fervis_fact_result_payload_schema_rev_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(evidence_refs_json)",
        name="fervis_fact_result_evidence_refs_json_json_valid_ck",
    ),
    sa.CheckConstraint(
        "payload_json IS NULL OR JSON_VALID(payload_json)",
        name="fervis_fact_result_payload_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_fact_result_fact_idx",
    fervis_fact_result.c.run_id,
    fervis_fact_result.c.requested_fact_id,
)
sa.Index(
    "fervis_fact_result_kind_idx",
    fervis_fact_result.c.run_id,
    fervis_fact_result.c.result_kind,
)
sa.Index("fervis_fact_result_run_id_idx", fervis_fact_result.c.run_id)
sa.Index(
    "fervis_fact_result_requested_fact_id_idx", fervis_fact_result.c.requested_fact_id
)
sa.Index(
    "fervis_fact_result_produced_by_step_id_idx",
    fervis_fact_result.c.produced_by_step_id,
)

fervis_memory_artifact = sa.Table(
    "fervis_memory_artifact",
    metadata,
    sa.Column("memory_artifact_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_memory_artifact_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "produced_by_step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id",
            name="fervis_memory_artifact_produced_by_step_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("source_kind", sa.String(64), nullable=False),
    sa.Column(
        "requested_fact_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_requested_fact.requested_fact_id",
            name="fervis_memory_artifact_requested_fact_id_fk",
        ),
        nullable=True,
    ),
    sa.Column(
        "fact_result_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_fact_result.fact_result_id",
            name="fervis_memory_artifact_fact_result_id_fk",
        ),
        nullable=True,
    ),
    sa.Column("payload_schema", sa.String(128), nullable=False),
    sa.Column("payload_schema_rev", sa.Integer(), nullable=False),
    sa.Column("payload_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "memory_artifact_id", name="fervis_memory_art_run_id_uq"
    ),
    sa.CheckConstraint(
        "payload_schema_rev" + " >= 0",
        name="fervis_memory_artifact_payload_schema_rev_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(payload_json)",
        name="fervis_memory_artifact_payload_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_memory_art_source_idx",
    fervis_memory_artifact.c.run_id,
    fervis_memory_artifact.c.source_kind,
)
sa.Index(
    "fervis_memory_art_req_idx",
    fervis_memory_artifact.c.run_id,
    fervis_memory_artifact.c.requested_fact_id,
)
sa.Index(
    "fervis_memory_art_result_idx",
    fervis_memory_artifact.c.run_id,
    fervis_memory_artifact.c.fact_result_id,
)
sa.Index("fervis_memory_artifact_run_id_idx", fervis_memory_artifact.c.run_id)
sa.Index(
    "fervis_memory_artifact_produced_by_step_id_idx",
    fervis_memory_artifact.c.produced_by_step_id,
)
sa.Index(
    "fervis_memory_artifact_requested_fact_id_idx",
    fervis_memory_artifact.c.requested_fact_id,
)
sa.Index(
    "fervis_memory_artifact_fact_result_id_idx", fervis_memory_artifact.c.fact_result_id
)

fervis_clarification_request = sa.Table(
    "fervis_clarification_request",
    metadata,
    sa.Column("clarification_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_clarification_request_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "fact_result_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_fact_result.fact_result_id",
            name="fervis_clarification_request_fact_result_id_fk",
        ),
        nullable=True,
    ),
    sa.Column(
        "step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id", name="fervis_clarification_request_step_id_fk"
        ),
        nullable=True,
    ),
    sa.Column("need", sa.String(64), nullable=False),
    sa.Column("reason", sa.String(64), nullable=False),
    sa.Column("payload_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "clarification_id", name="fervis_clar_req_run_id_uq"),
    sa.CheckConstraint(
        "JSON_VALID(payload_json)",
        name="fervis_clarification_request_payload_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_clar_req_need_idx",
    fervis_clarification_request.c.run_id,
    fervis_clarification_request.c.need,
)
sa.Index(
    "fervis_clar_req_reason_idx",
    fervis_clarification_request.c.run_id,
    fervis_clarification_request.c.reason,
)
sa.Index(
    "fervis_clarification_request_run_id_idx", fervis_clarification_request.c.run_id
)
sa.Index(
    "fervis_clarification_request_fact_result_id_idx",
    fervis_clarification_request.c.fact_result_id,
)
sa.Index(
    "fervis_clarification_request_step_id_idx", fervis_clarification_request.c.step_id
)

fervis_clarification_response = sa.Table(
    "fervis_clarification_response",
    metadata,
    sa.Column("response_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_clarification_response_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "clarification_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_clarification_request.clarification_id",
            name="fervis_clarification_response_clarification_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("source_message_ref", sa.String(255), nullable=False),
    sa.Column("selected_option_id", sa.String(128), nullable=False),
    sa.Column("response_text", sa.Text(), nullable=False),
    sa.Column("evidence_ref", sa.String(255), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "response_id", name="fervis_clar_resp_run_id_uq"),
)
sa.Index(
    "fervis_clar_resp_req_idx",
    fervis_clarification_response.c.run_id,
    fervis_clarification_response.c.clarification_id,
)
sa.Index(
    "fervis_clarification_response_run_id_idx", fervis_clarification_response.c.run_id
)
sa.Index(
    "fervis_clarification_response_clarification_id_idx",
    fervis_clarification_response.c.clarification_id,
)

fervis_answer = sa.Table(
    "fervis_answer",
    metadata,
    sa.Column("answer_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey("fervis_question_run.run_id", name="fervis_answer_run_id_fk"),
        nullable=False,
    ),
    sa.Column(
        "run_result_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_result.run_result_id", name="fervis_answer_run_result_id_fk"
        ),
        nullable=False,
        unique=True,
    ),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("run_id", "answer_id", name="fervis_answer_run_id_uq"),
)
sa.Index("fervis_answer_run_id_idx", fervis_answer.c.run_id)

fervis_answer_output = sa.Table(
    "fervis_answer_output",
    metadata,
    sa.Column("answer_output_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_answer_output_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "answer_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_answer.answer_id", name="fervis_answer_output_answer_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "fact_result_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_fact_result.fact_result_id",
            name="fervis_answer_output_fact_result_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("output_key", sa.String(128), nullable=False),
    sa.Column("value_kind", sa.String(64), nullable=False),
    sa.Column("value_json", sa.JSON(), nullable=False),
    sa.Column("proof_node_refs_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "answer_output_id", name="fervis_answer_output_run_id_uq"
    ),
    sa.UniqueConstraint(
        "run_id", "answer_id", "output_key", name="fervis_answer_output_key_uq"
    ),
    sa.CheckConstraint(
        "JSON_VALID(value_json)", name="fervis_answer_output_value_json_json_valid_ck"
    ),
    sa.CheckConstraint(
        "JSON_VALID(proof_node_refs_json)",
        name="fervis_answer_output_proof_node_refs_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_answer_output_ans_idx",
    fervis_answer_output.c.run_id,
    fervis_answer_output.c.answer_id,
)
sa.Index(
    "fervis_answer_output_fact_idx",
    fervis_answer_output.c.run_id,
    fervis_answer_output.c.fact_result_id,
)
sa.Index("fervis_answer_output_run_id_idx", fervis_answer_output.c.run_id)
sa.Index("fervis_answer_output_answer_id_idx", fervis_answer_output.c.answer_id)
sa.Index(
    "fervis_answer_output_fact_result_id_idx", fervis_answer_output.c.fact_result_id
)

fervis_answer_presentation = sa.Table(
    "fervis_answer_presentation",
    metadata,
    sa.Column("presentation_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_answer_presentation_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "answer_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_answer.answer_id", name="fervis_answer_presentation_answer_id_fk"
        ),
        nullable=False,
    ),
    sa.Column("client_key", sa.String(64), nullable=False),
    sa.Column("locale", sa.String(32), nullable=False),
    sa.Column("presentation_kind", sa.String(64), nullable=False),
    sa.Column("rendered_value", sa.Text(), nullable=True),
    sa.Column(
        "render_step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id",
            name="fervis_answer_presentation_render_step_id_fk",
        ),
        nullable=False,
    ),
    sa.Column("payload_schema", sa.String(128), nullable=False),
    sa.Column("payload_schema_rev", sa.Integer(), nullable=True),
    sa.Column("payload_json", sa.JSON(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "presentation_id", name="fervis_answer_pres_run_id_uq"
    ),
    sa.UniqueConstraint(
        "run_id",
        "answer_id",
        "client_key",
        "locale",
        "presentation_kind",
        name="fervis_answer_pres_shape_uq",
    ),
    sa.CheckConstraint(
        "((payload_json IS NULL) AND (rendered_value IS NOT NULL)) OR ((payload_json IS NOT NULL) AND (rendered_value IS NULL))",
        name="fervis_answer_pres_body_ck",
    ),
    sa.CheckConstraint(
        "payload_schema_rev" + " >= 0",
        name="fervis_answer_presentation_payload_schema_rev_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "payload_json IS NULL OR JSON_VALID(payload_json)",
        name="fervis_answer_presentation_payload_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_answer_pres_ans_idx",
    fervis_answer_presentation.c.run_id,
    fervis_answer_presentation.c.answer_id,
)
sa.Index("fervis_answer_presentation_run_id_idx", fervis_answer_presentation.c.run_id)
sa.Index(
    "fervis_answer_presentation_answer_id_idx", fervis_answer_presentation.c.answer_id
)
sa.Index(
    "fervis_answer_presentation_render_step_id_idx",
    fervis_answer_presentation.c.render_step_id,
)

fervis_execution_proof_graph = sa.Table(
    "fervis_execution_proof_graph",
    metadata,
    sa.Column("proof_graph_id", sa.String(128), primary_key=True, nullable=False),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_question_run.run_id", name="fervis_execution_proof_graph_run_id_fk"
        ),
        nullable=False,
    ),
    sa.Column(
        "fact_result_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_fact_result.fact_result_id",
            name="fervis_execution_proof_graph_fact_result_id_fk",
        ),
        nullable=False,
        unique=True,
    ),
    sa.Column(
        "compile_step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id",
            name="fervis_execution_proof_graph_compile_step_id_fk",
        ),
        nullable=False,
    ),
    sa.Column(
        "execute_step_id",
        sa.String(128),
        sa.ForeignKey(
            "fervis_run_step.step_id",
            name="fervis_execution_proof_graph_execute_step_id_fk",
        ),
        nullable=True,
    ),
    sa.Column("payload_schema", sa.String(128), nullable=False),
    sa.Column("payload_schema_rev", sa.Integer(), nullable=False),
    sa.Column("payload_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint(
        "run_id", "proof_graph_id", name="fervis_proof_graph_run_id_uq"
    ),
    sa.CheckConstraint(
        "payload_schema_rev" + " >= 0",
        name="fervis_execution_proof_graph_payload_schema_rev_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(payload_json)",
        name="fervis_execution_proof_graph_payload_json_json_valid_ck",
    ),
)
sa.Index(
    "fervis_proof_fact_idx",
    fervis_execution_proof_graph.c.run_id,
    fervis_execution_proof_graph.c.fact_result_id,
)
sa.Index(
    "fervis_execution_proof_graph_run_id_idx", fervis_execution_proof_graph.c.run_id
)
sa.Index(
    "fervis_execution_proof_graph_compile_step_id_idx",
    fervis_execution_proof_graph.c.compile_step_id,
)
sa.Index(
    "fervis_execution_proof_graph_execute_step_id_idx",
    fervis_execution_proof_graph.c.execute_step_id,
)

fervis_run_work_item = sa.Table(
    "fervis_run_work_item",
    metadata,
    sa.Column(
        "id",
        sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
        primary_key=True,
        nullable=False,
        autoincrement=True,
    ),
    sa.Column("run_id", sa.String(128), nullable=False, unique=True),
    sa.Column("conversation_id", sa.String(128), nullable=False),
    sa.Column("tenant_id", sa.String(128), nullable=False),
    sa.Column("user_id", sa.String(128), nullable=False),
    sa.Column("read_context_ref", sa.JSON(), nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("spec_kind", sa.String(32), nullable=False),
    sa.Column("execution_spec", sa.JSON(), nullable=False),
    sa.Column("idempotency_key", sa.String(255), nullable=True),
    sa.Column("attempt_count", sa.Integer(), nullable=False),
    sa.Column("active_attempt", sa.Integer(), nullable=False),
    sa.Column("max_attempts", sa.Integer(), nullable=False),
    sa.Column("lease_owner", sa.String(128), nullable=True),
    sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("last_error", sa.Text(), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        "attempt_count" + " >= 0",
        name="fervis_run_work_item_attempt_count_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "active_attempt" + " >= 0",
        name="fervis_run_work_item_active_attempt_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "max_attempts" + " >= 0",
        name="fervis_run_work_item_max_attempts_nonnegative_ck",
    ),
    sa.CheckConstraint(
        "JSON_VALID(execution_spec)",
        name="fervis_run_work_item_execution_spec_json_valid_ck",
    ),
)
sa.Index(
    "fervis_work_idempotency_unique",
    fervis_run_work_item.c.tenant_id,
    fervis_run_work_item.c.conversation_id,
    fervis_run_work_item.c.idempotency_key,
    unique=True,
    sqlite_where=sa.text("(idempotency_key IS NOT NULL)"),
)
sa.Index(
    "fervis_work_active_conversation_unique",
    fervis_run_work_item.c.tenant_id,
    fervis_run_work_item.c.conversation_id,
    unique=True,
    sqlite_where=sa.text("(status IN ('QUEUED', 'RUNNING'))"),
)
sa.Index(
    "fervis_work_claim_idx",
    fervis_run_work_item.c.status,
    fervis_run_work_item.c.next_attempt_at,
    fervis_run_work_item.c.created_at,
)
sa.Index(
    "fervis_work_lease_idx",
    fervis_run_work_item.c.status,
    fervis_run_work_item.c.lease_expires_at,
)
sa.Index(
    "fervis_work_conv_idx",
    fervis_run_work_item.c.tenant_id,
    fervis_run_work_item.c.conversation_id,
)
sa.Index(
    "fervis_run_work_item_conversation_id_idx", fervis_run_work_item.c.conversation_id
)
sa.Index("fervis_run_work_item_tenant_id_idx", fervis_run_work_item.c.tenant_id)
sa.Index("fervis_run_work_item_status_idx", fervis_run_work_item.c.status)
sa.Index(
    "fervis_run_work_item_lease_expires_at_idx", fervis_run_work_item.c.lease_expires_at
)
sa.Index(
    "fervis_run_work_item_next_attempt_at_idx", fervis_run_work_item.c.next_attempt_at
)


def _add_run_scoped_foreign_keys() -> None:
    _run_fk(
        fervis_model_call,
        ("run_id", "step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_model_call_run_step_fk",
    )
    _run_fk(
        fervis_model_call_usage,
        ("run_id", "model_call_id"),
        "fervis_model_call",
        ("run_id", "model_call_id"),
        "fervis_model_call_usage_run_call_fk",
    )
    _run_fk(
        fervis_run_artifact,
        ("run_id", "step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_run_artifact_run_step_fk",
    )
    _run_fk(
        fervis_run_artifact,
        ("run_id", "model_call_id"),
        "fervis_model_call",
        ("run_id", "model_call_id"),
        "fervis_run_artifact_run_model_call_fk",
    )
    _run_fk(
        fervis_source_read,
        ("run_id", "step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_source_read_run_step_fk",
    )
    _run_fk(
        fervis_source_read,
        ("run_id", "catalog_endpoint_id"),
        "fervis_catalog_endpoint",
        ("run_id", "catalog_endpoint_id"),
        "fervis_source_read_run_catalog_fk",
    )
    _run_fk(
        fervis_source_read,
        ("run_id", "artifact_id"),
        "fervis_run_artifact",
        ("run_id", "artifact_id"),
        "fervis_source_read_run_artifact_fk",
    )
    _run_fk(
        fervis_runtime_error_detail,
        ("run_id", "run_result_id"),
        "fervis_run_result",
        ("run_id", "run_result_id"),
        "fervis_runtime_error_run_result_fk",
    )
    _run_fk(
        fervis_runtime_error_detail,
        ("run_id", "failed_step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_runtime_error_run_step_fk",
    )
    _run_fk(
        fervis_requested_fact,
        ("run_id", "produced_by_step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_requested_fact_run_step_fk",
    )
    _run_fk(
        fervis_fact_result,
        ("run_id", "requested_fact_id"),
        "fervis_requested_fact",
        ("run_id", "requested_fact_id"),
        "fervis_fact_result_run_requested_fact_fk",
    )
    _run_fk(
        fervis_fact_result,
        ("run_id", "produced_by_step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_fact_result_run_step_fk",
    )
    _run_fk(
        fervis_memory_artifact,
        ("run_id", "produced_by_step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_memory_artifact_run_step_fk",
    )
    _run_fk(
        fervis_memory_artifact,
        ("run_id", "requested_fact_id"),
        "fervis_requested_fact",
        ("run_id", "requested_fact_id"),
        "fervis_memory_artifact_run_requested_fact_fk",
    )
    _run_fk(
        fervis_memory_artifact,
        ("run_id", "fact_result_id"),
        "fervis_fact_result",
        ("run_id", "fact_result_id"),
        "fervis_memory_artifact_run_fact_result_fk",
    )
    _run_fk(
        fervis_clarification_request,
        ("run_id", "fact_result_id"),
        "fervis_fact_result",
        ("run_id", "fact_result_id"),
        "fervis_clarification_request_run_fact_result_fk",
    )
    _run_fk(
        fervis_clarification_request,
        ("run_id", "step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_clarification_request_run_step_fk",
    )
    _run_fk(
        fervis_clarification_response,
        ("run_id", "clarification_id"),
        "fervis_clarification_request",
        ("run_id", "clarification_id"),
        "fervis_clarification_response_run_request_fk",
    )
    _run_fk(
        fervis_answer,
        ("run_id", "run_result_id"),
        "fervis_run_result",
        ("run_id", "run_result_id"),
        "fervis_answer_run_result_fk",
    )
    _run_fk(
        fervis_answer_output,
        ("run_id", "answer_id"),
        "fervis_answer",
        ("run_id", "answer_id"),
        "fervis_answer_output_run_answer_fk",
    )
    _run_fk(
        fervis_answer_output,
        ("run_id", "fact_result_id"),
        "fervis_fact_result",
        ("run_id", "fact_result_id"),
        "fervis_answer_output_run_fact_result_fk",
    )
    _run_fk(
        fervis_answer_presentation,
        ("run_id", "answer_id"),
        "fervis_answer",
        ("run_id", "answer_id"),
        "fervis_answer_presentation_run_answer_fk",
    )
    _run_fk(
        fervis_answer_presentation,
        ("run_id", "render_step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_answer_presentation_run_step_fk",
    )
    _run_fk(
        fervis_execution_proof_graph,
        ("run_id", "fact_result_id"),
        "fervis_fact_result",
        ("run_id", "fact_result_id"),
        "fervis_execution_proof_graph_run_fact_result_fk",
    )
    _run_fk(
        fervis_execution_proof_graph,
        ("run_id", "compile_step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_execution_proof_graph_run_compile_step_fk",
    )
    _run_fk(
        fervis_execution_proof_graph,
        ("run_id", "execute_step_id"),
        "fervis_run_step",
        ("run_id", "step_id"),
        "fervis_execution_proof_graph_run_execute_step_fk",
    )


def _run_fk(
    table: sa.Table,
    columns: tuple[str, ...],
    referred_table: str,
    referred_columns: tuple[str, ...],
    name: str,
) -> None:
    table.append_constraint(
        sa.ForeignKeyConstraint(
            columns,
            [f"{referred_table}.{column}" for column in referred_columns],
            name=name,
        )
    )


_add_run_scoped_foreign_keys()
