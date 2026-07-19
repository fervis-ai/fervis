from __future__ import annotations

from dataclasses import dataclass

from fervis.lineage.enums import (
    AnswerValueKind,
    FactResultKind,
    MemoryArtifactSourceKind,
    PresentationClientKey,
    PresentationKind,
    ProofEdgeRole,
    ProofNodeKind,
    QuestionRunKind,
    RunResultKind,
    RunStepKind,
    RunStepKey,
    RunTriggerKind,
    SourceReadStatus,
)
from fervis.lineage.payloads.execution_proof_graph import (
    EXECUTION_PROOF_GRAPH_SCHEMA,
    EXECUTION_PROOF_GRAPH_SCHEMA_REV,
)
from fervis.lineage.memory_artifacts import MemoryArtifactRow
from fervis.lineage.step_summary import (
    StepSemanticItem,
    StepSummaryItem,
    merge_step_summary_json,
    step_semantic_json,
)
from fervis.lineage.views.agent import agent_lineage_view
from fervis.lineage.views.detail import LineageRenderDetail
from fervis.lineage.views.explain import ExplainView
from fervis.lineage.views.explanation import answer_explanation_view
from fervis.lineage.views.json_payload import view_json
from fervis.lineage.views.query import (
    AnswerOutputRow,
    AnswerPresentationRow,
    AnswerRow,
    CatalogEndpointRow,
    ConversationRow,
    FactResultRow,
    LineageQueryPort,
    LineageRows,
    ProofGraphRow,
    QuestionRow,
    RequestedFactRow,
    RunResultRow,
    RunRow,
    SourceReadRow,
    StepRow,
)
from tests.testkit.execution_proof_graph import proof_graph_payload, proof_node
from fervis.lineage.views.render import render_lineage
from fervis.lineage.views.service import AnswerLineageService
from fervis.lineage.views.timeline import lineage_timeline_view


def test_lineage_view_keeps_proof_source_reads_scoped_to_their_run() -> None:
    view = AnswerLineageService(_OverbroadLineageQuery(_lineage_rows())).for_answer(
        "answer_1"
    )

    proof = view.questions[0].runs[0].requested_facts[0].fact_results[0].proof

    assert proof is not None
    assert [item.catalog_endpoint.endpoint_name for item in proof.source_reads] == [
        "list_shift_compensation_list"
    ]


def test_lineage_view_keeps_answer_outputs_scoped_to_their_run() -> None:
    view = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_cross_run_answer_output())
    ).for_answer("answer_1")

    outputs = view.questions[0].runs[0].requested_facts[0].answer_outputs

    assert [(item.output_key, item.value) for item in outputs] == [
        ("answer_1", "staff:staff_id=staff_1")
    ]


def test_lineage_view_preserves_structured_answer_output_values() -> None:
    view = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_structured_answer_output())
    ).for_answer("answer_1")

    output = view.questions[0].runs[0].requested_facts[0].answer_outputs[0]

    assert output.value_kind == AnswerValueKind.TABLE.value
    assert output.value_json == {
        "kind": "table",
        "columns": ["staff", "sales"],
        "rows": [["staff_1", "12000.00"]],
    }


def test_compact_lineage_render_hides_memory_storage_mechanics() -> None:
    view = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_memory_artifact())
    ).for_answer("answer_1")

    output = render_lineage(lineage_timeline_view(view))

    assert "source question: previous question" in output
    assert "fervis.memory_artifact" not in output
    assert "payload_schema" not in output


def test_lineage_render_scopes_proof_to_selected_answer_output() -> None:
    view = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_two_answer_output_branches())
    ).for_answer("answer_1")

    output = render_lineage(lineage_timeline_view(view), answer_output="answer_1")

    assert "list_shift_compensation_list" in output
    assert "list_bonus_report" not in output


def test_lineage_render_omits_full_answer_presentation_when_output_scoped() -> None:
    view = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_two_answer_output_branches())
    ).for_answer("answer_1")

    output = render_lineage(lineage_timeline_view(view), answer_output="answer_1")

    assert "staff:staff_id=staff_1" in output
    assert "location_1" not in output


def test_answer_explanation_json_exposes_semantic_step_contract() -> None:
    lineage = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_semantic_step())
    ).for_answer("answer_1")
    payload = view_json(
        answer_explanation_view(
            ExplainView(
                lineage=lineage,
                timeline=lineage_timeline_view(lineage),
            )
        )
    )

    step = payload["lineage"]["verbose"]["questions"][0]["runs"][0]["steps"][0]

    assert step["stepKey"] == "question_contract"
    assert step["semantic"] == {
        "requestedFacts": [
            {
                "requestedFactId": "fact_1",
                "description": "sales at ABC Mall this month",
            }
        ],
        "knownInputs": [
            {
                "inputId": "fact_1_entity_1",
                "text": "ABC Mall",
                "kind": "literal_text",
                "role": "reference_value",
                "description": "store",
                "resolvedValueText": "ABC Mall",
            }
        ],
        "resolverCandidates": [],
        "groundingResults": [],
        "interpretedInputs": [
            {
                "inputId": "fact_1_time_1",
                "inputText": "this month",
                "kind": "time",
                "value": "2026-06-01 to 2026-06-30",
                "label": "this month",
                "detail": "month",
            }
        ],
        "conversationClauses": [],
    }


def test_agent_lineage_exposes_semantics_and_structured_decision_basis() -> None:
    lineage = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_semantic_step())
    ).for_answer("answer_1")

    payload = agent_lineage_view(
        lineage_timeline_view(lineage),
        detail=LineageRenderDetail.VERBOSE,
    )

    step = payload["questions"][0]["runs"][0]["steps"][0]
    assert step["semantic"]["knownInputs"][0] == {
        "inputId": "fact_1_entity_1",
        "text": "ABC Mall",
        "kind": "literal_text",
        "role": "reference_value",
        "description": "store",
        "resolvedValueText": "ABC Mall",
    }
    assert step["decisions"][0]["items"] == (
        {
            "text": "source_1: RETAIN - matches the requested sales measure.",
            "is_explanation": True,
            "subject": "source_1",
            "disposition": "RETAIN",
            "basis": "Matches the requested sales measure.",
        },
    )


def test_answer_explanation_json_exposes_conversation_resolution_semantics() -> None:
    lineage = AnswerLineageService(
        _OverbroadLineageQuery(_lineage_rows_with_conversation_resolution_step())
    ).for_answer("answer_1")
    payload = view_json(
        answer_explanation_view(
            ExplainView(
                lineage=lineage,
                timeline=lineage_timeline_view(lineage),
            )
        )
    )

    step = payload["lineage"]["verbose"]["questions"][0]["runs"][0]["steps"][0]

    assert step["stepKey"] == "conversation_resolution"
    assert step["semantic"]["conversationClauses"] == [
        {
            "currentClauseText": "what about last month?",
            "resolvedText": "how many completed in-person sales last month?",
            "resolvedValues": ["last month"],
        }
    ]


@dataclass(frozen=True)
class _OverbroadLineageQuery(LineageQueryPort):
    rows: LineageRows

    def run_id_for_answer(self, answer_id: str) -> str | None:
        return "run_1" if answer_id == "answer_1" else None

    def run_by_id(self, run_id: str) -> RunRow | None:
        return next((run for run in self.rows.runs if run.run_id == run_id), None)

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        return (run_id,)

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        return tuple(
            run.run_id for run in self.rows.runs if run.question_id == question_id
        )

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        del conversation_id
        return tuple(run.run_id for run in self.rows.runs)

    def lineage_rows_for_run_ids(self, run_ids: tuple[str, ...]) -> LineageRows:
        del run_ids
        return self.rows


def _lineage_rows() -> LineageRows:
    return LineageRows(
        conversations=(ConversationRow("conversation_1", "tenant_1"),),
        questions=(
            QuestionRow("question_1", "conversation_1", 1, "Which staff earned most?"),
        ),
        runs=(
            RunRow(
                "run_1",
                "question_1",
                1,
                QuestionRunKind.MODEL_ASSISTED,
                RunTriggerKind.INITIAL,
            ),
            RunRow(
                "run_2",
                "question_1",
                2,
                QuestionRunKind.MODEL_ASSISTED,
                RunTriggerKind.INITIAL,
            ),
        ),
        steps=(
            StepRow(
                "step_compile",
                "run_1",
                1,
                RunStepKey.COMPILE,
                RunStepKind.DETERMINISTIC,
            ),
            StepRow(
                "step_execute",
                "run_1",
                2,
                RunStepKey.EXECUTE,
                RunStepKind.DETERMINISTIC,
            ),
        ),
        run_results=(RunResultRow("result_1", "run_1", RunResultKind.ANSWERED),),
        requested_facts=(
            RequestedFactRow(
                "fact_1",
                "run_1",
                "step_compile",
                "fact_1",
                "staff member who earned most",
                "ranked_groups",
            ),
        ),
        fact_results=(
            FactResultRow(
                "fact_result_1",
                "run_1",
                "fact_1",
                "step_execute",
                FactResultKind.ANSWERED,
            ),
        ),
        answers=(AnswerRow("answer_1", "run_1", "result_1"),),
        answer_outputs=(
            AnswerOutputRow(
                "answer_output_1",
                "run_1",
                "answer_1",
                "fact_result_1",
                "answer_1",
                AnswerValueKind.ENTITY,
                {
                    "kind": "entity",
                    "entity_kind": "staff",
                    "key_id": "primary_key",
                    "components": {"staff_id": "staff_1"},
                },
                proof_node_refs_json=("answer_output:fact_1:answer_1",),
            ),
        ),
        catalog_endpoints=(
            CatalogEndpointRow(
                "list_shift_compensation_list",
                "run_1",
                "django_retail_ops_list_shift_compensation_list:test",
                "list_shift_compensation_list",
                "django",
                "django_app",
                ("retail_ops",),
                "GET",
                "/v1/shift-compensations/",
            ),
            CatalogEndpointRow(
                "wrong_run_endpoint",
                "run_2",
                "django_retail_ops_wrong_run_endpoint:test",
                "wrong_run_endpoint",
                "django",
                "django_app",
                ("retail_ops",),
                "GET",
                "/v1/wrong-run/",
            ),
        ),
        source_reads=(
            SourceReadRow(
                "source_read_1",
                "run_2",
                "step_execute",
                "wrong_run_endpoint",
                {},
                SourceReadStatus.SUCCEEDED,
            ),
            SourceReadRow(
                "source_read_1",
                "run_1",
                "step_execute",
                "list_shift_compensation_list",
                {},
                SourceReadStatus.SUCCEEDED,
            ),
        ),
        proof_graphs=(
            ProofGraphRow(
                "proof_1",
                "run_1",
                "fact_result_1",
                "step_compile",
                "step_execute",
                EXECUTION_PROOF_GRAPH_SCHEMA,
                EXECUTION_PROOF_GRAPH_SCHEMA_REV,
                proof_graph_payload(
                    nodes=(
                        proof_node(
                            "relation:source_1",
                            ProofNodeKind.RELATION.value,
                            proof_refs=("source_read:source_read_1",),
                        ),
                        proof_node(
                            "answer_output:fact_1:answer_1",
                            ProofNodeKind.ANSWER_OUTPUT.value,
                        ),
                    ),
                    edges=(
                        {
                            "source": "relation:source_1",
                            "target": "answer_output:fact_1:answer_1",
                            "role": ProofEdgeRole.PRODUCES.value,
                        },
                    ),
                ),
            ),
        ),
    )


def _lineage_rows_with_cross_run_answer_output() -> LineageRows:
    rows = _lineage_rows()
    return LineageRows(
        **{
            **rows.__dict__,
            "answer_outputs": (
                *rows.answer_outputs,
                AnswerOutputRow(
                    "answer_output_cross_run",
                    "run_2",
                    "answer_2",
                    "fact_result_1",
                    "answer_1",
                    AnswerValueKind.ENTITY,
                    {
                        "kind": "entity",
                        "entity_kind": "staff",
                        "key_id": "primary_key",
                        "components": {"staff_id": "wrong_staff"},
                    },
                ),
            ),
        }
    )


def _lineage_rows_with_structured_answer_output() -> LineageRows:
    rows = _lineage_rows()
    return LineageRows(
        **{
            **rows.__dict__,
            "answer_outputs": (
                AnswerOutputRow(
                    "answer_output_1",
                    "run_1",
                    "answer_1",
                    "fact_result_1",
                    "answer_1",
                    AnswerValueKind.TABLE,
                    {
                        "kind": "table",
                        "columns": ["staff", "sales"],
                        "rows": [["staff_1", "12000.00"]],
                    },
                    proof_node_refs_json=("answer_output:fact_1:answer_1",),
                ),
            ),
        }
    )


def _lineage_rows_with_semantic_step() -> LineageRows:
    rows = _lineage_rows()
    return LineageRows(
        **{
            **rows.__dict__,
            "steps": (
                StepRow(
                    "step_compile",
                    "run_1",
                    1,
                    RunStepKey.QUESTION_CONTRACT,
                    RunStepKind.MODEL_TURN,
                    output_summary_json=merge_step_summary_json(
                        step_semantic_json(
                            StepSemanticItem(
                                kind="requested_fact",
                                payload={
                                    "requested_fact_id": "fact_1",
                                    "description": "sales at ABC Mall this month",
                                },
                            ),
                            StepSemanticItem(
                                kind="known_input",
                                payload={
                                    "input_id": "fact_1_entity_1",
                                    "text": "ABC Mall",
                                    "kind": "literal_text",
                                    "role": "reference_value",
                                    "description": "store",
                                    "resolved_value_text": "ABC Mall",
                                },
                            ),
                            StepSemanticItem(
                                kind="interpreted_input",
                                payload={
                                    "input_id": "fact_1_time_1",
                                    "input_text": "this month",
                                    "kind": "time",
                                    "value": "2026-06-01 to 2026-06-30",
                                    "label": "this month",
                                    "detail": "month",
                                },
                            ),
                        ),
                        StepSummaryItem(
                            text=(
                                "source_1: RETAIN - matches the requested sales "
                                "measure."
                            ),
                            is_explanation=True,
                            subject="source_1",
                            disposition="RETAIN",
                            basis="Matches the requested sales measure.",
                        ),
                    ),
                ),
                rows.steps[1],
            ),
        }
    )


def _lineage_rows_with_conversation_resolution_step() -> LineageRows:
    rows = _lineage_rows()
    return LineageRows(
        **{
            **rows.__dict__,
            "steps": (
                StepRow(
                    "step_compile",
                    "run_1",
                    1,
                    RunStepKey.CONVERSATION_RESOLUTION,
                    RunStepKind.MODEL_TURN,
                    output_summary_json=step_semantic_json(
                        StepSemanticItem(
                            kind="conversation_clause",
                            payload={
                                "current_clause_text": "what about last month?",
                                "resolved_text": (
                                    "how many completed in-person sales last month?"
                                ),
                                "resolved_values": ("last month",),
                            },
                        ),
                    ),
                ),
                rows.steps[1],
            ),
        }
    )


def _lineage_rows_with_memory_artifact() -> LineageRows:
    rows = _lineage_rows()
    return LineageRows(
        **{
            **rows.__dict__,
            "memory_artifacts": (
                MemoryArtifactRow(
                    "memory_artifact_1",
                    "run_1",
                    "step_execute",
                    MemoryArtifactSourceKind.FACT_RESULT,
                    "fervis.memory_artifact",
                    1,
                    {
                        "sourceQuestion": "previous question",
                        "sourceAnswer": "previous answer",
                        "outcome": "answered",
                        "addresses": [
                            {
                                "address": "value.answer_1",
                                "kind": "value",
                            }
                        ],
                    },
                    fact_result_id="fact_result_1",
                ),
            ),
        }
    )


def _lineage_rows_with_two_answer_output_branches() -> LineageRows:
    rows = _lineage_rows()
    return LineageRows(
        **{
            **rows.__dict__,
            "answer_outputs": (
                *rows.answer_outputs,
                AnswerOutputRow(
                    "answer_output_2",
                    "run_1",
                    "answer_1",
                    "fact_result_1",
                    "answer_2",
                    AnswerValueKind.ENTITY,
                    {"entity_type": "location", "entity_id": "location_1"},
                    proof_node_refs_json=("answer_output:fact_1:answer_2",),
                ),
            ),
            "catalog_endpoints": (
                *rows.catalog_endpoints,
                CatalogEndpointRow(
                    "list_bonus_report",
                    "run_1",
                    "django_retail_ops_list_bonus_report:test",
                    "list_bonus_report",
                    "django",
                    "django_app",
                    ("retail_ops",),
                    "GET",
                    "/v1/bonus-report/",
                ),
            ),
            "source_reads": (
                *rows.source_reads,
                SourceReadRow(
                    "source_read_2",
                    "run_1",
                    "step_execute",
                    "list_bonus_report",
                    {},
                    SourceReadStatus.SUCCEEDED,
                ),
            ),
            "proof_graphs": (
                ProofGraphRow(
                    "proof_1",
                    "run_1",
                    "fact_result_1",
                    "step_compile",
                    "step_execute",
                    EXECUTION_PROOF_GRAPH_SCHEMA,
                    EXECUTION_PROOF_GRAPH_SCHEMA_REV,
                    proof_graph_payload(
                        nodes=(
                            proof_node(
                                "relation:payroll",
                                ProofNodeKind.RELATION.value,
                                proof_refs=("source_read:source_read_1",),
                            ),
                            proof_node(
                                "relation:bonus",
                                ProofNodeKind.RELATION.value,
                                proof_refs=("source_read:source_read_2",),
                            ),
                            proof_node(
                                "answer_output:fact_1:answer_1",
                                ProofNodeKind.ANSWER_OUTPUT.value,
                            ),
                            proof_node(
                                "answer_output:fact_1:answer_2",
                                ProofNodeKind.ANSWER_OUTPUT.value,
                            ),
                        ),
                        edges=(
                            {
                                "source": "relation:payroll",
                                "target": "answer_output:fact_1:answer_1",
                                "role": ProofEdgeRole.PRODUCES.value,
                            },
                            {
                                "source": "relation:bonus",
                                "target": "answer_output:fact_1:answer_2",
                                "role": ProofEdgeRole.PRODUCES.value,
                            },
                        ),
                    ),
                ),
            ),
            "answer_presentations": (
                AnswerPresentationRow(
                    "presentation_1",
                    "run_1",
                    "answer_1",
                    PresentationClientKey.DEFAULT,
                    "",
                    PresentationKind.TEXT,
                    "step_execute",
                    rendered_value="staff_1 at location_1",
                ),
            ),
        }
    )
