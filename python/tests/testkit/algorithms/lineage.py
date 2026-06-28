from __future__ import annotations

from dataclasses import dataclass

from fervis.lineage.enums import (
    AnswerValueKind,
    ClarificationBasis,
    FactResultKind,
    MemoryArtifactSourceKind,
    PresentationClientKey,
    PresentationKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
    RuntimeErrorKind,
    SourceReadStatus,
)
from fervis.lineage.memory_artifacts import MemoryArtifactRow
from fervis.lineage.views.query import (
    AnswerOutputRow,
    AnswerPresentationRow,
    AnswerRow,
    CatalogEndpointRow,
    ClarificationRequestRow,
    ClarificationResponseRow,
    ConversationRow,
    FactResultRow,
    LineageQueryPort,
    LineageRows,
    ProofGraphRow,
    QuestionRow,
    RequestedFactRow,
    RunResultRow,
    RunRow,
    RuntimeErrorRow,
    SourceReadRow,
    StepRow,
)
from fervis.lineage.views.input_lineage import (
    input_lineage_view,
    render_input_lineage,
)
from fervis.lineage.views.render import render_lineage
from fervis.lineage.views.service import (
    AnswerLineageService,
    ConversationLineageService,
    QuestionLineageService,
)
from fervis.lineage.views.timeline import lineage_timeline_view
from tests.testkit.assertions import subset_mismatches


def run_lineage_explain_case(payload: dict) -> list[str]:
    input_payload = payload["input"]
    query = _FixtureLineageQuery(_rows(input_payload["dataset"]))
    view = _lineage_root(query, input_payload["root"])
    rendered = render_lineage(lineage_timeline_view(view))
    errors = _compare_rendered(rendered, payload["expect"], line_key="lines")
    expected_view = payload["expect"].get("result_contains", {}).get("view")
    if expected_view:
        errors.extend(
            subset_mismatches(
                actual={"view": _portable_lineage_view(view)},
                expected_subset={"view": expected_view},
            )
        )
    for root in input_payload.get("alternate_roots", ()):
        alternate = render_lineage(lineage_timeline_view(_lineage_root(query, root)))
        errors.extend(
            _compare_rendered(
                alternate,
                payload["expect"],
                line_key="alternate_lines",
            )
    )
    return errors


def run_lineage_input_lineage_case(payload: dict) -> list[str]:
    input_payload = payload["input"]
    query = _FixtureLineageQuery(_rows(input_payload["dataset"]))
    rendered = render_input_lineage(
        input_lineage_view(
            _lineage_root(query, input_payload["root"]),
            answer_output=input_payload.get("answer_output"),
        )
    )
    return _compare_rendered(rendered, payload["expect"], line_key="lines")


def fixture_lineage_query(payload: dict) -> LineageQueryPort:
    return _FixtureLineageQuery(_rows(payload))


def fixture_lineage_rows(payload: dict) -> LineageRows:
    return _rows(payload)


def _render_root(query: _FixtureLineageQuery, root: dict) -> str:
    return render_lineage(lineage_timeline_view(_lineage_root(query, root)))


def _lineage_root(query: _FixtureLineageQuery, root: dict):
    root_id = str(root["id"])
    kind = str(root["kind"])
    if kind == "answer":
        return AnswerLineageService(query).for_answer(root_id)
    if kind == "question":
        return QuestionLineageService(query).for_question(root_id)
    if kind == "run":
        return QuestionLineageService(query).for_run(root_id)
    if kind == "conversation":
        return ConversationLineageService(query).for_conversation(root_id)
    raise ValueError(f"unsupported lineage root kind: {kind}")


def _compare_rendered(rendered: str, expect: dict, *, line_key: str) -> list[str]:
    errors: list[str] = []
    for line in expect.get("result_contains", {}).get(line_key, ()):
        if line not in rendered:
            errors.append(f"missing compact lineage line: {line}")
    for text in expect.get("text_excludes", ()):
        if text in rendered:
            errors.append(f"compact lineage included excluded text: {text}")
    return errors


def _portable_lineage_view(view) -> dict:
    return {
        "questions": {
            question.question_id: {
                "runs": {
                    run.run_id: _portable_run_view(run) for run in question.runs
                }
            }
            for question in view.questions
        },
        "runs": {
            run.run_id: _portable_run_view(run)
            for question in view.questions
            for run in question.runs
        },
    }


def _portable_run_view(run) -> dict:
    return {
        "result_kind": run.result_kind,
        "activated_memory_ids": list(run.activated_memory_ids),
    }


@dataclass(frozen=True)
class _FixtureLineageQuery(LineageQueryPort):
    rows: LineageRows

    def run_id_for_answer(self, answer_id: str) -> str | None:
        for answer in self.rows.answers:
            if answer.answer_id == answer_id:
                return answer.run_id
        return None

    def run_by_id(self, run_id: str) -> RunRow | None:
        for run in self.rows.runs:
            if run.run_id == run_id:
                return run
        return None

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        return tuple(run.run_id for run in self.rows.runs if run.run_id == run_id)

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        return tuple(
            run.run_id for run in self.rows.runs if run.question_id == question_id
        )

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        question_ids = {
            question.question_id
            for question in self.rows.questions
            if question.conversation_id == conversation_id
        }
        return tuple(
            run.run_id for run in self.rows.runs if run.question_id in question_ids
        )

    def lineage_rows_for_run_ids(self, run_ids: tuple[str, ...]) -> LineageRows:
        run_id_set = set(run_ids)
        run_rows = tuple(run for run in self.rows.runs if run.run_id in run_id_set)
        question_ids = {run.question_id for run in run_rows}
        questions = tuple(
            question
            for question in self.rows.questions
            if question.question_id in question_ids
        )
        conversation_ids = {question.conversation_id for question in questions}
        source_reads = tuple(
            item for item in self.rows.source_reads if item.run_id in run_id_set
        )
        catalog_endpoint_ids = {
            source_read.catalog_endpoint_id for source_read in source_reads
        }
        return LineageRows(
            conversations=tuple(
                item
                for item in self.rows.conversations
                if item.conversation_id in conversation_ids
            ),
            questions=questions,
            runs=run_rows,
            steps=tuple(item for item in self.rows.steps if item.run_id in run_id_set),
            run_results=tuple(
                item for item in self.rows.run_results if item.run_id in run_id_set
            ),
            runtime_errors=tuple(
                item for item in self.rows.runtime_errors if item.run_id in run_id_set
            ),
            clarification_requests=tuple(
                item
                for item in self.rows.clarification_requests
                if item.run_id in run_id_set
            ),
            clarification_responses=tuple(
                item
                for item in self.rows.clarification_responses
                if item.run_id in run_id_set
            ),
            requested_facts=tuple(
                item for item in self.rows.requested_facts if item.run_id in run_id_set
            ),
            fact_results=tuple(
                item for item in self.rows.fact_results if item.run_id in run_id_set
            ),
            memory_artifacts=tuple(
                item
                for item in self.rows.memory_artifacts
                if item.run_id in run_id_set
            ),
            answers=tuple(
                item for item in self.rows.answers if item.run_id in run_id_set
            ),
            answer_outputs=tuple(
                item for item in self.rows.answer_outputs if item.run_id in run_id_set
            ),
            answer_presentations=tuple(
                item
                for item in self.rows.answer_presentations
                if item.run_id in run_id_set
            ),
            catalog_endpoints=tuple(
                item
                for item in self.rows.catalog_endpoints
                if item.catalog_endpoint_id in catalog_endpoint_ids
            ),
            source_reads=source_reads,
            proof_graphs=tuple(
                item for item in self.rows.proof_graphs if item.run_id in run_id_set
            ),
        )


def _rows(payload: dict) -> LineageRows:
    return LineageRows(
        conversations=tuple(
            ConversationRow(
                conversation_id=str(item["conversation_id"]),
                tenant_id=str(item["tenant_id"]),
            )
            for item in payload.get("conversations", ())
        ),
        questions=tuple(
            QuestionRow(
                question_id=str(item["question_id"]),
                conversation_id=str(item["conversation_id"]),
                conversation_sequence=int(item["conversation_sequence"]),
                original_question=str(item["original_question"]),
            )
            for item in payload.get("questions", ())
        ),
        runs=tuple(
            RunRow(
                run_id=str(item["run_id"]),
                question_id=str(item["question_id"]),
                run_number=int(item["run_number"]),
                trigger_kind=RunTriggerKind(str(item["trigger_kind"])),
                integrated_question=str(item["integrated_question"]),
                previous_run_id=item.get("previous_run_id"),
                trigger_clarification_response_run_id=item.get(
                    "trigger_clarification_response_run_id"
                ),
                trigger_clarification_response_id=item.get(
                    "trigger_clarification_response_id"
                ),
            )
            for item in payload.get("runs", ())
        ),
        steps=tuple(_step(item) for item in payload.get("steps", ())),
        run_results=tuple(
            RunResultRow(
                run_result_id=str(item["run_result_id"]),
                run_id=str(item["run_id"]),
                result_kind=RunResultKind(str(item["result_kind"])),
            )
            for item in payload.get("run_results", ())
        ),
        runtime_errors=tuple(
            RuntimeErrorRow(
                runtime_error_detail_id=str(item["runtime_error_detail_id"]),
                run_id=str(item["run_id"]),
                run_result_id=str(item["run_result_id"]),
                failed_step_id=item.get("failed_step_id"),
                error_kind=RuntimeErrorKind(str(item["error_kind"])),
                message=str(item["message"]),
            )
            for item in payload.get("runtime_errors", ())
        ),
        clarification_requests=tuple(
            ClarificationRequestRow(
                clarification_id=str(item["clarification_id"]),
                run_id=str(item["run_id"]),
                basis=ClarificationBasis(str(item["basis"])),
                question_text=str(item["question_text"]),
                fact_result_id=item.get("fact_result_id"),
                step_id=item.get("step_id"),
                options_json=tuple(item.get("options_json") or ()),
                evidence_refs_json=tuple(item.get("evidence_refs_json") or ()),
            )
            for item in payload.get("clarification_requests", ())
        ),
        clarification_responses=tuple(
            ClarificationResponseRow(
                response_id=str(item["response_id"]),
                run_id=str(item["run_id"]),
                clarification_id=str(item["clarification_id"]),
                evidence_ref=str(item["evidence_ref"]),
                source_message_ref=str(item.get("source_message_ref") or ""),
                selected_option_id=str(item.get("selected_option_id") or ""),
                response_text=str(item.get("response_text") or ""),
            )
            for item in payload.get("clarification_responses", ())
        ),
        requested_facts=tuple(
            RequestedFactRow(
                requested_fact_id=str(item["requested_fact_id"]),
                run_id=str(item["run_id"]),
                produced_by_step_id=str(item["produced_by_step_id"]),
                fact_key=str(item["fact_key"]),
                description=str(item.get("description") or ""),
                answer_expression_family=str(item["answer_expression_family"]),
                requested_fact_json=dict(item.get("requested_fact_json") or {}),
                answer_requests_json=dict(item.get("answer_requests_json") or {}),
            )
            for item in payload.get("requested_facts", ())
        ),
        fact_results=tuple(
            FactResultRow(
                fact_result_id=str(item["fact_result_id"]),
                run_id=str(item["run_id"]),
                requested_fact_id=str(item["requested_fact_id"]),
                produced_by_step_id=str(item["produced_by_step_id"]),
                result_kind=FactResultKind(str(item["result_kind"])),
                evidence_refs_json=tuple(item.get("evidence_refs_json") or ()),
                payload_schema=str(item.get("payload_schema") or ""),
                payload_schema_rev=item.get("payload_schema_rev"),
                payload_json=item.get("payload_json"),
            )
            for item in payload.get("fact_results", ())
        ),
        memory_artifacts=tuple(
            MemoryArtifactRow(
                memory_artifact_id=str(item["memory_artifact_id"]),
                run_id=str(item["run_id"]),
                produced_by_step_id=str(item["produced_by_step_id"]),
                source_kind=MemoryArtifactSourceKind(str(item["source_kind"])),
                payload_schema=str(item["payload_schema"]),
                payload_schema_rev=int(item["payload_schema_rev"]),
                payload_json=dict(item["payload_json"]),
                requested_fact_id=item.get("requested_fact_id"),
                fact_result_id=item.get("fact_result_id"),
            )
            for item in payload.get("memory_artifacts", ())
        ),
        answers=tuple(
            AnswerRow(
                answer_id=str(item["answer_id"]),
                run_id=str(item["run_id"]),
                run_result_id=str(item["run_result_id"]),
            )
            for item in payload.get("answers", ())
        ),
        answer_outputs=tuple(
            AnswerOutputRow(
                answer_output_id=str(item["answer_output_id"]),
                run_id=str(item["run_id"]),
                answer_id=str(item["answer_id"]),
                fact_result_id=str(item["fact_result_id"]),
                output_key=str(item["output_key"]),
                value_kind=AnswerValueKind(str(item["value_kind"])),
                value_json=dict(item["value_json"]),
                proof_node_refs_json=tuple(item.get("proof_node_refs_json") or ()),
            )
            for item in payload.get("answer_outputs", ())
        ),
        answer_presentations=tuple(
            AnswerPresentationRow(
                presentation_id=str(item["presentation_id"]),
                run_id=str(item["run_id"]),
                answer_id=str(item["answer_id"]),
                client_key=PresentationClientKey(str(item.get("client_key") or "default")),
                locale=str(item.get("locale") or "default"),
                presentation_kind=PresentationKind(str(item["presentation_kind"])),
                render_step_id=str(item["render_step_id"]),
                rendered_value=str(item.get("rendered_value") or ""),
                payload_schema=str(item.get("payload_schema") or ""),
                payload_schema_rev=item.get("payload_schema_rev"),
                payload_json=item.get("payload_json"),
            )
            for item in payload.get("answer_presentations", ())
        ),
        catalog_endpoints=tuple(
            CatalogEndpointRow(
                catalog_endpoint_id=str(item["catalog_endpoint_id"]),
                run_id=str(item["run_id"]),
                catalog_endpoint_key=str(item["catalog_endpoint_key"]),
                endpoint_name=str(item["endpoint_name"]),
                framework_kind=str(item["framework_kind"]),
                source_namespace_kind=str(item["source_namespace_kind"]),
                source_namespace_path_json=tuple(
                    str(value)
                    for value in item.get("source_namespace_path_json") or ()
                ),
                route_method=str(item["route_method"]),
                route_path_template=str(item["route_path_template"]),
                route_name=str(item.get("route_name") or ""),
                api_schema_operation_id=str(
                    item.get("api_schema_operation_id") or ""
                ),
                handler_ref=str(item.get("handler_ref") or ""),
                domain_resource_names_json=tuple(
                    str(value)
                    for value in item.get("domain_resource_names_json") or ()
                ),
            )
            for item in payload.get("catalog_endpoints", ())
        ),
        source_reads=tuple(
            SourceReadRow(
                source_read_id=str(item["source_read_id"]),
                run_id=str(item["run_id"]),
                step_id=str(item["step_id"]),
                catalog_endpoint_id=str(item["catalog_endpoint_id"]),
                args_json=dict(item.get("args_json") or {}),
                status=SourceReadStatus(str(item["status"])),
                row_count=item.get("row_count"),
                completeness_json=dict(item.get("completeness_json") or {}),
                response_hash=str(item.get("response_hash") or ""),
                artifact_id=item.get("artifact_id"),
                error_json=dict(item.get("error_json") or {}),
            )
            for item in payload.get("source_reads", ())
        ),
        proof_graphs=tuple(
            ProofGraphRow(
                proof_graph_id=str(item["proof_graph_id"]),
                run_id=str(item["run_id"]),
                fact_result_id=str(item["fact_result_id"]),
                compile_step_id=str(item["compile_step_id"]),
                execute_step_id=item.get("execute_step_id"),
                payload_schema=str(item["payload_schema"]),
                payload_schema_rev=int(item["payload_schema_rev"]),
                payload_json=dict(item["payload_json"]),
            )
            for item in payload.get("proof_graphs", ())
        ),
    )


def _step(item: dict) -> StepRow:
    return StepRow(
        step_id=str(item["step_id"]),
        run_id=str(item["run_id"]),
        sequence=int(item["sequence"]),
        step_key=RunStepKey(str(item["step_key"])),
        kind=RunStepKind(str(item["kind"])),
        input_summary_json=dict(item.get("input_summary_json") or {}),
        output_summary_json=dict(item.get("output_summary_json") or {}),
        error_json=dict(item.get("error_json") or {}),
    )
