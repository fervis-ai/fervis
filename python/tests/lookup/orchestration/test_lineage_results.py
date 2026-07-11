from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from fervis.lineage.enums import (
    AnswerValueKind,
    FactResultKind,
    MemoryArtifactSourceKind,
    ProofEdgeRole,
    ProofNodeKind,
    RunResultKind,
    RuntimeErrorKind,
)
from fervis.lineage.recorder import (
    AnsweredRunResultWrite,
    FactualTerminalRunResultWrite,
    RunStepWrite,
    RuntimeErrorResultWrite,
)
from fervis.lookup.clarification import (
    MissingAnswerMetric,
    TargetReferenceNotFound,
    clarify,
)
from fervis.lookup.answer_program.instantiation import (
    ExecutionProofEdge,
    ExecutionProofGraph,
    ExecutionProofNode,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.outcomes.model import (
    AnswerResult,
    EmptyRelation,
    EmptyRelationKind,
    FactResult,
    NeedsClarification,
    NoData,
    OutcomeKind,
)
from fervis.lookup.answer_program.render_spec import (
    RenderRelationOutput,
    RenderScalarOutput,
    RenderSpec,
)
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerOutput,
)
from fervis.lookup.turn_prompts.context import active_clarification_context
from fervis.lookup.answer_rendering import RenderedFact
from fervis.lookup.lineage.results import (
    LineagePersistenceUnavailable,
    RuntimeErrorTerminal,
    record_answered_result_lineage,
    record_lookup_result_lineage,
    runtime_error_terminal_result,
)
from fervis.lookup.lineage.steps import LineageRuntimeStepSink
from fervis.lookup.orchestration.result_synthesis import _synthesize_result
from fervis.lookup.orchestration.limits import _limit_before_next_model_turn
from fervis.lookup.orchestration.request import LookupRequest
from fervis.memory.addresses import FactAddress


def test_answered_lineage_records_only_fulfilled_answer_outputs() -> None:
    recorder = _Recorder()
    record_answered_result_lineage(
        request=_request("run_support_output"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=AnswerResult(
                proof_refs=("source_read:read_1", "source_read:read_2"),
            )
        ),
        rendered=RenderedFact(
            kind=OutcomeKind.ANSWER,
            rows=({"answer_1": "staff_1", "support_label": "Ada"},),
        ),
        answer="staff_1",
        question_contract=_question_contract({"fact_1": "answer_1"}),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        render_step_id="step_render",
        proof_graph=_proof_graph("answer_1"),
        answer_plan=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
                ),
            )
        ),
        proof_node_refs_by_render_output_id={
            "answer_1": ("answer_output:fact_1:answer_1",),
        },
    )

    answered = recorder.answered_results[0]
    assert [output.output_key for output in answered.outputs] == ["answer_1"]
    assert answered.outputs[0].proof_node_refs_json == ["answer_output:fact_1:answer_1"]


def test_answered_lineage_records_memory_artifact_from_fact_addresses() -> None:
    recorder = _Recorder()
    record_answered_result_lineage(
        request=_request("run_memory_artifact"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=AnswerResult(
                render_spec=RenderSpec(
                    scalar_outputs=(
                        RenderScalarOutput(id="answer_1", scalar_id="answer_1"),
                    )
                ),
                scalars={"answer_1": 14},
                proof_refs=("source_read:read_1",),
            )
        ),
        rendered=RenderedFact(
            kind=OutcomeKind.ANSWER,
            scalars={"answer_1": 14},
        ),
        answer="14",
        question_contract=_question_contract({"fact_1": "answer_1"}),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        render_step_id="step_render",
        proof_graph=_proof_graph("answer_1"),
        answer_plan=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
                ),
            )
        ),
        proof_node_refs_by_render_output_id={
            "answer_1": ("answer_output:fact_1:answer_1",),
        },
        conversation_resolution_activation={
            "activated_memory_ids": ["artifact_old.value.total"],
        },
    )

    memory_artifact = _single_memory_artifact(
        recorder.answered_results[0].memory_artifacts,
        source_kind=MemoryArtifactSourceKind.FACT_RESULT,
    )

    assert memory_artifact.fact_result_id == (
        recorder.answered_results[0].fact_results[0].fact_result_id
    )
    assert memory_artifact.payload_json["sourceKind"] == "fact_result"
    assert memory_artifact.payload_json["addresses"] == [
        {
            "address": "value.answer_1",
            "kind": "value",
            "value": {"type": "decimal", "value": "14"},
            "derivation": {
                "source": "operation_output",
                "answer_output_ids": ["answer_1"],
            },
            "evidence": {"stepIds": ["source_read:read_1"]},
        }
    ]
    assert memory_artifact.payload_json["provenance"][
        "conversation_resolution_activation"
    ] == {"activated_memory_ids": ["artifact_old.value.total"]}


def test_answered_lineage_records_entity_output_from_execution_relation() -> None:
    recorder = _Recorder()
    record_answered_result_lineage(
        request=_request("run_entity_output"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=AnswerResult(
                render_spec=RenderSpec(
                    relation_outputs=(
                        RenderRelationOutput(
                            id="answer_1",
                            relation_id="result",
                            field_id="staff_id",
                        ),
                    )
                ),
                relations=(
                    RelationRows(
                        id="result",
                        rows=({"staff_id": "staff_1"},),
                        grain_keys=("staff_id",),
                        identity_type="staff",
                    ),
                ),
                proof_refs=("source_read:read_1",),
            )
        ),
        rendered=RenderedFact(
            kind=OutcomeKind.ANSWER,
            rows=({"answer_1": "staff_1"},),
        ),
        answer="staff_1",
        question_contract=_question_contract({"fact_1": "answer_1"}),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        render_step_id="step_render",
        proof_graph=_proof_graph("answer_1"),
        answer_plan=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
                ),
            )
        ),
        proof_node_refs_by_render_output_id={
            "answer_1": ("answer_output:fact_1:answer_1",),
        },
    )

    output = recorder.answered_results[0].outputs[0]

    assert output.value_kind is AnswerValueKind.ENTITY
    assert output.value_json == {
        "kind": "entity",
        "entity_type": "staff",
        "entity_id": "staff_1",
    }


def test_answered_lineage_links_each_output_to_its_requested_fact() -> None:
    recorder = _Recorder()
    record_answered_result_lineage(
        request=_request("run_multi_fact"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=AnswerResult(
                proof_refs=("source_read:read_1", "source_read:read_2"),
            )
        ),
        rendered=RenderedFact(
            kind=OutcomeKind.ANSWER,
            rows=({"answer_1": "London", "answer_2": "14"},),
        ),
        answer="London\n14",
        question_contract=_question_contract(
            {
                "fact_1": "answer_1",
                "fact_2": "answer_2",
            }
        ),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        render_step_id="step_render",
        proof_graph=_proof_graph("answer_1", "answer_2"),
        answer_plan=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
                ),
                FactFulfillment(
                    requested_fact_id="fact_2",
                    answer_output_id="answer_2",
                    render_output_id="answer_2",
                ),
            )
        ),
        proof_node_refs_by_render_output_id={
            "answer_1": ("answer_output:fact_1:answer_1",),
            "answer_2": ("answer_output:fact_2:answer_2",),
        },
    )

    answered = recorder.answered_results[0]
    fact_result_by_requested_fact = {
        fact.requested_fact_id: fact.fact_result_id for fact in answered.fact_results
    }
    requested_fact_id_by_key = {
        fact.fact_key: fact.requested_fact_id for fact in answered.requested_facts
    }
    fact_key_by_requested_fact_id = {
        requested_fact_id: fact_key
        for fact_key, requested_fact_id in requested_fact_id_by_key.items()
    }
    output_fact_results = {
        output.output_key: output.fact_result_id for output in answered.outputs
    }
    evidence_by_fact = {
        fact_key_by_requested_fact_id[fact.requested_fact_id]: fact.evidence_refs_json
        for fact in answered.fact_results
    }
    assert output_fact_results == {
        "answer_1": fact_result_by_requested_fact[requested_fact_id_by_key["fact_1"]],
        "answer_2": fact_result_by_requested_fact[requested_fact_id_by_key["fact_2"]],
    }
    assert evidence_by_fact == {
        "fact_1": ["source_read:read_1"],
        "fact_2": ["source_read:read_2"],
    }


def test_answered_lineage_memory_artifacts_are_requested_fact_scoped() -> None:
    recorder = _Recorder()
    known_input = FactAddress.entity(
        address="entity.area.london",
        resource="area",
        reference_text="London",
        identity={"area_id": "area_1"},
    )
    record_answered_result_lineage(
        request=_request("run_fact_scoped_memory"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=AnswerResult(
                render_spec=RenderSpec(
                    scalar_outputs=(
                        RenderScalarOutput(id="answer_1", scalar_id="answer_1"),
                        RenderScalarOutput(id="answer_2", scalar_id="answer_2"),
                    )
                ),
                scalars={"answer_1": 14, "answer_2": 9},
                proof_refs=("source_read:read_1",),
            )
        ),
        rendered=RenderedFact(
            kind=OutcomeKind.ANSWER,
            scalars={"answer_1": 14, "answer_2": 9},
        ),
        answer="14\n9",
        question_contract=_question_contract(
            {
                "fact_1": "answer_1",
                "fact_2": "answer_2",
            }
        ),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        render_step_id="step_render",
        proof_graph=_proof_graph("answer_1", "answer_2"),
        answer_plan=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
                ),
                FactFulfillment(
                    requested_fact_id="fact_2",
                    answer_output_id="answer_2",
                    render_output_id="answer_2",
                ),
            )
        ),
        proof_node_refs_by_render_output_id={
            "answer_1": ("answer_output:fact_1:answer_1",),
            "answer_2": ("answer_output:fact_2:answer_2",),
        },
        extra_fact_addresses=(known_input,),
        known_input_step_id="step_grounding",
    )

    artifacts_by_fact = {
        artifact.payload_json["provenance"]["requestedFactKey"]: artifact.payload_json
        for artifact in recorder.answered_results[0].memory_artifacts
        if artifact.source_kind is MemoryArtifactSourceKind.FACT_RESULT
    }

    assert {
        fact_id: (
            [address["address"] for address in payload["addresses"]],
            [
                item["id"]
                for item in payload["provenance"]["question_contract"][
                    "answer_requests"
                ]
            ],
        )
        for fact_id, payload in artifacts_by_fact.items()
    } == {
        "fact_1": (
            ["value.answer_1"],
            ["fact_1"],
        ),
        "fact_2": (
            ["value.answer_2"],
            ["fact_2"],
        ),
    }
    known_input_artifact = _single_memory_artifact(
        recorder.answered_results[0].memory_artifacts,
        source_kind=MemoryArtifactSourceKind.KNOWN_INPUT,
    )
    assert known_input_artifact.produced_by_step_id == "step_grounding"
    assert known_input_artifact.payload_json["addresses"][0]["address"] == (
        "entity.area.london"
    )
    assert all(
        "entity.area.london"
        not in [address["address"] for address in payload["addresses"]]
        for payload in artifacts_by_fact.values()
    )
    requested_fact_artifacts = [
        artifact
        for artifact in recorder.answered_results[0].memory_artifacts
        if artifact.source_kind is MemoryArtifactSourceKind.REQUESTED_FACT
    ]
    assert {artifact.requested_fact_id for artifact in requested_fact_artifacts} == {
        fact.requested_fact_id for fact in recorder.answered_results[0].requested_facts
    }


def test_answered_lineage_rejects_missing_fulfillment_render_output() -> None:
    recorder = _Recorder()

    with pytest.raises(ValueError, match="render output 'answer_1' is unavailable"):
        record_answered_result_lineage(
            request=_request("run_missing_render_output"),
            ports=_ports(recorder),
            fact_result=_answer_result(),
            rendered=RenderedFact(
                kind=OutcomeKind.ANSWER,
                rows=({"support_label": "Ada"},),
            ),
            answer="Ada",
            question_contract=_question_contract({"fact_1": "answer_1"}),
            question_contract_step_id="step_contract",
            compile_step_id="step_compile",
            execute_step_id="step_execute",
            render_step_id="step_render",
            proof_graph=_proof_graph("answer_1"),
            answer_plan=AnswerProgram(
                fulfillment=(
                    FactFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        render_output_id="answer_1",
                    ),
                )
            ),
            proof_node_refs_by_render_output_id={
                "answer_1": ("answer_output:fact_1:answer_1",),
            },
        )


def test_terminal_lineage_records_clarification_result() -> None:
    recorder = _Recorder()
    record_lookup_result_lineage(
        request=_request("run_needs_clarification"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=NeedsClarification(
                clarifications=(
                    clarify(
                        TargetReferenceNotFound(
                            clarification_id="clarify_1",
                            requested_fact_id="fact_1",
                            known_input_id="area",
                            source_text="",
                            target_label="area",
                        )
                    ),
                )
            )
        ),
        rendered=RenderedFact(kind=OutcomeKind.NEEDS_CLARIFICATION),
        answer="Which area should I use?",
        question_contract=_question_contract({"fact_1": "answer_1"}),
        question_contract_step_id="step_contract",
        compile_step_id=None,
        execute_step_id=None,
        render_step_id="step_render",
        proof_graph=None,
        answer_plan=None,
        proof_node_refs_by_render_output_id={},
    )

    terminal = recorder.terminal_results[0]
    assert terminal.result.result_kind is RunResultKind.FACTUAL_TERMINAL
    assert terminal.fact_results[0].result_kind is FactResultKind.NEEDS_CLARIFICATION
    clarification = terminal.clarifications[0]
    assert clarification.fact_result_id == terminal.fact_results[0].fact_result_id
    assert clarification.clarification_id == clarification.payload_json["id"]
    assert terminal.fact_results[0].payload_json == {
        "clarificationIds": [clarification.clarification_id],
    }
    memory_artifact = _single_memory_artifact(
        terminal.memory_artifacts,
        source_kind=MemoryArtifactSourceKind.FACT_RESULT,
    )
    assert memory_artifact.payload_json["addresses"][0]["clarificationQuestions"] == [
        "Which area should I use?"
    ]


def test_terminal_lineage_records_execution_proof_graph_for_no_data() -> None:
    recorder = _Recorder()
    record_lookup_result_lineage(
        request=_request("run_no_data"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=NoData(
                empty_relation=EmptyRelation(
                    kind=EmptyRelationKind.ANSWER_ROWS,
                    relation_id="result",
                    requested_fact_ids=("fact_1",),
                    proof_refs=("source_read:read_1",),
                )
            )
        ),
        rendered=RenderedFact(kind=OutcomeKind.NO_DATA),
        answer="No data",
        question_contract=_question_contract({"fact_1": "answer_1"}),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        render_step_id="step_render",
        proof_graph=_proof_graph("answer_1"),
        answer_plan=None,
        proof_node_refs_by_render_output_id={},
    )

    terminal = recorder.terminal_results[0]
    assert [fact.result_kind for fact in terminal.fact_results] == [
        FactResultKind.NO_DATA
    ]
    assert len(terminal.proof_graphs) == 1
    assert terminal.proof_graphs[0].fact_result_id == (
        terminal.fact_results[0].fact_result_id
    )
    assert terminal.proof_graphs[0].payload_json["nodes"]


def test_terminal_lineage_memory_artifacts_are_requested_fact_scoped() -> None:
    recorder = _Recorder()
    record_lookup_result_lineage(
        request=_request("run_multi_clarification"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=NeedsClarification(
                clarifications=(
                    clarify(
                        TargetReferenceNotFound(
                            clarification_id="clarify_1",
                            requested_fact_id="fact_1",
                            known_input_id="area",
                            source_text="",
                            target_label="area",
                        )
                    ),
                    clarify(
                        MissingAnswerMetric(
                            clarification_id="clarify_2",
                            requested_fact_id="fact_2",
                            source_text="total",
                            metric_needed="total",
                            proof_refs=("question_contract:metric",),
                        )
                    ),
                )
            )
        ),
        rendered=RenderedFact(kind=OutcomeKind.NEEDS_CLARIFICATION),
        answer="Clarification needed",
        question_contract=_question_contract(
            {
                "fact_1": "answer_1",
                "fact_2": "answer_2",
            }
        ),
        question_contract_step_id="step_contract",
        compile_step_id=None,
        execute_step_id=None,
        render_step_id="step_render",
        proof_graph=None,
        answer_plan=None,
        proof_node_refs_by_render_output_id={},
    )

    payloads_by_fact_result = {
        artifact.fact_result_id: artifact.payload_json
        for artifact in recorder.terminal_results[0].memory_artifacts
        if artifact.source_kind is MemoryArtifactSourceKind.FACT_RESULT
    }
    fact_key_by_requested_fact_id = {
        fact.requested_fact_id: fact.fact_key
        for fact in recorder.terminal_results[0].requested_facts
    }
    fact_result_id_by_fact_key = {
        fact_key_by_requested_fact_id[fact.requested_fact_id]: fact.fact_result_id
        for fact in recorder.terminal_results[0].fact_results
    }

    assert payloads_by_fact_result[fact_result_id_by_fact_key["fact_1"]]["addresses"][
        0
    ]["clarificationQuestions"] == ["Which area should I use?"]
    assert payloads_by_fact_result[fact_result_id_by_fact_key["fact_2"]]["addresses"][
        0
    ]["clarificationQuestions"] == ["Which metric should I use?"]


def test_pre_contract_terminal_lineage_records_run_scoped_clarification_memory() -> (
    None
):
    recorder = _Recorder()
    record_lookup_result_lineage(
        request=_request("run_pre_contract_clarification"),
        ports=_ports(recorder),
        fact_result=FactResult(
            outcome=NeedsClarification(
                clarifications=(
                    clarify(
                        TargetReferenceNotFound(
                            clarification_id="clarify_reference",
                            requested_fact_id="question_contract",
                            known_input_id="london",
                            source_text="",
                            target_label="London",
                        )
                    ),
                )
            )
        ),
        rendered=RenderedFact(kind=OutcomeKind.NEEDS_CLARIFICATION),
        answer="Which London should I use?",
        question_contract=None,
        question_contract_step_id="step_question_contract",
        compile_step_id=None,
        execute_step_id=None,
        render_step_id="step_render",
        proof_graph=None,
        answer_plan=None,
        proof_node_refs_by_render_output_id={},
    )

    terminal = recorder.terminal_results[0]
    assert terminal.requested_facts == ()
    assert terminal.fact_results == ()
    memory_artifact = _single_memory_artifact(
        terminal.memory_artifacts,
        source_kind=MemoryArtifactSourceKind.RUN_TERMINAL,
    )
    assert memory_artifact.payload_json["addresses"][0]["clarificationQuestions"] == [
        "Which London should I use?"
    ]


def test_active_clarification_context_ignores_addressless_terminal_artifacts() -> None:
    request_artifact = {
        "artifactId": "mem_requested_fact_1",
        "sourceKind": "requested_fact",
        "outcome": "needs_clarification",
        "sourceQuestion": "How much did we make yesterday?",
        "provenance": {"question_contract": {"answer_requests": []}},
    }
    clarification_artifact = {
        "artifactId": "mem_fact_result_1",
        "sourceKind": "fact_result",
        "outcome": "needs_clarification",
        "sourceQuestion": "How much did we make yesterday?",
        "addresses": [
            {
                "address": "outcome.needs_clarification",
                "kind": "outcome",
                "terminal": "needs_clarification",
                "clarificationQuestions": ["Which location?"],
            }
        ],
    }

    context = active_clarification_context(
        {"factArtifacts": [request_artifact, clarification_artifact]},
        current_question="ABC Mall",
    )

    assert context is not None
    assert len(context.exchanges) == 1
    assert context.exchanges[0].questions == ("Which location?",)
    assert context.exchanges[0].answer == "ABC Mall"


def test_lineage_required_rejects_missing_sink() -> None:
    with pytest.raises(LineagePersistenceUnavailable):
        record_lookup_result_lineage(
            request=_request("run_missing_sink"),
            ports=SimpleNamespace(lineage_step_sink=None, lineage_required=True),
            fact_result=_answer_result(),
            rendered=RenderedFact(
                kind=OutcomeKind.ANSWER,
                rows=({"answer_1": "staff_1"},),
            ),
            answer="staff_1",
            question_contract=_question_contract({"fact_1": "answer_1"}),
            question_contract_step_id="step_contract",
            compile_step_id="step_compile",
            execute_step_id="step_execute",
            render_step_id="step_render",
            proof_graph=_proof_graph("answer_1"),
            answer_plan=AnswerProgram(
                fulfillment=(
                    FactFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        render_output_id="answer_1",
                    ),
                )
            ),
            proof_node_refs_by_render_output_id={
                "answer_1": ("answer_output:fact_1:answer_1",),
            },
        )


def test_runtime_error_terminal_records_canonical_lineage_result() -> None:
    recorder = _Recorder()

    result = runtime_error_terminal_result(
        RuntimeErrorTerminal(
            run_id="run_runtime_error",
            error_code="planning_failed",
            message="source selection failed",
            usage={"costUsd": 1},
        ),
        recorder=recorder,
    )

    assert result.status == "FAILED"
    assert result.error == "planning_failed"
    runtime_error = recorder.runtime_errors[0]
    assert runtime_error.result.run_id == "run_runtime_error"
    assert runtime_error.result.result_kind is RunResultKind.RUNTIME_ERROR
    assert runtime_error.error.message == "source selection failed"


def test_policy_limit_failure_records_canonical_runtime_error_lineage() -> None:
    recorder = _Recorder()
    result = _limit_before_next_model_turn(
        SimpleNamespace(
            policy_port=_PolicyPort(),
            lineage_step_sink=LineageRuntimeStepSink(
                run_id="run_policy_limit",
                recorder=recorder,
            ),
            lineage_required=True,
        ),
        "run_policy_limit",
    )

    assert result is not None
    assert result.status == "FAILED"
    assert result.error == "max_budget_exceeded"
    runtime_error = recorder.runtime_errors[0]
    assert runtime_error.result.run_id == "run_policy_limit"
    assert runtime_error.error.message == "max_budget_exceeded"
    assert runtime_error.error.error_kind is RuntimeErrorKind.POLICY_LIMIT_EXCEEDED


def test_rendering_lineage_failure_fails_closed_with_runtime_error_result() -> None:
    recorder = _FailingAnsweredRecorder()

    result = _synthesize_result(
        request=_request("run_lineage_failure"),
        ports=_ports(recorder, run_id="run_lineage_failure"),
        fact_result=FactResult(
            outcome=AnswerResult(
                render_spec=RenderSpec(
                    scalar_outputs=(
                        RenderScalarOutput(id="answer_1", scalar_id="answer_1"),
                    )
                ),
                scalars={"answer_1": 14},
                proof_refs=("source_read:read_1",),
            )
        ),
        status="COMPLETED",
        usage={"costUsd": 1},
        question_contract=_question_contract({"fact_1": "answer_1"}),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        proof_graph=_proof_graph("answer_1"),
        answer_plan=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
                ),
            )
        ),
        proof_node_refs_by_render_output_id={
            "answer_1": ("answer_output:fact_1:answer_1",),
        },
    )

    assert result.status == "FAILED"
    assert result.error == "lineage_persistence_failed"
    runtime_error = recorder.runtime_errors[0]
    assert runtime_error.result.run_id == "run_lineage_failure"
    assert runtime_error.error.failed_step_id == recorder.steps[0].step_id
    assert runtime_error.error.message == "answered lineage write failed"


def test_planning_failure_records_canonical_runtime_error_kind() -> None:
    recorder = _Recorder()

    runtime_error_terminal_result(
        RuntimeErrorTerminal(
            run_id="run_planning_failed",
            error_code="planning_failed",
            message="planning failed",
        ),
        recorder=recorder,
    )

    assert (
        recorder.runtime_errors[0].error.error_kind is RuntimeErrorKind.PLANNING_FAILED
    )


def test_runtime_error_terminal_preserves_original_error_when_lineage_write_fails() -> (
    None
):
    result = runtime_error_terminal_result(
        RuntimeErrorTerminal(
            run_id="run_provider_auth_failed",
            error_code="provider_authentication_failed",
            message="provider authentication failed",
        ),
        recorder=_FailingRuntimeErrorRecorder(),
        lineage_required=True,
    )

    assert result.status == "FAILED"
    assert result.error == "provider_authentication_failed"


def test_lineage_failure_terminal_still_returns_failed_result_when_error_write_fails() -> (
    None
):
    result = _synthesize_result(
        request=_request("run_lineage_failure_write_failure"),
        ports=_ports(
            _FailingAnsweredAndRuntimeErrorRecorder(),
            run_id="run_lineage_failure_write_failure",
        ),
        fact_result=FactResult(
            outcome=AnswerResult(
                render_spec=RenderSpec(
                    scalar_outputs=(
                        RenderScalarOutput(id="answer_1", scalar_id="answer_1"),
                    )
                ),
                scalars={"answer_1": 14},
                proof_refs=("source_read:read_1",),
            )
        ),
        status="COMPLETED",
        usage={"costUsd": 1},
        question_contract=_question_contract({"fact_1": "answer_1"}),
        question_contract_step_id="step_contract",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        proof_graph=_proof_graph("answer_1"),
        answer_plan=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="fact_1",
                    answer_output_id="answer_1",
                    render_output_id="answer_1",
                ),
            )
        ),
        proof_node_refs_by_render_output_id={
            "answer_1": ("answer_output:fact_1:answer_1",),
        },
    )

    assert result.status == "FAILED"
    assert result.error == "lineage_persistence_failed"


@dataclass
class _Recorder:
    answered_results: list[AnsweredRunResultWrite] = field(default_factory=list)
    terminal_results: list[FactualTerminalRunResultWrite] = field(default_factory=list)
    runtime_errors: list[RuntimeErrorResultWrite] = field(default_factory=list)
    steps: list[RunStepWrite] = field(default_factory=list)

    def record_step(self, step: RunStepWrite) -> RunStepWrite:
        self.steps.append(step)
        return step

    def record_answered_result(
        self,
        answered_result: AnsweredRunResultWrite,
    ) -> AnsweredRunResultWrite:
        self.answered_results.append(answered_result)
        return answered_result

    def record_factual_terminal_result(
        self,
        terminal_result: FactualTerminalRunResultWrite,
    ) -> FactualTerminalRunResultWrite:
        self.terminal_results.append(terminal_result)
        return terminal_result

    def record_runtime_error_result(
        self,
        runtime_error: RuntimeErrorResultWrite,
    ) -> RuntimeErrorResultWrite:
        self.runtime_errors.append(runtime_error)
        return runtime_error


class _FailingAnsweredRecorder(_Recorder):
    def record_answered_result(
        self,
        answered_result: AnsweredRunResultWrite,
    ) -> AnsweredRunResultWrite:
        del answered_result
        raise ValueError("answered lineage write failed")


class _FailingAnsweredAndRuntimeErrorRecorder(_FailingAnsweredRecorder):
    def record_runtime_error_result(
        self,
        runtime_error: RuntimeErrorResultWrite,
    ) -> RuntimeErrorResultWrite:
        del runtime_error
        raise ValueError("runtime error lineage write failed")


class _FailingRuntimeErrorRecorder(_Recorder):
    def record_runtime_error_result(
        self,
        runtime_error: RuntimeErrorResultWrite,
    ) -> RuntimeErrorResultWrite:
        del runtime_error
        raise ValueError("runtime error lineage write failed")


class _PolicyPort:
    def failure_before_next_model_turn(self) -> SimpleNamespace:
        return SimpleNamespace(
            status="FAILED",
            error="max_budget_exceeded",
            result_data={"reason": "budget"},
            usage={"costUsd": 5},
        )


def _ports(recorder: _Recorder, *, run_id: str = "run_1") -> SimpleNamespace:
    return SimpleNamespace(
        lineage_step_sink=LineageRuntimeStepSink(run_id=run_id, recorder=recorder),
    )


def _single_memory_artifact(
    artifacts: tuple[object, ...],
    *,
    source_kind: MemoryArtifactSourceKind,
) -> object:
    matches = [
        artifact for artifact in artifacts if artifact.source_kind is source_kind
    ]
    assert len(matches) == 1
    return matches[0]


def _request(run_id: str) -> LookupRequest:
    return LookupRequest(
        question="Question?",
        run_id=run_id,
        tenant_id="tenant_1",
        provider_preferences={"provider": "fake", "modelKey": "FAKE"},
    )


def _answer_result() -> FactResult:
    return FactResult(outcome=AnswerResult(proof_refs=("source_read:read_1",)))


def _question_contract(answer_output_id_by_fact_id: dict[str, str]) -> QuestionContract:
    return QuestionContract(
        requested_facts=tuple(
            RequestedFact(
                id=fact_id,
                description="requested fact",
                answer_expression=RequestedFactAnswerExpression(
                    family=RequestedFactAnswerExpressionFamily.SCALAR_VALUE,
                ),
                answer_outputs=(
                    RequestedFactAnswerOutput(id=answer_output_id),
                ),
            )
            for fact_id, answer_output_id in answer_output_id_by_fact_id.items()
        ),
    )


def _proof_graph(*answer_output_ids: str) -> ExecutionProofGraph:
    answer_output_nodes = tuple(
        ExecutionProofNode(
            id=f"answer_output:fact_{index}:answer_{index}",
            kind=ProofNodeKind.ANSWER_OUTPUT,
        )
        for index, _ in enumerate(answer_output_ids, start=1)
    )
    relation_nodes = tuple(
        ExecutionProofNode(
            id=f"relation:fact_{index}",
            kind=ProofNodeKind.RELATION,
            proof_refs=(f"source_read:read_{index}",),
        )
        for index, _ in enumerate(answer_output_ids, start=1)
    )
    return ExecutionProofGraph(
        nodes=(*relation_nodes, *answer_output_nodes),
        edges=tuple(
            ExecutionProofEdge(
                source=f"relation:fact_{index}",
                target=f"answer_output:fact_{index}:answer_{index}",
                role=ProofEdgeRole.PRODUCES,
            )
            for index, _ in enumerate(answer_output_ids, start=1)
        ),
    )
