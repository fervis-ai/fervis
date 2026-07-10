from __future__ import annotations

from typing import Callable

from fervis.lineage.enums import (
    AnswerValueKind,
    FactResultKind,
    PresentationClientKey,
    PresentationKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
)
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.recorder import (
    AnsweredRunResultWrite,
    AnswerProgramWrite,
    AnswerOutputWrite,
    AnswerPresentationWrite,
    AnswerWrite,
    ExecutionProofGraphWrite,
    FactResultWrite,
    ProgramInvocationBundleWrite,
    ProgramInvocationWrite,
    RequestedFactWrite,
    RunResultWrite,
    RunStepWrite,
)
from fervis.questions.ports import LookupExecutionRequest


TerminalAnswerWriter = Callable[[LookupExecutionRequest, str, dict[str, object]], None]


def make_terminal_answer_writer(recorder: LineageRecorderPort) -> TerminalAnswerWriter:
    def write_terminal_answer(
        request: LookupExecutionRequest,
        answer: str,
        result_data: dict[str, object],
    ) -> None:
        run_id = request.run_id
        recorder.record_program_invocation(
            ProgramInvocationBundleWrite(
                program=AnswerProgramWrite(
                    program_id=f"{run_id}:program",
                    schema_revision=1,
                    canonical_json="{}",
                ),
                invocation=ProgramInvocationWrite(
                    invocation_id=f"{run_id}:invocation",
                    run_id=run_id,
                    program_id=f"{run_id}:program",
                    bindings_json="{}",
                ),
            )
        )
        for step in (
            RunStepWrite(
                step_id=f"{run_id}:contract",
                run_id=run_id,
                sequence=1,
                step_key=RunStepKey.QUESTION_CONTRACT,
                kind=RunStepKind.MODEL_TURN,
            ),
            RunStepWrite(
                step_id=f"{run_id}:compile",
                run_id=run_id,
                sequence=2,
                step_key=RunStepKey.COMPILE,
                kind=RunStepKind.DETERMINISTIC,
            ),
            RunStepWrite(
                step_id=f"{run_id}:execute",
                run_id=run_id,
                sequence=3,
                step_key=RunStepKey.EXECUTE,
                kind=RunStepKind.DETERMINISTIC,
            ),
            RunStepWrite(
                step_id=f"{run_id}:render",
                run_id=run_id,
                sequence=4,
                step_key=RunStepKey.RENDER,
                kind=RunStepKind.DETERMINISTIC,
            ),
        ):
            recorder.record_step(step)
        recorder.record_answered_result(
            AnsweredRunResultWrite(
                result=RunResultWrite(
                    run_result_id=f"{run_id}:result",
                    run_id=run_id,
                    result_kind=RunResultKind.ANSWERED,
                ),
                requested_facts=(
                    RequestedFactWrite(
                        requested_fact_id=f"{run_id}:fact",
                        run_id=run_id,
                        produced_by_step_id=f"{run_id}:contract",
                        fact_key="fact_1",
                        answer_expression_family="scalar_aggregate",
                    ),
                ),
                fact_results=(
                    FactResultWrite(
                        fact_result_id=f"{run_id}:fact-result",
                        run_id=run_id,
                        requested_fact_id=f"{run_id}:fact",
                        produced_by_step_id=f"{run_id}:execute",
                        result_kind=FactResultKind.ANSWERED,
                    ),
                ),
                proof_graphs=(
                    ExecutionProofGraphWrite(
                        proof_graph_id=f"{run_id}:proof",
                        run_id=run_id,
                        fact_result_id=f"{run_id}:fact-result",
                        compile_step_id=f"{run_id}:compile",
                        execute_step_id=f"{run_id}:execute",
                        payload_schema="fervis.execution_proof_graph",
                        payload_schema_rev=1,
                        payload_json={
                            "nodes": [
                                {
                                    "id": f"{run_id}:evidence",
                                    "kind": "relation",
                                    "proof_refs": [f"question:{run_id}"],
                                },
                                {
                                    "id": f"{run_id}:answer-node",
                                    "kind": "answer_output",
                                    "proof_refs": [],
                                },
                            ],
                            "edges": [
                                {
                                    "source": f"{run_id}:evidence",
                                    "target": f"{run_id}:answer-node",
                                    "role": "produces",
                                }
                            ],
                        },
                    ),
                ),
                answer=AnswerWrite(
                    answer_id=f"{run_id}:answer",
                    run_id=run_id,
                    run_result_id=f"{run_id}:result",
                ),
                outputs=(
                    AnswerOutputWrite(
                        answer_output_id=f"{run_id}:answer-output",
                        run_id=run_id,
                        answer_id=f"{run_id}:answer",
                        fact_result_id=f"{run_id}:fact-result",
                        output_key="answer_1",
                        value_kind=AnswerValueKind.NUMBER,
                        value_json=result_data,
                        proof_node_refs_json=[f"{run_id}:answer-node"],
                    ),
                ),
                presentations=(
                    AnswerPresentationWrite(
                        presentation_id=f"{run_id}:presentation",
                        run_id=run_id,
                        answer_id=f"{run_id}:answer",
                        presentation_kind=PresentationKind.TEXT,
                        render_step_id=f"{run_id}:render",
                        client_key=PresentationClientKey.DEFAULT,
                        rendered_value=answer,
                    ),
                ),
            )
        )

    return write_terminal_answer
