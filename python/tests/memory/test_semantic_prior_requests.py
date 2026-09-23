from __future__ import annotations

from dataclasses import replace

from fervis.lineage.enums import ProgramInvocationKind
from fervis.lookup.answer_program import AnswerProgram, BindingSet
from fervis.lookup.contract_codec import answer_program_id
from fervis.lookup.answer_program.persistence import (
    StoredProgramInvocation,
    program_invocation,
)
from fervis.memory.artifacts import FactOutcome, build_fact_artifact
from fervis.memory.conversation_context.semantic_frames import prior_request_frames
from tests.testkit.semantic_question_contracts import semantic_question_contract


def test_current_program_invocation_projects_one_semantic_prior_request() -> None:
    contract = semantic_question_contract(
        requested_fact_id="fact_1",
        output_ids=("output_1",),
        description="tasks",
    )
    program = AnswerProgram(
        inputs=contract.inputs,
        input_denotations=contract.input_denotations,
        fact_template=contract.requested_facts,
    )
    bindings = BindingSet()
    invocation = program_invocation(
        run_id="run_1",
        program_id=answer_program_id(program),
        bindings=bindings,
        kind=ProgramInvocationKind.COMPILED_QUESTION,
    )
    artifact = build_fact_artifact(
        artifact_id="artifact_1",
        outcome=FactOutcome.ANSWERED,
        provenance={
            "runId": "run_1",
            "requestedFactKey": "fact_1",
        },
        source_question="Which tasks?",
    )

    [prior_request] = prior_request_frames(
        (artifact,),
        invocations_by_run_id={
            "run_1": StoredProgramInvocation(invocation=invocation, program=program)
        },
    )

    assert prior_request.requested_fact_ref == "fact_1"
    assert prior_request.frame.callable is not None
    assert prior_request.frame.callable.base_invocation_id == invocation.invocation_id
    assert {part.kind.value for part in prior_request.frame.parts} >= {
        "subject",
        "requested_output",
        "selection",
    }


def test_multi_fact_invocation_does_not_expose_fact_scoped_callable_program() -> None:
    contract = semantic_question_contract(
        requested_fact_id="fact_1",
        output_ids=("output_1",),
        description="tasks",
    )
    [fact] = contract.requested_facts
    second_fact = replace(fact, id="fact_2")
    program = AnswerProgram(
        inputs=contract.inputs,
        input_denotations=contract.input_denotations,
        fact_template=(fact, second_fact),
    )
    invocation = program_invocation(
        run_id="run_1",
        program_id=answer_program_id(program),
        bindings=BindingSet(),
        kind=ProgramInvocationKind.COMPILED_QUESTION,
    )
    artifacts = tuple(
        build_fact_artifact(
            artifact_id=f"artifact_{index}",
            outcome=FactOutcome.ANSWERED,
            provenance={
                "runId": "run_1",
                "requestedFactKey": fact_id,
            },
            source_question="Which tasks?",
        )
        for index, fact_id in enumerate(("fact_1", "fact_2"), start=1)
    )

    frames = prior_request_frames(
        artifacts,
        invocations_by_run_id={
            "run_1": StoredProgramInvocation(invocation=invocation, program=program)
        },
    )

    assert len(frames) == 2
    assert all(item.frame.callable is None for item in frames)
