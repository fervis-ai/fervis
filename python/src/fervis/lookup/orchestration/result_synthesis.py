"""Lookup result synthesis from fact outcomes and deterministic rendering."""

from typing import Any

from fervis.lookup.errors import ErrorCode
from fervis.lookup.memory.outcomes import (
    fact_result_answer_addresses,
    fact_result_outcome_address,
)
from fervis.lookup.outcomes.model import FactResult
from fervis.lookup.answer_rendering import (
    RenderedFact,
    render_fact_result,
    rendered_fact_text,
)
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupRuntimePorts,
)
from fervis.lookup.orchestration.result import LookupResult
from fervis.lookup.lineage.steps import record_render_step
from fervis.lookup.lineage.results import (
    LineagePersistenceUnavailable,
    RuntimeErrorTerminal,
    record_lookup_result_lineage,
    runtime_error_terminal_result,
)


def _synthesize_result(
    *,
    request: LookupRequest,
    ports: LookupRuntimePorts,
    fact_result: FactResult,
    status: str,
    usage: dict[str, Any],
    question_contract: Any = None,
    grounded_values: tuple[Any, ...] = (),
    extra_fact_addresses: tuple[Any, ...] = (),
    known_input_step_id: str | None = None,
    question_contract_step_id: str | None = None,
    compile_step_id: str | None = None,
    execute_step_id: str | None = None,
    proof_graph: Any = None,
    answer_plan: Any = None,
    proof_node_refs_by_render_output_id: dict[str, tuple[str, ...]] | None = None,
    conversation_resolution_activation: dict[str, Any] | None = None,
) -> LookupResult:
    rendered = render_fact_result(fact_result)
    deterministic_answer = rendered_fact_text(rendered)
    render_step = None
    try:
        render_step = record_render_step(
            ports,
            kind=rendered.kind.value,
            row_count=len(rendered.rows),
        )
        record_lookup_result_lineage(
            request=request,
            ports=ports,
            fact_result=fact_result,
            rendered=rendered,
            answer=deterministic_answer,
            question_contract=question_contract,
            question_contract_step_id=question_contract_step_id,
            compile_step_id=compile_step_id,
            execute_step_id=execute_step_id,
            render_step_id=render_step.step_id if render_step is not None else None,
            proof_graph=proof_graph,
            answer_plan=answer_plan,
            proof_node_refs_by_render_output_id=(
                proof_node_refs_by_render_output_id or {}
            ),
            grounded_values=grounded_values,
            extra_fact_addresses=extra_fact_addresses,
            known_input_step_id=known_input_step_id,
            conversation_resolution_activation=conversation_resolution_activation,
        )
    except LineagePersistenceUnavailable:
        return _lineage_failure_result(
            request=request,
            ports=ports,
            render_step_id=render_step.step_id if render_step is not None else None,
            message=ErrorCode.LINEAGE_PERSISTENCE_FAILED,
            usage=usage,
        )
    except Exception as exc:
        return _lineage_failure_result(
            request=request,
            ports=ports,
            render_step_id=render_step.step_id if render_step is not None else None,
            message=str(exc),
            usage=usage,
        )
    return _rendered_lookup_result(
        request=request,
        ports=ports,
        fact_result=fact_result,
        rendered=rendered,
        status=status,
        usage=usage,
        answer=deterministic_answer,
        question_contract=question_contract,
        grounded_values=grounded_values,
        extra_fact_addresses=extra_fact_addresses,
    )


def _lineage_failure_result(
    *,
    request: LookupRequest,
    ports: LookupRuntimePorts,
    render_step_id: str | None,
    message: str,
    usage: dict[str, Any],
) -> LookupResult:
    sink = ports.lineage_step_sink
    return runtime_error_terminal_result(
        RuntimeErrorTerminal(
            run_id=request.run_id,
            failed_step_id=render_step_id,
            error_code=ErrorCode.LINEAGE_PERSISTENCE_FAILED,
            message=message,
            usage=usage,
        ),
        recorder=sink.recorder if sink is not None else None,
        lineage_required=getattr(ports, "lineage_required", False),
    )


def _rendered_lookup_result(
    *,
    request: LookupRequest,
    ports: LookupRuntimePorts,
    fact_result: FactResult,
    rendered: RenderedFact,
    status: str,
    usage: dict[str, Any],
    answer: str,
    question_contract: Any = None,
    grounded_values: tuple[Any, ...] = (),
    extra_fact_addresses: tuple[Any, ...] = (),
) -> LookupResult:
    answer_addresses = fact_result_answer_addresses(
        fact_result,
        question_contract=question_contract,
        grounded_values=grounded_values,
    )
    return LookupResult(
        status=status,
        answer=answer,
        fact_result=fact_result,
        rendered_fact=rendered,
        fact_addresses=tuple(
            item.to_dict() for item in (*extra_fact_addresses, *answer_addresses)
        ),
        fact_outcome_addresses=_fact_outcome_addresses(fact_result),
        usage=usage,
    )


def _fact_outcome_addresses(fact_result: FactResult) -> tuple[dict[str, Any], ...]:
    outcome_address = fact_result_outcome_address(fact_result)
    return (outcome_address.to_dict(),) if outcome_address is not None else ()
