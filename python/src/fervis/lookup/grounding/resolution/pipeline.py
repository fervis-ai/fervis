"""Canonicalize question-contract known inputs before fact planning."""

from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import (
    EntityTargetResolverSelection,
)
from fervis.lookup.conversation_resolution.overlay import (
    ConversationResolutionOverlay,
)
from fervis.lookup.grounding.model import (
    CanonicalInputLedger,
    GroundingRequest,
)
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lookup.grounding.turn import GroundingTurnResult, generate_grounding
from fervis.lookup.lineage.source_reads import SourceReadLineageScope
from fervis.lookup.source_reads.response import SourceReadFailedError
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.fact_plan.row_sources import build_row_source_catalog
from fervis.lookup.question_contract import QuestionContract
from fervis.memory.identities import project_memory_identity_values
from fervis.memory.projection import fact_artifacts_from_context

from .deterministic import (
    _deterministic_known_inputs,
    resolve_time_resolutions,
    time_anchor_period_from_memory_address,
    time_resolution_tasks,
)
from .model import GroundingOutput
from .references import (
    _execute_reference_compatibilities,
    _reference_binding_tasks,
    _resolve_reference_tasks,
)
from .values import _dedupe_values


class GroundingSourceReadError(RuntimeError):
    def __init__(
        self,
        *,
        source_read_error: SourceReadFailedError,
        turn: GroundingTurnResult | None,
    ) -> None:
        self.source_read_error = source_read_error
        self.turn = turn
        self.usage = getattr(turn, "usage", {}) if turn is not None else {}
        super().__init__(str(source_read_error))


def ground_question_inputs(
    *,
    question: str,
    question_contract: QuestionContract,
    full_catalog: RelationCatalog,
    resolver_catalog: RelationCatalog,
    data_access_port: Any,
    runtime_values: RuntimeValueContext | None,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
    resolver_selections: tuple[EntityTargetResolverSelection, ...] = (),
    conversation_context: dict[str, Any] | None = None,
    active_memory_ids: frozenset[str] | None = None,
    conversation_resolution_overlay: ConversationResolutionOverlay | None = None,
    source_read_lineage: SourceReadLineageScope | None = None,
    host: HostPromptContext = HostPromptContext(),
) -> GroundingOutput:
    values = []
    uses = []
    issues = []
    resolver_row_sources = build_row_source_catalog(resolver_catalog)

    active_time_anchor_periods = _active_time_anchor_periods(
        conversation_context=dict(conversation_context or {}),
        active_memory_ids=active_memory_ids or frozenset(),
    )
    deterministic = _deterministic_known_inputs(
        question_contract,
        runtime_values=runtime_values,
        active_time_anchor_periods=active_time_anchor_periods,
    )
    values.extend(deterministic.values)
    issues.extend(deterministic.issues)
    time_tasks = time_resolution_tasks(question_contract)

    reference_tasks = _reference_binding_tasks(
        question_contract,
        resolver_row_sources=resolver_row_sources,
    )
    memory_identity_projection = project_memory_identity_values(
        fact_artifacts_from_context(dict(conversation_context or {}))
    )
    resolved_references = _resolve_reference_tasks(
        reference_tasks,
        full_catalog=full_catalog,
        resolver_row_sources=resolver_row_sources,
        data_access_port=data_access_port,
        memory_identity_values=_active_memory_identity_values(
            memory_identity_projection.identity_values,
            active_memory_ids=active_memory_ids or frozenset(),
        ),
    )
    values.extend(resolved_references.values)
    issues.extend(resolved_references.issues)
    turn: GroundingTurnResult | None = None
    compatibilities = []
    model_tasks = resolved_references.model_tasks
    if model_tasks or time_tasks:
        turn = generate_grounding(
            request=GroundingRequest(
                question=question,
                tasks=model_tasks,
                time_tasks=time_tasks,
                conversation_context=dict(conversation_context or {}),
                conversation_resolution_overlay=conversation_resolution_overlay,
                host=host,
            ),
            model_port=model_port,
            provider=provider,
            model_key=model_key,
            max_thinking_tokens=max_thinking_tokens,
        )
        time_ledger = resolve_time_resolutions(
            turn.result.time_resolutions,
            question_contract=question_contract,
            runtime_values=runtime_values,
            active_time_anchor_periods=active_time_anchor_periods,
        )
        values.extend(time_ledger.values)
        uses.extend(time_ledger.uses)
        issues.extend(time_ledger.issues)
        compatibilities.extend(turn.result.compatibilities)
    try:
        reference_ledger = _execute_reference_compatibilities(
            compatibilities=tuple(compatibilities),
            tasks=model_tasks,
            resolver_selections=resolver_selections,
            full_catalog=full_catalog,
            resolver_row_sources=resolver_row_sources,
            data_access_port=data_access_port,
            source_read_lineage=source_read_lineage,
        )
    except SourceReadFailedError as exc:
        raise GroundingSourceReadError(
            source_read_error=exc,
            turn=turn,
        ) from exc
    values.extend(reference_ledger.values)
    uses.extend(reference_ledger.uses)
    issues.extend(reference_ledger.issues)

    return GroundingOutput(
        ledger=CanonicalInputLedger(
            values=_dedupe_values(tuple(values)),
            uses=tuple(uses),
            issues=tuple(issues),
        ),
        turn=turn,
    )


def _active_memory_identity_values(
    identity_values: tuple[Any, ...],
    *,
    active_memory_ids: frozenset[str],
) -> tuple[Any, ...]:
    if not active_memory_ids:
        return ()
    return tuple(
        identity
        for identity in identity_values
        if _memory_identity_id(identity) in active_memory_ids
    )


def _memory_identity_id(identity: Any) -> str:
    source = getattr(identity, "source", {}) or {}
    artifact_id = str(source.get("artifact_id") or "")
    address = str(source.get("address") or "")
    if not artifact_id or not address:
        return ""
    return f"{artifact_id}.{address}"


def _active_time_anchor_periods(
    *,
    conversation_context: dict[str, Any],
    active_memory_ids: frozenset[str],
) -> dict[str, dict[str, str]]:
    if not active_memory_ids:
        return {}
    output: dict[str, dict[str, str]] = {}
    for artifact in fact_artifacts_from_context(conversation_context):
        for address in artifact.addresses:
            memory_id = f"{artifact.artifact_id}.{address.address}"
            if memory_id not in active_memory_ids:
                continue
            anchor_period = time_anchor_period_from_memory_address(address)
            if anchor_period is not None:
                output[memory_id] = anchor_period
    return output
