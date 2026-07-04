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
from fervis.lookup.fact_plan.values import FactValue
from fervis.lookup.grounding.model import (
    CanonicalInputLedger,
    GroundedValueCertification,
    GroundedValueCertificationMethod,
    GroundingRequest,
)
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lookup.grounding.turn import GroundingTurnResult, generate_grounding
from fervis.lookup.lineage.source_reads import SourceReadLineageScope
from fervis.lookup.source_reads.response import SourceReadFailedError
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.fact_plan.row_sources import build_row_source_catalog
from fervis.lookup.question_contract import (
    KnownInputSource,
    QuestionContract,
    RequestedFactKnownInput,
)
from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole
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
    _reference_known_input_bindings,
    _resolve_reference_tasks,
)
from .values import _dedupe_values, _grounded_value_id


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
    certifications = []
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
    certifications.extend(deterministic.certifications)
    time_tasks = time_resolution_tasks(question_contract)

    reference_tasks = _reference_binding_tasks(
        question_contract,
        resolver_row_sources=resolver_row_sources,
    )
    overlay_imports = _overlay_canonical_identity_imports(
        question_contract=question_contract,
        conversation_resolution_overlay=conversation_resolution_overlay,
        active_memory_ids=active_memory_ids or frozenset(),
    )
    values.extend(overlay_imports.values)
    certifications.extend(overlay_imports.certifications)
    imported_known_input_ids = frozenset(
        _known_input_id_from_proof_ref(proof_ref)
        for value in overlay_imports.values
        for proof_ref in value.proof_refs
        if proof_ref.startswith("known_input:")
    )
    if imported_known_input_ids:
        reference_tasks = tuple(
            task
            for task in reference_tasks
            if task.known_input_id not in imported_known_input_ids
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
    certifications.extend(resolved_references.certifications)
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
        certifications.extend(time_ledger.certifications)
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
    certifications.extend(reference_ledger.certifications)

    return GroundingOutput(
        ledger=CanonicalInputLedger(
            values=_dedupe_values(tuple(values)),
            uses=tuple(uses),
            issues=tuple(issues),
            certifications=tuple(certifications),
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


def _overlay_canonical_identity_imports(
    *,
    question_contract: QuestionContract,
    conversation_resolution_overlay: ConversationResolutionOverlay | None,
    active_memory_ids: frozenset[str],
) -> CanonicalInputLedger:
    if conversation_resolution_overlay is None:
        return CanonicalInputLedger()
    effective_active_memory_ids = active_memory_ids or frozenset(
        conversation_resolution_overlay.activated_memory_ids
    )
    overlay_by_ref = {
        item.resolved_input_ref: item
        for item in conversation_resolution_overlay.resolved_question_inputs
        if item.kind == KnownInputKind.LITERAL
        and item.resolved_input_ref
        and item.resolved_canonical_value is not None
    }
    if not overlay_by_ref:
        return CanonicalInputLedger()
    values: list[FactValue] = []
    certifications: list[GroundedValueCertification] = []
    for known, requested_fact_ids in _reference_known_input_bindings(question_contract):
        if not _can_import_overlay_identity(known):
            continue
        overlay = overlay_by_ref.get(known.resolved_input_ref)
        if overlay is None:
            continue
        if (
            known.text != overlay.source_text
            or known.resolved_value_text != overlay.resolved_value_text
            or known.role != overlay.role
        ):
            continue
        canonical = overlay.resolved_canonical_value
        if canonical is None or canonical.kind != "identity":
            continue
        evidence_refs = tuple(
            ref for ref in overlay.evidence_refs if ref in effective_active_memory_ids
        )
        if not evidence_refs:
            continue
        proof_refs = (
            f"known_input:{known.id}",
            f"resolved_question_input:{overlay.resolved_input_ref}",
            *canonical.proof_refs,
        )
        value = FactValue.identity(
            id=_grounded_value_id(known.id),
            identity_type=canonical.identity_type,
            identity_field=canonical.identity_field,
            value=canonical.value,
            display_value=known.resolved_value_text,
            proof_refs=proof_refs,
            applies_to_requested_fact_ids=requested_fact_ids,
        )
        values.append(value)
        certifications.append(
            GroundedValueCertification(
                value_id=value.id,
                method=GroundedValueCertificationMethod.IMPORTED_PRIOR_IDENTITY,
                authority_refs=canonical.proof_refs,
                lineage_refs=(
                    f"known_input:{known.id}",
                    f"resolved_question_input:{overlay.resolved_input_ref}",
                    *(
                        f"conversation_resolution_evidence:{ref}"
                        for ref in evidence_refs
                    ),
                ),
            )
        )
    return CanonicalInputLedger(
        values=tuple(values),
        certifications=tuple(certifications),
    )


def _can_import_overlay_identity(known: RequestedFactKnownInput) -> bool:
    return (
        known.kind == KnownInputKind.LITERAL
        and known.source == KnownInputSource.CONVERSATION_RESOLUTION
        and known.role == LiteralInputRole.REFERENCE_VALUE
        and bool(known.resolved_input_ref)
    )


def _known_input_id_from_proof_ref(proof_ref: str) -> str:
    return proof_ref.removeprefix("known_input:")


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
