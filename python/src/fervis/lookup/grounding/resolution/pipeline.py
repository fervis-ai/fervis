"""Canonicalize question-contract known inputs before fact planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import (
    EntityTargetResolverSelection,
)
from fervis.lookup.conversation_resolution.overlay import (
    ConversationResolutionOverlay,
    LiteralQuestionInputOverlay,
    ResolvedCanonicalIdentityOverlay,
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


@dataclass(frozen=True)
class _OverlayIdentityImports:
    ledger: CanonicalInputLedger = CanonicalInputLedger()
    known_input_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _OverlayIdentityImport:
    known_input_id: str
    value: FactValue
    certification: GroundedValueCertification


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
    current_active_memory_ids = (
        active_memory_ids if active_memory_ids is not None else frozenset()
    )

    active_time_anchor_periods = _active_time_anchor_periods(
        conversation_context=dict(conversation_context or {}),
        active_memory_ids=current_active_memory_ids,
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
    )
    values.extend(overlay_imports.ledger.values)
    certifications.extend(overlay_imports.ledger.certifications)
    if overlay_imports.known_input_ids:
        reference_tasks = tuple(
            task
            for task in reference_tasks
            if task.known_input_id not in overlay_imports.known_input_ids
        )
    try:
        resolved_references = _resolve_reference_tasks(
            reference_tasks,
            full_catalog=full_catalog,
            resolver_row_sources=resolver_row_sources,
            data_access_port=data_access_port,
            source_read_lineage=source_read_lineage,
        )
    except SourceReadFailedError as exc:
        raise GroundingSourceReadError(
            source_read_error=exc,
            turn=None,
        ) from exc
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


def _overlay_canonical_identity_imports(
    *,
    question_contract: QuestionContract,
    conversation_resolution_overlay: ConversationResolutionOverlay | None,
) -> _OverlayIdentityImports:
    if conversation_resolution_overlay is None:
        return _OverlayIdentityImports()

    overlay_by_ref = _literal_identity_overlays_by_ref(conversation_resolution_overlay)
    imports: list[_OverlayIdentityImport] = []
    for known, requested_fact_ids in _reference_known_input_bindings(question_contract):
        imported = _overlay_identity_import(
            known=known,
            requested_fact_ids=requested_fact_ids,
            overlay_by_ref=overlay_by_ref,
        )
        if imported is not None:
            imports.append(imported)

    return _overlay_identity_imports_ledger(tuple(imports))


def _literal_identity_overlays_by_ref(
    overlay: ConversationResolutionOverlay,
) -> dict[str, LiteralQuestionInputOverlay]:
    output: dict[str, LiteralQuestionInputOverlay] = {}
    for item in overlay.resolved_question_inputs:
        if (
            item.kind == KnownInputKind.LITERAL
            and item.resolved_input_ref
            and item.resolved_canonical_identity is not None
        ):
            output[item.resolved_input_ref] = item
    return output


def _overlay_identity_import(
    *,
    known: RequestedFactKnownInput,
    requested_fact_ids: tuple[str, ...],
    overlay_by_ref: dict[str, LiteralQuestionInputOverlay],
) -> _OverlayIdentityImport | None:
    if not _can_import_overlay_identity(known):
        return None
    overlay = overlay_by_ref.get(known.resolved_input_ref)
    if overlay is None or not _overlay_identity_matches_known(known, overlay):
        return None
    canonical = overlay.resolved_canonical_identity
    if canonical is None:
        return None
    value = _overlay_identity_value(
        known=known,
        overlay=overlay,
        canonical=canonical,
        requested_fact_ids=requested_fact_ids,
    )
    return _OverlayIdentityImport(
        known_input_id=known.id,
        value=value,
        certification=_overlay_identity_certification(
            known=known,
            overlay=overlay,
            canonical=canonical,
            value=value,
        ),
    )


def _overlay_identity_matches_known(
    known: RequestedFactKnownInput,
    overlay: LiteralQuestionInputOverlay,
) -> bool:
    return (
        known.text == overlay.source_text
        and known.occurrence == overlay.occurrence
        and known.resolved_value_text == overlay.resolved_value_text
        and known.field_label_text == overlay.field_label_text
        and known.value_meaning_hint == overlay.value_meaning_hint
        and known.role == overlay.role
    )


def _overlay_identity_value(
    *,
    known: RequestedFactKnownInput,
    overlay: LiteralQuestionInputOverlay,
    canonical: ResolvedCanonicalIdentityOverlay,
    requested_fact_ids: tuple[str, ...],
) -> FactValue:
    return FactValue.identity(
        id=_grounded_value_id(known.id),
        identity_type=canonical.identity_type,
        identity_field=canonical.identity_field,
        value=canonical.value,
        display_value=known.resolved_value_text,
        proof_refs=(
            f"known_input:{known.id}",
            f"resolved_question_input:{overlay.resolved_input_ref}",
            *canonical.authority_refs,
        ),
        applies_to_requested_fact_ids=requested_fact_ids,
    )


def _overlay_identity_certification(
    *,
    known: RequestedFactKnownInput,
    overlay: LiteralQuestionInputOverlay,
    canonical: ResolvedCanonicalIdentityOverlay,
    value: FactValue,
) -> GroundedValueCertification:
    return GroundedValueCertification(
        value_id=value.id,
        method=GroundedValueCertificationMethod.IMPORTED_PRIOR_IDENTITY,
        authority_refs=canonical.authority_refs,
        lineage_refs=(
            f"known_input:{known.id}",
            f"resolved_question_input:{overlay.resolved_input_ref}",
            *canonical.lineage_refs,
        ),
    )


def _overlay_identity_imports_ledger(
    imports: tuple[_OverlayIdentityImport, ...],
) -> _OverlayIdentityImports:
    return _OverlayIdentityImports(
        ledger=CanonicalInputLedger(
            values=tuple(item.value for item in imports),
            certifications=tuple(item.certification for item in imports),
        ),
        known_input_ids=frozenset(item.known_input_id for item in imports),
    )


def _can_import_overlay_identity(known: RequestedFactKnownInput) -> bool:
    return (
        known.kind == KnownInputKind.LITERAL
        and known.source == KnownInputSource.CONVERSATION_RESOLUTION
        and known.role == LiteralInputRole.REFERENCE_VALUE
        and bool(known.resolved_input_ref)
    )


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
