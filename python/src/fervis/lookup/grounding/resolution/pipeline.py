"""Canonicalize question-contract known inputs before fact planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
    ResolvedCanonicalIdentity,
    ResolvedIdentityInput,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.grounding.model import (
    CanonicalInputLedger,
    GroundedInputUse,
    GroundedValueCertification,
    GroundedValueCertificationMethod,
    GroundingIssue,
    GroundingRequest,
    ExpectedInputIdentity,
    InputBindingSelection,
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
    RequestedFactLiteralInput,
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
    _execute_reference_selections,
    _reference_known_input_bindings,
    _resolve_reference_tasks,
    reference_binding_row_sources,
    reference_input_binding_tasks,
)
from .values import _dedupe_values, _grounded_value_id
from fervis.lookup.clarification.model import (
    GroundingIdentityResponse,
    GroundingTextResponse,
)


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
class _ResolvedIdentityImports:
    ledger: CanonicalInputLedger = CanonicalInputLedger()
    known_input_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _ResolvedIdentityImport:
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
    conversation_context: dict[str, Any] | None = None,
    active_memory_ids: frozenset[str] | None = None,
    conversation_resolution: CompiledConversationResolution | None = None,
    source_read_lineage: SourceReadLineageScope | None = None,
    host: HostPromptContext = HostPromptContext(),
    selected_input_ids: frozenset[str] | None = None,
    expected_input_identities: Mapping[str, ExpectedInputIdentity] | None = None,
    clarification_response: GroundingIdentityResponse | GroundingTextResponse | None = None,
) -> GroundingOutput:
    values: list[FactValue] = []
    uses: list[GroundedInputUse] = []
    issues: list[GroundingIssue] = []
    certifications: list[GroundedValueCertification] = []
    full_row_sources = build_row_source_catalog(full_catalog)
    reference_grounding_sources = build_row_source_catalog(resolver_catalog)
    input_binding_sources = reference_binding_row_sources(
        full_row_sources=full_row_sources,
        resolver_row_sources=reference_grounding_sources,
    )
    current_active_memory_ids = (
        active_memory_ids if active_memory_ids is not None else frozenset()
    )
    clarified = _clarification_identity_imports(
        clarification_response,
        question_contract=question_contract,
    )
    values.extend(clarified.values)
    certifications.extend(clarified.certifications)
    clarified_input_ids = frozenset(
        value.known_input_id for value in clarified.values
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
    values.extend(
        value
        for value in deterministic.values
        if _selected(value.known_input_id, selected_input_ids=selected_input_ids)
    )
    issues.extend(
        issue
        for issue in deterministic.issues
        if _selected(issue.known_input_id, selected_input_ids=selected_input_ids)
    )
    certifications.extend(
        certification
        for certification in deterministic.certifications
        if any(value.id == certification.value_id for value in values)
    )
    time_tasks = tuple(
        task
        for task in time_resolution_tasks(question_contract)
        if _selected(task.known_input_id, selected_input_ids=selected_input_ids)
    )

    reference_tasks = reference_input_binding_tasks(
        question_contract,
        resolver_row_sources=input_binding_sources,
        expected_input_identities=expected_input_identities,
    )
    reference_tasks = tuple(
        task
        for task in reference_tasks
        if _selected(task.known_input_id, selected_input_ids=selected_input_ids)
        and task.known_input_id not in clarified_input_ids
    )
    identity_imports = _resolved_canonical_identity_imports(
        question_contract=question_contract,
        conversation_resolution=conversation_resolution,
    )
    if selected_input_ids is not None:
        identity_imports = _selected_identity_imports(
            identity_imports,
            selected_input_ids=selected_input_ids,
        )
    values.extend(identity_imports.ledger.values)
    certifications.extend(identity_imports.ledger.certifications)
    if identity_imports.known_input_ids:
        reference_tasks = tuple(
            task
            for task in reference_tasks
            if task.known_input_id not in identity_imports.known_input_ids
        )
    try:
        resolved_references = _resolve_reference_tasks(
            reference_tasks,
            full_catalog=full_catalog,
            resolver_row_sources=input_binding_sources,
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
    selections: list[InputBindingSelection] = []
    model_tasks = resolved_references.model_tasks
    if model_tasks or time_tasks:
        turn = generate_grounding(
            request=GroundingRequest(
                question=question,
                tasks=model_tasks,
                time_tasks=time_tasks,
                conversation_context=dict(conversation_context or {}),
                host=host,
                clarification_source=(
                    clarification_response.source
                    if isinstance(clarification_response, GroundingTextResponse)
                    else None
                ),
                clarification_known_input_id=(
                    clarification_response.known_input_id
                    if isinstance(clarification_response, GroundingTextResponse)
                    else ""
                ),
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
        selections.extend(turn.result.selections)
    try:
        reference_ledger = _execute_reference_selections(
            selections=tuple(selections),
            tasks=model_tasks,
            full_catalog=full_catalog,
            resolver_row_sources=input_binding_sources,
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


def _selected(
    known_input_id: str,
    *,
    selected_input_ids: frozenset[str] | None,
) -> bool:
    return selected_input_ids is None or known_input_id in selected_input_ids


def _clarification_identity_imports(
    answer: GroundingIdentityResponse | GroundingTextResponse | None,
    *,
    question_contract: QuestionContract,
) -> CanonicalInputLedger:
    if answer is None:
        return CanonicalInputLedger()
    if not isinstance(answer, GroundingIdentityResponse):
        return CanonicalInputLedger()
    option = answer.option
    requested_facts = {
        fact.id: {known.id for known in fact.known_inputs}
        for fact in question_contract.requested_facts
    }
    fact_inputs = requested_facts.get(answer.requested_fact_id)
    if fact_inputs is None or answer.known_input_id not in fact_inputs:
        raise ValueError("clarification subject does not match the current contract")
    if not all(
        (option.entity_kind, option.key_id, option.matched_field, option.matched_value)
    ):
        return CanonicalInputLedger()
    value = FactValue.identity(
        id=_grounded_value_id(answer.known_input_id),
        known_input_id=answer.known_input_id,
        entity_kind=option.entity_kind,
        key_id=option.key_id,
        key_component_id=option.matched_field,
        value=option.matched_value,
        display_value=option.matched_label or option.label,
        proof_refs=(
            f"clarification:{answer.clarification_id}",
            f"clarification_response:{answer.response_id}",
        ),
        source_refs=(option.resolver_read_id,) if option.resolver_read_id else (),
        applies_to_requested_fact_ids=(answer.requested_fact_id,),
    )
    certification = GroundedValueCertification(
        value_id=value.id,
        method=GroundedValueCertificationMethod.CLARIFICATION_SELECTION,
        authority_refs=(option.resolver_read_id,) if option.resolver_read_id else (),
        lineage_refs=(f"clarification_response:{answer.response_id}",),
    )
    return CanonicalInputLedger(
        values=(value,),
        certifications=(certification,),
    )


def _selected_identity_imports(
    imports: _ResolvedIdentityImports,
    *,
    selected_input_ids: frozenset[str],
) -> _ResolvedIdentityImports:
    selected_value_ids = {
        value.id
        for value in imports.ledger.values
        if value.known_input_id in selected_input_ids
    }
    return _ResolvedIdentityImports(
        ledger=CanonicalInputLedger(
            values=tuple(
                value
                for value in imports.ledger.values
                if value.id in selected_value_ids
            ),
            certifications=tuple(
                item
                for item in imports.ledger.certifications
                if item.value_id in selected_value_ids
            ),
        ),
        known_input_ids=imports.known_input_ids & selected_input_ids,
    )


def _resolved_canonical_identity_imports(
    *,
    question_contract: QuestionContract,
    conversation_resolution: CompiledConversationResolution | None,
) -> _ResolvedIdentityImports:
    if conversation_resolution is None:
        return _ResolvedIdentityImports()

    identities_by_ref = {
        item.input_ref: item for item in conversation_resolution.identity_inputs()
    }
    imports: list[_ResolvedIdentityImport] = []
    for known, requested_fact_ids in _reference_known_input_bindings(question_contract):
        imported = _resolved_identity_import(
            known=known,
            requested_fact_ids=requested_fact_ids,
            identities_by_ref=identities_by_ref,
        )
        if imported is not None:
            imports.append(imported)

    return _resolved_identity_imports_ledger(tuple(imports))


def _resolved_identity_import(
    *,
    known: RequestedFactKnownInput,
    requested_fact_ids: tuple[str, ...],
    identities_by_ref: dict[str, ResolvedIdentityInput],
) -> _ResolvedIdentityImport | None:
    match known:
        case RequestedFactLiteralInput() as literal:
            if not _can_import_resolved_identity(literal):
                return None
        case _:
            return None
    resolved = identities_by_ref.get(literal.resolved_input_ref)
    if resolved is None or not _resolved_identity_matches_known(literal, resolved):
        return None
    canonical = resolved.canonical_identity
    value = _resolved_identity_value(
        known=literal,
        resolved=resolved,
        canonical=canonical,
        requested_fact_ids=requested_fact_ids,
    )
    return _ResolvedIdentityImport(
        known_input_id=literal.id,
        value=value,
        certification=_resolved_identity_certification(
            known=literal,
            resolved=resolved,
            canonical=canonical,
            value=value,
        ),
    )


def _resolved_identity_matches_known(
    known: RequestedFactLiteralInput,
    resolved: ResolvedIdentityInput,
) -> bool:
    return (
        known.text == resolved.value_source_text
        and known.occurrence == resolved.occurrence
        and known.resolved_value_text == resolved.resolved_value_text
        and known.field_label_text == resolved.field_label_text
        and known.value_meaning_hint == resolved.value_meaning_hint
        and known.role == resolved.role
    )


def _resolved_identity_value(
    *,
    known: RequestedFactLiteralInput,
    resolved: ResolvedIdentityInput,
    canonical: ResolvedCanonicalIdentity,
    requested_fact_ids: tuple[str, ...],
) -> FactValue:
    return FactValue.identity(
        id=_grounded_value_id(known.id),
        known_input_id=known.id,
        entity_kind=canonical.entity_kind,
        key_id=canonical.key_id,
        key_component_id=canonical.key_component_id,
        value=canonical.value,
        display_value=known.resolved_value_text,
        proof_refs=(
            f"known_input:{known.id}",
            f"resolved_question_input:{resolved.input_ref}",
            *canonical.authority_refs,
        ),
        applies_to_requested_fact_ids=requested_fact_ids,
    )


def _resolved_identity_certification(
    *,
    known: RequestedFactLiteralInput,
    resolved: ResolvedIdentityInput,
    canonical: ResolvedCanonicalIdentity,
    value: FactValue,
) -> GroundedValueCertification:
    return GroundedValueCertification(
        value_id=value.id,
        method=GroundedValueCertificationMethod.IMPORTED_PRIOR_IDENTITY,
        authority_refs=canonical.authority_refs,
        lineage_refs=(
            f"known_input:{known.id}",
            f"resolved_question_input:{resolved.input_ref}",
            *canonical.lineage_refs,
        ),
    )


def _resolved_identity_imports_ledger(
    imports: tuple[_ResolvedIdentityImport, ...],
) -> _ResolvedIdentityImports:
    return _ResolvedIdentityImports(
        ledger=CanonicalInputLedger(
            values=tuple(item.value for item in imports),
            certifications=tuple(item.certification for item in imports),
        ),
        known_input_ids=frozenset(item.known_input_id for item in imports),
    )


def _can_import_resolved_identity(known: RequestedFactLiteralInput) -> bool:
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
