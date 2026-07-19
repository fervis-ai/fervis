"""Bounded named-input resolver-to-target edges for read eligibility."""

from __future__ import annotations

from dataclasses import replace
import re

from fervis.lookup.answer_program.values import FactValue, IdentityValuePayload
from fervis.lookup.grounding.model import CompatibleInputBinding, KnownInputBindingTask
from fervis.lookup.read_eligibility.model import CanonicalInputOption


def canonical_input_options(
    *,
    requested_fact_id: str,
    binding_tasks: tuple[KnownInputBindingTask, ...],
    compatible_reference_bindings: tuple[CompatibleInputBinding, ...],
    known_input_tokens_by_id: dict[str, str],
    canonical_values: tuple[FactValue, ...] = (),
) -> tuple[CanonicalInputOption, ...]:
    """Return every fact-local canonical meaning supplied by grounding or memory."""

    compatible_bindings_by_option_id = {
        binding.option_id: binding for binding in compatible_reference_bindings
    }
    options_by_meaning: dict[
        tuple[str, str, str, tuple[str, ...]],
        CanonicalInputOption,
    ] = {}
    for task in binding_tasks:
        if not _task_applies(task, requested_fact_id=requested_fact_id):
            continue
        for resolver_option in task.options:
            compatible_binding = compatible_bindings_by_option_id.get(
                resolver_option.id
            )
            if (
                compatible_binding is None
                or resolver_option.known_input_id not in known_input_tokens_by_id
            ):
                continue
            candidate = resolver_option.candidate
            component_ids = tuple(
                component.component_id for component in candidate.key_components
            )
            meaning_key = (
                resolver_option.known_input_id,
                candidate.entity_kind,
                candidate.key_id,
                component_ids,
            )
            existing = options_by_meaning.get(meaning_key)
            if existing is not None:
                options_by_meaning[meaning_key] = replace(
                    existing,
                    resolver_bindings=(
                        *existing.resolver_bindings,
                        compatible_binding,
                    ),
                )
                continue
            known_input_token = known_input_tokens_by_id[
                resolver_option.known_input_id
            ]
            options_by_meaning[meaning_key] = CanonicalInputOption(
                id=_canonical_option_id(
                    known_input_token=known_input_token,
                    entity_kind=candidate.entity_kind,
                    key_id=candidate.key_id,
                    component_ids=component_ids,
                ),
                requested_fact_id=requested_fact_id,
                known_input_id=resolver_option.known_input_id,
                known_input_token=known_input_token,
                entity_kind=candidate.entity_kind,
                key_id=candidate.key_id,
                component_ids=component_ids,
                resolver_bindings=(compatible_binding,),
            )
    for value in canonical_values:
        payload = value.payload
        if (
            not value.known_input_id
            or value.known_input_id not in known_input_tokens_by_id
            or not isinstance(payload, IdentityValuePayload)
            or not _value_applies(value, requested_fact_id=requested_fact_id)
        ):
            continue
        known_input_token = known_input_tokens_by_id[value.known_input_id]
        component_ids = tuple(
            component.component_id for component in payload.key.components
        )
        meaning_key = (
            value.known_input_id,
            payload.entity_kind,
            payload.key_id,
            component_ids,
        )
        existing = options_by_meaning.get(meaning_key)
        if existing is not None and existing.canonical_value_id not in {"", value.id}:
            raise ValueError("canonical input meaning has several certified values")
        options_by_meaning[meaning_key] = CanonicalInputOption(
            id=_canonical_option_id(
                known_input_token=known_input_token,
                entity_kind=payload.entity_kind,
                key_id=payload.key_id,
                component_ids=component_ids,
            ),
            requested_fact_id=requested_fact_id,
            known_input_id=value.known_input_id,
            known_input_token=known_input_token,
            entity_kind=payload.entity_kind,
            key_id=payload.key_id,
            component_ids=component_ids,
            canonical_value_id=value.id,
        )
    return tuple(options_by_meaning.values())


def interpretation_question(*, known_input_text: str, answer_fact: str) -> str:
    return (
        "Which shown canonical result represents what "
        f"{known_input_text} denotes in requested fact {answer_fact}?"
    )


def _value_applies(value: FactValue, *, requested_fact_id: str) -> bool:
    return (
        not value.applies_to_requested_fact_ids
        or requested_fact_id in value.applies_to_requested_fact_ids
    )


def _task_applies(
    task: KnownInputBindingTask,
    *,
    requested_fact_id: str,
) -> bool:
    applicable = task.applies_to_requested_fact_ids
    return (
        requested_fact_id == task.requested_fact_id
        or not applicable
        or requested_fact_id in applicable
    )


def _canonical_option_id(
    *,
    known_input_token: str,
    entity_kind: str,
    key_id: str,
    component_ids: tuple[str, ...],
) -> str:
    return ".".join(
        (
            known_input_token,
            _identifier(entity_kind),
            _identifier(key_id),
            _identifier("_".join(component_ids)),
        )
    )


def _identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_") or "value"
