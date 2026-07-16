"""Bounded named-input resolver-to-target edges for read eligibility."""

from __future__ import annotations

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
    resolver_options = tuple(
        (option, compatible_bindings_by_option_id[option.id])
        for task in binding_tasks
        if _task_applies(task, requested_fact_id=requested_fact_id)
        for option in task.options
        if option.id in compatible_bindings_by_option_id
        if option.known_input_id in known_input_tokens_by_id
    )
    output: list[CanonicalInputOption] = []
    for resolver_option, compatible_binding in resolver_options:
        candidate = resolver_option.candidate
        known_input_token = known_input_tokens_by_id[resolver_option.known_input_id]
        output.append(
            CanonicalInputOption(
                id=_canonical_option_id(
                    known_input_token=known_input_token,
                    authority=resolver_option.id,
                    resolver_read_id=candidate.resolver_read_id,
                ),
                requested_fact_id=requested_fact_id,
                known_input_id=resolver_option.known_input_id,
                known_input_token=known_input_token,
                entity_kind=candidate.entity_kind,
                key_id=candidate.key_id,
                component_ids=tuple(
                    component.component_id for component in candidate.key_components
                ),
                resolver_binding=compatible_binding,
            )
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
        output.append(
            CanonicalInputOption(
                id=_canonical_option_id(
                    known_input_token=known_input_token,
                    authority=value.id,
                    resolver_read_id="canonical",
                ),
                requested_fact_id=requested_fact_id,
                known_input_id=value.known_input_id,
                known_input_token=known_input_token,
                entity_kind=payload.entity_kind,
                key_id=payload.key_id,
                component_ids=tuple(
                    component.component_id for component in payload.key.components
                ),
                canonical_value_id=value.id,
            )
        )
    return tuple(output)


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
    authority: str,
    resolver_read_id: str,
) -> str:
    return ".".join(
        (
            known_input_token,
            _identifier(resolver_read_id),
            _identifier(authority),
        )
    )


def _identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_") or "value"
