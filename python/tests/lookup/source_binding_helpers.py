from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

from fervis.lookup.source_binding.plan_targets import (
    source_binding_fact_field_id,
    source_binding_fact_id_from_field,
)
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lookup.source_binding import (
    SourceBindingRequest,
    SourceCandidateDiscoveryRequest,
    source_candidate_discovery_registry,
)

from tests.lookup.prompt_sections import prompt_section_payload


def source_binding_request(**kwargs: Any) -> SourceBindingRequest:
    discovery_request = SourceCandidateDiscoveryRequest(
        question=kwargs["question"],
        question_contract=kwargs["question_contract"],
        requested_facts=kwargs["requested_facts"],
        relation_catalog=kwargs["relation_catalog"],
        catalog_selection=kwargs["catalog_selection"],
        same_scope_relation_catalog=kwargs.get("same_scope_relation_catalog"),
        memory_inputs=kwargs.get("memory_inputs", {}),
        active_memory_ids=kwargs.get("active_memory_ids", ()),
        available_values=kwargs.get("available_values", ()),
        available_value_uses=kwargs.get("available_value_uses", ()),
        read_eligibility=kwargs.get("read_eligibility"),
        conversation_context=kwargs.get("conversation_context", {}),
        conversation_resolution=kwargs.get("conversation_resolution"),
        host=kwargs.get("host") or HostPromptContext(),
    )
    kwargs["source_candidates"] = source_candidate_discovery_registry(
        discovery_request
    )
    return SourceBindingRequest(**kwargs)


@dataclass(frozen=True)
class _PromptBindingTarget:
    binding_target_id: str
    requested_fact_id: str
    source_candidate_id: str
    plan_shape: str
    requirement_id: str
    answer_output_ids: tuple[str, ...]
    payload: dict[str, Any]

    @property
    def requires_answer_fulfillment(self) -> bool:
        return bool(self.answer_output_ids)


def source_fulfills_for_candidate(
    candidate: dict[str, Any],
    *,
    field_ids: tuple[str, ...],
    answer_output_ids: tuple[str, ...] = ("answer_1",),
) -> dict[str, dict[str, Any]]:
    evidence_ids = tuple(
        _candidate_evidence_id(candidate, field_id=field_id) for field_id in field_ids
    )
    evidence_text = ", ".join(evidence_ids)
    output: dict[str, dict[str, Any]] = {}
    for answer_output_id in answer_output_ids:
        support_set_id = _candidate_fulfillment_choice_id(
            candidate,
            answer_output_id=answer_output_id,
            evidence_ids=evidence_ids,
        )
        output[answer_output_id] = {
            "match_basis_explanation": (
                f"{answer_output_id} is fulfilled by {evidence_text} because "
                "the selected source evidence provides the requested output."
            ),
            "fulfillment_choice_id": support_set_id,
        }
    return output


def source_fulfills_fields_for_candidate(
    candidate: dict[str, Any],
    *,
    field_ids_by_answer_output: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for answer_output_id, field_ids in field_ids_by_answer_output.items():
        evidence_ids = tuple(
            _candidate_evidence_id(candidate, field_id=field_id)
            for field_id in field_ids
        )
        evidence_text = ", ".join(evidence_ids)
        support_set_id = _candidate_fulfillment_choice_id(
            candidate,
            answer_output_id=answer_output_id,
            evidence_ids=evidence_ids,
        )
        output[answer_output_id] = {
            "match_basis_explanation": (
                f"{answer_output_id} is fulfilled by {evidence_text} because "
                "the selected source evidence provides the requested output."
            ),
            "fulfillment_choice_id": support_set_id,
        }
    return output


def source_fulfills_keys_for_candidate(
    candidate: dict[str, Any],
    *,
    key_ids_by_answer_output: dict[str, str],
    row_path_ids_by_answer_output: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    choices = (
        _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
    )
    output: dict[str, dict[str, Any]] = {}
    row_path_ids = row_path_ids_by_answer_output or {}
    for answer_output_id, key_id in key_ids_by_answer_output.items():
        matching_choice_ids = [
            str(choice.get("fulfillment_choice_id") or "")
            for choice in choices
            if isinstance(choice, dict)
            and str(choice.get("answer_output_id") or "") == answer_output_id
            and _choice_targets_entity_key(
                choice,
                key_id=key_id,
                row_path_id=row_path_ids.get(answer_output_id, ""),
            )
        ]
        if len(matching_choice_ids) != 1:
            raise AssertionError(
                f"candidate key choice not found for {answer_output_id}:{key_id}"
            )
        choice_id = matching_choice_ids[0]
        output[answer_output_id] = {
            "match_basis_explanation": (
                f"{answer_output_id} is fulfilled by declared candidate key {key_id}."
            ),
            "fulfillment_choice_id": choice_id,
        }
    return output


def _choice_targets_entity_key(
    choice: dict[str, Any],
    *,
    key_id: str,
    row_path_id: str,
) -> bool:
    return any(
        (
            str(item.get("key_id") or item.get("target_key_id") or "") == key_id
            or str(item.get("entity_key") or "").endswith(f".{key_id}")
        )
        and (not row_path_id or str(item.get("row_path_id") or "") == row_path_id)
        for slot in choice.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
        for item in slot.get("entity_evidence") or ()
        if isinstance(item, dict)
    )


def source_fulfills_by_row_population_for_candidate(
    candidate: dict[str, Any],
    *,
    answer_output_ids: tuple[str, ...] = ("answer_1",),
    row_path_id: str = "",
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for answer_output_id in answer_output_ids:
        support_set_id = _candidate_row_population_support_set_id(
            candidate,
            answer_output_id=answer_output_id,
            row_path_id=row_path_id,
        )
        output[answer_output_id] = {
            "match_basis_explanation": (
                f"{answer_output_id} is fulfilled by the selected row "
                "population because list-row output fields are projected from "
                "that returned population."
            ),
            "fulfillment_choice_id": support_set_id,
        }
    return output


def source_candidate_answer_population(
    prompt: str,
    *,
    binding_target_id: str,
) -> dict[str, str]:
    target = _binding_target_by_id(prompt).get(binding_target_id)
    if target is None:
        raise AssertionError(f"source binding target not found: {binding_target_id}")
    return _answer_population(
        prompt,
        target.source_candidate_id,
        binding_target=target,
    )


def source_candidate_with_fields(
    prompt_or_payload: str | dict[str, Any],
    *,
    requested_fact_id: str | None = None,
    kind: str | None = None,
    read_id: str | None = None,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> dict[str, Any]:
    payload = (
        _source_candidate_prompt_payload(prompt_or_payload)
        if isinstance(prompt_or_payload, str)
        else prompt_or_payload
    )
    eligible_candidate_ids = _binding_target_source_ids_for_requested_fact(
        prompt_or_payload,
        requested_fact_id=requested_fact_id,
    )
    for candidate in _all_source_candidates(payload):
        candidate_id = _candidate_id(candidate)
        if (
            eligible_candidate_ids is not None
            and candidate_id not in eligible_candidate_ids
        ):
            continue
        if kind is not None and candidate.get("kind") != kind:
            continue
        if read_id is not None and str(candidate.get("read_id") or "") != read_id:
            continue
        field_ids = _candidate_field_ids(candidate)
        if set(required) <= field_ids and not (set(forbidden) & field_ids):
            return candidate
    raise AssertionError(
        f"prompt missing {kind or 'source'} candidate with fields {required} "
        f"and without fields {forbidden}"
    )


def _binding_target_source_ids_for_requested_fact(
    prompt_or_payload: str | dict[str, Any],
    *,
    requested_fact_id: str | None,
) -> frozenset[str] | None:
    if not requested_fact_id or not isinstance(prompt_or_payload, str):
        return None
    return frozenset(
        target.source_candidate_id
        for target in _binding_targets(prompt_or_payload)
        if target.requested_fact_id == requested_fact_id
    )


def source_candidate_with_kind(
    prompt_or_payload: str | dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    payload = (
        _source_candidate_prompt_payload(prompt_or_payload)
        if isinstance(prompt_or_payload, str)
        else prompt_or_payload
    )
    for candidate in _all_source_candidates(payload):
        if candidate.get("kind") == kind:
            return candidate
    raise AssertionError(f"prompt missing {kind} candidate")


def _candidate_id(candidate: dict[str, Any]) -> str:
    value = candidate.get("source_candidate_id")
    if not isinstance(value, str) or not value:
        raise AssertionError("source candidate requires source_candidate_id")
    return value


def _required_prompt_text(payload: dict[str, Any], *, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"prompt payload requires {key}")
    return value


def _candidate_field_ids(candidate: dict[str, Any]) -> set[str]:
    surface = _candidate_binding_surface(candidate)
    field_ids = {
        str(item.get("field_id") or item.get("id") or "")
        for field_source in (
            surface.get("evidence_items") or (),
            surface.get("fields") or (),
            candidate.get("fields") or (),
        )
        for item in field_source
        if isinstance(item, dict)
    }
    field_ids |= {
        str(field.get("field_id") or "")
        for row in candidate.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    }
    field_ids |= _candidate_fulfillment_field_ids(candidate)
    return {field_id for field_id in field_ids if field_id}


def _candidate_fulfillment_field_ids(candidate: dict[str, Any]) -> set[str]:
    return {
        field_id
        for support_set in (
            _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
        )
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
        for item in _slot_evidence_items(slot)
        if isinstance(item, dict)
        for field_id in _evidence_field_ids(item)
    }


def _candidate_row_population_support_set_id(
    candidate: dict[str, Any],
    *,
    answer_output_id: str,
    row_path_id: str,
) -> str:
    for support_set in (
        _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        if not isinstance(support_set, dict):
            continue
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        if _support_set_has_row_population_evidence(
            support_set,
            row_path_id=row_path_id,
        ):
            return str(support_set.get("fulfillment_choice_id") or "")
    raise AssertionError(f"row population support set not found for {answer_output_id}")


def _support_set_has_row_population_evidence(
    support_set: dict[str, Any],
    *,
    row_path_id: str,
) -> bool:
    return any(
        isinstance(slot, dict)
        and any(
            isinstance(item, dict)
            and (
                not row_path_id
                or str(item.get("row_path_id") or "") == row_path_id
                or f".{row_path_id}." in str(item.get("evidence_id") or "")
            )
            for item in slot.get("row_count_basis_evidence") or ()
        )
        for slot in support_set.get("fulfillment_slots") or ()
    )


def _candidate_fulfillment_choice_id(
    candidate: dict[str, Any],
    *,
    answer_output_id: str,
    evidence_ids: tuple[str, ...],
) -> str:
    expected = set(evidence_ids)
    matches: list[dict[str, Any]] = []
    for support_set in (
        _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        if not isinstance(support_set, dict):
            continue
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        support_set_evidence_ids = {
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
        }
        if expected <= support_set_evidence_ids:
            matches.append(support_set)
    if matches:
        return str(
            max(
                matches,
                key=lambda item: _support_set_match_score(
                    item, expected_evidence_ids=expected
                ),
            ).get("fulfillment_choice_id")
            or ""
        )
    raise AssertionError(
        f"fulfillment support set not found for {answer_output_id}:{sorted(expected)}"
    )


def _support_set_match_score(
    support_set: dict[str, Any],
    *,
    expected_evidence_ids: set[str],
) -> tuple[int, int, int]:
    slots = tuple(
        slot
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
    )
    expected_is_metric = any(
        item.get("evidence_id") in expected_evidence_ids
        for slot in slots
        for item in slot.get("metric_measure_evidence") or ()
        if isinstance(item, dict)
    )
    expected_is_count_basis = any(
        item.get("evidence_id") in expected_evidence_ids
        for slot in slots
        for item in slot.get("row_count_basis_evidence") or ()
        if isinstance(item, dict)
    )
    expected_is_entity = any(
        item.get("evidence_id") in expected_evidence_ids
        for slot in slots
        for item in slot.get("entity_evidence") or ()
        if isinstance(item, dict)
    )
    has_count_basis = any(slot.get("row_count_basis_evidence") for slot in slots)
    has_metric = any(slot.get("metric_measure_evidence") for slot in slots)
    has_entity = any(slot.get("entity_evidence") for slot in slots)
    if expected_is_metric:
        return (
            3,
            int(has_count_basis),
            -int(has_entity),
        )
    if expected_is_count_basis:
        return (3, int(has_count_basis), -int(has_entity))
    if expected_is_entity:
        return (2, int(has_entity), -int(has_metric))
    return (1, 0, -len(slots))


def _candidate_evidence_id(candidate: dict[str, Any], *, field_id: str) -> str:
    for support_set in (
        _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        if not isinstance(support_set, dict):
            continue
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for item in _slot_evidence_items(slot):
                if not isinstance(item, dict):
                    continue
                candidate_field_id = str(item.get("field_id") or "")
                evidence_id = str(item.get("evidence_id") or "")
                candidate_field_ids = _evidence_field_ids(item)
                if (
                    candidate_field_id == field_id
                    or candidate_field_id.endswith(f".{field_id}")
                    or field_id in candidate_field_ids
                ):
                    return evidence_id
    for item in _candidate_binding_surface(candidate).get("evidence_items") or ():
        if not isinstance(item, dict):
            continue
        candidate_field_id = str(item.get("field_id") or "")
        evidence_id = str(item.get("evidence_id") or "")
        if candidate_field_id == field_id or candidate_field_id.endswith(
            f".{field_id}"
        ):
            return evidence_id
    for item in (
        _candidate_binding_surface(candidate).get("fields")
        or candidate.get("fields")
        or candidate.get("columns")
        or ()
    ):
        if not isinstance(item, dict):
            continue
        candidate_field_id = str(item.get("field_id") or item.get("id") or "")
        if candidate_field_id == field_id or candidate_field_id.endswith(
            f".{field_id}"
        ):
            return candidate_field_id
    value_id = str(candidate.get("value_id") or "")
    return value_id or field_id


def source_binding_payload_from_fact_plan(
    fact_plan: dict[str, Any],
    *,
    prompt: str = "",
) -> dict[str, Any]:
    payload = _source_binding_payload_from_fact_plan_raw(fact_plan, prompt=prompt)
    return source_binding_payload_for_one_call(payload, prompt=prompt)


def _source_binding_payload_from_fact_plan_raw(
    fact_plan: dict[str, Any],
    *,
    prompt: str = "",
) -> dict[str, Any]:
    outcome = fact_plan.get("outcome")
    if isinstance(outcome, dict) and outcome.get("kind") != "fact_plan":
        return _first_source_binding_payload_from_prompt(prompt)
    bindings, _ = extract_source_bindings(fact_plan, prompt=prompt)
    return {
        "outcome": {
            "kind": "source_bindings",
            **_fact_binding_fields(
                bindings,
                prompt=prompt,
            ),
        }
    }


_SOURCE_BINDING_INVOCATION_FIELDS = frozenset(
    {
        "binding_target_id",
        "answer_population",
        "fulfillment_decisions",
        "param_decisions",
        "resolved_input_applications",
        "row_predicate_reviews",
        "finite_choice_param_reviews",
    }
)


def source_binding_payload_from_fact_plan_with_invocation_overrides(
    fact_plan: dict[str, Any],
    *,
    prompt: str,
    invocation_overrides: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    payload = _source_binding_payload_from_fact_plan_raw(fact_plan, prompt=prompt)
    invocations = _source_binding_invocations(payload.get("outcome", {}))
    default_decisions_by_override: list[dict[str, dict[str, str]]] = []
    for override in invocation_overrides:
        invocation = _source_binding_invocation_for_override(
            invocations,
            override=override,
            prompt=prompt,
        )
        default_decisions_by_override.append(
            _default_param_decisions(
                prompt=prompt,
                invocation=invocation,
                param_ids=tuple(override.get("use_default_param_ids") or ()),
            )
        )
        resolved_applications = resolved_input_applications_for_target(
            prompt,
            binding_target_id=str(invocation["binding_target_id"]),
            selections=tuple(override.get("resolved_input_applications") or ()),
        )
        invocation.update(
            {
                field: value
                for field, value in override.items()
                if field
                not in {
                    "binding_target_id",
                    "requested_fact_id",
                    "source_candidate_id",
                    "plan_shape",
                    "row_predicate_choices",
                    "use_default_param_ids",
                    "resolved_input_applications",
                }
            }
        )
        invocation["resolved_input_applications"] = resolved_applications
    normalized = source_binding_payload_for_one_call(payload, prompt=prompt)
    normalized_invocations = _source_binding_invocations(normalized.get("outcome", {}))
    for override, default_decisions in zip(
        invocation_overrides,
        default_decisions_by_override,
        strict=True,
    ):
        invocation = _source_binding_invocation_for_override(
            normalized_invocations,
            override=override,
            prompt=prompt,
        )
        invocation.setdefault("param_decisions", {}).update(default_decisions)
        reviews = invocation.get("finite_choice_param_reviews")
        if isinstance(reviews, dict):
            for param_id in default_decisions:
                reviews.pop(param_id, None)
        _apply_row_predicate_choices(
            invocation,
            choices=dict(override.get("row_predicate_choices") or {}),
        )
    return normalized


def _apply_row_predicate_choices(
    invocation: dict[str, Any],
    *,
    choices: dict[str, tuple[str, ...]],
) -> None:
    reviews = invocation.get("row_predicate_reviews")
    if not isinstance(reviews, dict):
        raise AssertionError("source binding invocation requires row predicate reviews")
    for field_id, included_values in choices.items():
        matching_ids = tuple(
            predicate_id
            for predicate_id in reviews
            if predicate_id.rsplit(".", 1)[-1] == field_id
        )
        if len(matching_ids) != 1:
            raise AssertionError(f"source binding row predicate not found: {field_id}")
        review = reviews[matching_ids[0]]
        for choice_review in review.get("choice_reviews") or ():
            included = str(choice_review.get("choice_option_id") or "") in set(
                included_values
            )
            for result in (choice_review.get("population_test_results") or {}).values():
                result["test_effect"] = (
                    "SATISFIES_TEST" if included else "CONFLICTS_WITH_TEST"
                )
                result["because"] = (
                    f"The conformance scenario {'includes' if included else 'excludes'} "
                    f"{field_id}={choice_review.get('choice_option_id')}."
                )


def _default_param_decisions(
    *,
    prompt: str,
    invocation: dict[str, Any],
    param_ids: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    if not param_ids:
        return {}
    target = _binding_target_for_invocation(prompt, invocation)
    options_by_param = _source_candidate_param_decision_options(prompt).get(
        target.source_candidate_id,
        {},
    )
    decisions: dict[str, dict[str, str]] = {}
    for param_id in param_ids:
        option = options_by_param.get(param_id) or {}
        if option.get("omit_decision") != "use_default":
            raise AssertionError(
                f"source binding param has no catalog default: {param_id}"
            )
        decisions[param_id] = {
            "population_intent": f"Use the declared default for {param_id}.",
            "match_basis_explanation": str(option.get("omit_meaning") or ""),
            "param_decision_id": str(option.get("non_bind_decision_id") or ""),
        }
    return decisions


def _source_binding_invocation_for_override(
    invocations: list[Any],
    *,
    override: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    if "binding_target_id" in override:
        binding_target_id = _required_fixture_text(
            override,
            key="binding_target_id",
        )
        matches = [
            invocation
            for invocation in invocations
            if isinstance(invocation, dict)
            and invocation.get("binding_target_id") == binding_target_id
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"source binding invocation not found: {binding_target_id}"
            )
        return matches[0]
    requested_fact_id = _required_fixture_text(override, key="requested_fact_id")
    source_candidate_id = _optional_fixture_text(override, key="source_candidate_id")
    matches = [
        invocation
        for invocation in invocations
        if isinstance(invocation, dict)
        for target in (_binding_target_for_invocation(prompt, invocation),)
        if target.requested_fact_id == requested_fact_id
        and (
            not source_candidate_id or target.source_candidate_id == source_candidate_id
        )
    ]
    if len(matches) != 1:
        target = (requested_fact_id, source_candidate_id or "<only-source-for-fact>")
        raise AssertionError(f"source binding invocation not found: {target}")
    return matches[0]


def _first_source_binding_payload_from_prompt(prompt: str) -> dict[str, Any]:
    payload = _source_candidate_prompt_payload(prompt)
    metric_fit = _metric_fit_contract_from_prompt(prompt)
    fitting_metric_ids_by_fact = {
        requested_fact_id: {
            evidence_id
            for evidence_id, review in reviews.items()
            if review.get("interpretation") == "FITS_REQUESTED_ANSWER"
        }
        for requested_fact_id, reviews in metric_fit[
            "fit_basis_interpretations"
        ].items()
    }
    source_invocations: list[dict[str, Any]] = []
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        candidate = _first_source_option_with_support_set(fact_sources)
        if not candidate:
            continue
        requested_fact_id = _required_prompt_text(
            fact_sources,
            key="requested_fact_id",
        )
        support_set = _first_fulfillment_support_set(
            candidate,
            fitting_metric_ids=fitting_metric_ids_by_fact.get(
                requested_fact_id,
                set(),
            ),
        )
        answer_output_id = str(support_set.get("answer_output_id") or "answer_1")
        source_candidate_id = _candidate_id(candidate)
        target = _binding_target_for_candidate(
            prompt,
            requested_fact_id=requested_fact_id,
            source_candidate_id=source_candidate_id,
        )
        source_invocations.append(
            {
                "binding_target_id": target.binding_target_id,
                "answer_population": {
                    "population_binding_id": _first_population_binding_id(candidate),
                    "intent_text": "fixture-selected source population",
                    "match_basis_explanation": (
                        "The fixture selects the first source population exposed "
                        "for the prompt."
                    ),
                    "population_test_results": (
                        satisfying_source_population_test_results(target)
                    ),
                },
                "fulfillment_decisions": {
                    answer_output_id: {
                        "match_basis_explanation": (
                            f"{answer_output_id} is fulfilled by the selected "
                            "support set."
                        ),
                        "fulfillment_choice_id": str(
                            support_set["fulfillment_choice_id"]
                        ),
                    }
                },
                "param_decisions": {},
                "resolved_input_applications": (
                    _unambiguous_resolved_input_applications(target)
                ),
                "row_predicate_reviews": {},
            }
        )
    return {
        "outcome": {
            "kind": "source_bindings",
            **_fact_binding_fields(
                source_invocations,
                prompt=prompt,
            ),
        }
    }


def satisfying_source_population_test_results(
    binding_target: dict[str, Any] | _PromptBindingTarget,
) -> dict[str, dict[str, str]]:
    target_payload = (
        binding_target.payload
        if isinstance(binding_target, _PromptBindingTarget)
        else binding_target
    )
    basis = target_payload.get("answer_population_test_basis")
    if not isinstance(basis, dict):
        return {}
    return {
        str(test_id): {
            "test_id": str(test_id),
            "test_question": str(test_basis["test_question"]),
            "role_scoped_test_question": str(
                test_basis["role_scoped_test_question"]
            ),
            "because": "The selected source population satisfies this test.",
            "test_effect": "SATISFIES_TEST",
        }
        for test_id, test_basis in basis.items()
        if isinstance(test_basis, dict)
    }


def satisfying_source_population_test_results_for_target(
    prompt: str,
    *,
    binding_target_id: str,
) -> dict[str, dict[str, str]]:
    target = _binding_target_by_id(prompt).get(binding_target_id)
    if target is None:
        raise AssertionError(f"source binding target not found: {binding_target_id}")
    return satisfying_source_population_test_results(target)


def resolved_input_applications_for_target(
    prompt: str,
    *,
    binding_target_id: str,
    selections: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not selections:
        return []
    target = _binding_target_by_id(prompt).get(binding_target_id)
    if target is None:
        raise AssertionError(f"source binding target not found: {binding_target_id}")
    application_surface = target.payload.get("resolved_input_application")
    if not isinstance(application_surface, dict):
        raise AssertionError("source binding target has no resolved-input surface")
    return [
        _resolved_input_application_from_selection(
            selection,
            application_surface=application_surface,
        )
        for selection in selections
    ]


def _resolved_input_application_from_selection(
    selection: dict[str, Any],
    *,
    application_surface: dict[str, Any],
) -> dict[str, Any]:
    selected_identity = (
        str(selection["target_kind"]),
        str(selection["target_id"]),
        str(selection["value_id"]),
        str(selection["value_component"]),
    )
    values = tuple(
        item
        for item in application_surface.get("resolved_values") or ()
        if isinstance(item, dict)
    )
    matching_values = tuple(
        value for value in values if value.get("value_id") == selected_identity[2]
    )
    targets_by_kind = application_surface.get("targets_by_kind") or {}
    if len(matching_values) != 1 or not isinstance(targets_by_kind, dict):
        raise AssertionError(
            "resolved-input application fixture must select one shown value: "
            f"{selected_identity!r}"
        )
    value = matching_values[0]
    components_by_kind = value.get("components_by_target_kind") or {}
    shown_components = components_by_kind.get(selected_identity[0]) or ()
    shown_targets = targets_by_kind.get(selected_identity[0]) or ()
    if selected_identity[1] not in shown_targets or selected_identity[3] not in shown_components:
        raise AssertionError(
            "resolved-input application fixture must select shown target and component: "
            f"{selected_identity!r}"
        )
    return {
        "target_kind": selected_identity[0],
        "target_id": selected_identity[1],
        "value_id": selected_identity[2],
        "value_component": selected_identity[3],
        "match_basis_explanation": (
            "Apply the selected resolved input to the selected source target."
        ),
        "population_test_results": _satisfying_application_test_results(value),
    }


def undecided_source_population_test_results(
    binding_target: dict[str, Any] | _PromptBindingTarget,
) -> dict[str, dict[str, str]]:
    target_payload = (
        binding_target.payload
        if isinstance(binding_target, _PromptBindingTarget)
        else binding_target
    )
    basis = target_payload.get("answer_population_test_basis")
    if not isinstance(basis, dict):
        return {}
    return {
        str(test_id): {
            "test_id": str(test_id),
            "test_question": str(test_basis["test_question"]),
            "role_scoped_test_question": str(
                test_basis["role_scoped_test_question"]
            ),
            "because": "The source population alone does not decide this test.",
            "test_effect": "DOES_NOT_DECIDE_TEST",
        }
        for test_id, test_basis in basis.items()
        if isinstance(test_basis, dict)
    }


def _first_source_option_with_support_set(
    fact_sources: dict[str, Any],
) -> dict[str, Any] | None:
    for candidate in _source_options_for_fact_sources(fact_sources):
        if _first_fulfillment_support_set(candidate):
            return candidate
    return None


def _first_fulfillment_support_set(
    candidate: dict[str, Any],
    *,
    fitting_metric_ids: set[str] | None = None,
) -> dict[str, Any]:
    support_sets = tuple(
        support_set
        for support_set in (
            _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
        )
        if isinstance(support_set, dict)
        and str(support_set.get("fulfillment_choice_id") or "")
    )
    if fitting_metric_ids:
        for support_set in support_sets:
            evidence_ids = {
                evidence_id
                for slot in support_set.get("fulfillment_slots") or ()
                if isinstance(slot, dict)
                for evidence_id in _slot_evidence_ids(slot)
            }
            if evidence_ids & fitting_metric_ids:
                return support_set
    if support_sets:
        return support_sets[0]
    return {}


def _first_population_binding_id(candidate: dict[str, Any]) -> str:
    for binding in (
        _candidate_binding_surface(candidate).get("population_bindings") or ()
    ):
        if isinstance(binding, dict) and str(
            binding.get("population_binding_id") or ""
        ):
            return str(binding["population_binding_id"])
    candidate_id = str(candidate.get("source_candidate_id") or "source")
    return f"pop.{candidate_id}.candidate_population"


def source_binding_payload_for_one_call(
    source_binding_payload: dict[str, Any],
    *,
    prompt: str,
) -> dict[str, Any]:
    outcome = source_binding_payload.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("kind") != "source_bindings":
        return source_binding_payload
    output = json.loads(json.dumps(source_binding_payload))
    finite_choice_values = _source_candidate_finite_choice_values(prompt)
    row_predicate_values = _source_candidate_row_predicate_values(prompt)
    population_roles = _source_candidate_population_roles(prompt)
    param_options = _source_candidate_param_decision_options(prompt)
    membership_tests_by_fact = _requested_fact_membership_tests(prompt)
    fulfillment_support_sets = _source_candidate_fulfillment_support_sets(prompt)
    metric_fit_contract = _metric_fit_contract_from_prompt(prompt)
    if (
        "metric_fit_bases" in output["outcome"]
        or "fit_basis_interpretations" in output["outcome"]
    ) and {
        "metric_fit_bases": output["outcome"].get("metric_fit_bases"),
        "fit_basis_interpretations": output["outcome"].get("fit_basis_interpretations"),
    } != metric_fit_contract:
        raise AssertionError(
            "source_binding_payload_for_one_call owns metric fit contract"
        )
    output["outcome"].update(metric_fit_contract)
    for invocation in _source_binding_invocations(output["outcome"]):
        if not isinstance(invocation, dict):
            continue
        unsupported = set(invocation) - _SOURCE_BINDING_INVOCATION_FIELDS
        if unsupported:
            raise AssertionError(
                "source binding fixture invocation uses unsupported fields: "
                f"{sorted(unsupported)}"
            )
        target = _binding_target_for_invocation(prompt, invocation)
        candidate_id = target.source_candidate_id
        requested_fact_id = _canonical_prompt_requested_fact_id(
            prompt,
            requested_fact_id=target.requested_fact_id,
        )
        invocation["binding_target_id"] = target.binding_target_id
        answer_population = invocation.get("answer_population")
        if not isinstance(answer_population, dict):
            raise AssertionError(
                "source binding fixture invocation requires answer_population"
            )
        answer_population.setdefault(
            "population_test_results",
            satisfying_source_population_test_results(target),
        )
        _canonicalize_invocation_fulfillment_support_sets(
            invocation,
            fulfillment_support_sets=fulfillment_support_sets.get(candidate_id, ()),
        )
        param_decisions = dict(invocation.get("param_decisions") or {})
        invocation["row_predicate_reviews"] = {
            **_row_predicate_reviews_for_candidate(
                row_predicate_values.get(candidate_id, {}),
                role=_row_predicate_review_role(
                    candidate_id,
                    population_roles.get(candidate_id, ()),
                ),
                membership_tests=membership_tests_by_fact.get(
                    requested_fact_id,
                    (),
                ),
            ),
            **dict(invocation.get("row_predicate_reviews") or {}),
        }
        if "finite_choice_param_reviews" not in invocation:
            invocation["finite_choice_param_reviews"] = {
                param_id: _choice_reviews_for_param_decision(
                    param_id=param_id,
                    choices=choices,
                    role=_first_population_role(population_roles.get(candidate_id, ())),
                    param_decision=param_decisions.pop(param_id, {}),
                    param_options=param_options.get(candidate_id, {}).get(param_id, {}),
                    membership_tests=membership_tests_by_fact.get(
                        requested_fact_id,
                        (),
                    ),
                )
                for param_id, choices in finite_choice_values.get(
                    candidate_id,
                    {},
                ).items()
            }
        invocation["param_decisions"] = param_decisions
        if "resolved_input_applications" not in invocation:
            invocation["resolved_input_applications"] = (
                _unambiguous_resolved_input_applications(target)
            )
    return output


def _source_binding_invocations(outcome: object) -> list[dict[str, Any]]:
    if not isinstance(outcome, dict):
        raise AssertionError("source binding outcome must be an object")
    return [
        invocation
        for field_id, fact_binding in outcome.items()
        if source_binding_fact_id_from_field(field_id) is not None
        if isinstance(fact_binding, dict)
        for role_id, invocation in fact_binding.items()
        if role_id != "plan_shape"
        if isinstance(invocation, dict)
    ]


def _fact_binding_fields(
    invocations: list[dict[str, Any]],
    *,
    prompt: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for invocation in invocations:
        target = _binding_target_for_invocation(prompt, invocation)
        field_id = source_binding_fact_field_id(target.requested_fact_id)
        fact_binding = output.setdefault(
            field_id,
            {"plan_shape": target.plan_shape},
        )
        if fact_binding["plan_shape"] != target.plan_shape:
            raise AssertionError("source binding fixture mixes plan shapes")
        fact_binding[target.requirement_id] = invocation
    return output


def _canonicalize_invocation_fulfillment_support_sets(
    invocation: dict[str, Any],
    *,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
) -> None:
    decisions = invocation.get("fulfillment_decisions")
    if not isinstance(decisions, dict) or not fulfillment_support_sets:
        return
    for answer_output_id, decision in decisions.items():
        if not isinstance(decision, dict):
            continue
        support_set_id = str(decision.get("fulfillment_choice_id") or "")
        replacement = _canonical_fulfillment_choice_id(
            support_set_id,
            answer_output_id=str(answer_output_id),
            fulfillment_support_sets=fulfillment_support_sets,
        )
        if replacement:
            decision["fulfillment_choice_id"] = replacement


def _canonical_fulfillment_choice_id(
    support_set_id: str,
    *,
    answer_output_id: str,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
) -> str:
    support_sets = tuple(
        support_set
        for support_set in fulfillment_support_sets
        if str(support_set.get("answer_output_id") or "") == answer_output_id
    )
    valid_ids = {
        str(support_set.get("fulfillment_choice_id") or "")
        for support_set in support_sets
    }
    if support_set_id in valid_ids:
        return support_set_id
    referenced_evidence_ids = _evidence_ids_referenced_by_text(
        support_set_id,
        fulfillment_support_sets=support_sets,
    )
    if not referenced_evidence_ids:
        return ""
    return _fulfillment_choice_id_for_evidence(
        answer_output_id=answer_output_id,
        evidence_ids=referenced_evidence_ids,
        fulfillment_support_sets=support_sets,
        source_candidate_id="",
    )


def _binding_target_for_invocation(
    prompt: str,
    invocation: dict[str, Any],
) -> _PromptBindingTarget:
    binding_target_id = _required_fixture_text(
        invocation,
        key="binding_target_id",
    )
    target = _binding_target_by_id(prompt).get(binding_target_id)
    if target is None:
        raise AssertionError(f"source binding target not found: {binding_target_id}")
    return target


def source_binding_target_id_for_candidate(
    prompt: str,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
    source_role: str = "",
    plan_shape: str = "",
    requires_fulfillment: bool | None = None,
) -> str:
    return _binding_target_id_for_candidate(
        prompt,
        requested_fact_id=requested_fact_id,
        source_candidate_id=source_candidate_id,
        source_role=source_role,
        plan_shape=plan_shape,
        requires_fulfillment=requires_fulfillment,
    )


def _binding_target_id_for_candidate(
    prompt: str,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
    source_role: str = "",
    plan_shape: str = "",
    requires_fulfillment: bool | None = None,
    allow_equivalent_targets: bool = False,
) -> str:
    target = _binding_target_for_candidate(
        prompt,
        requested_fact_id=requested_fact_id,
        source_candidate_id=source_candidate_id,
        source_role=source_role,
        plan_shape=plan_shape,
        requires_fulfillment=requires_fulfillment,
        allow_equivalent_targets=allow_equivalent_targets,
    )
    return target.binding_target_id


def _binding_target_for_candidate(
    prompt: str,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
    source_role: str = "",
    plan_shape: str = "",
    requires_fulfillment: bool | None = None,
    allow_equivalent_targets: bool = False,
) -> _PromptBindingTarget:
    matches = [
        target
        for target in _binding_targets(prompt)
        if target.requested_fact_id == requested_fact_id
        and (
            not source_candidate_id or target.source_candidate_id == source_candidate_id
        )
    ]
    if requires_fulfillment is not None:
        matches = [
            target
            for target in matches
            if target.requires_answer_fulfillment == requires_fulfillment
        ]
    if source_role:
        matches = _targets_for_source_role(matches, source_role=source_role)
    if plan_shape:
        matches = [target for target in matches if target.plan_shape == plan_shape]
    if len(matches) > 1 and allow_equivalent_targets:
        equivalent = _equivalent_binding_targets(matches)
        if equivalent:
            return equivalent
    if len(matches) != 1:
        target = (requested_fact_id, source_candidate_id or "<only-source-for-fact>")
        raise AssertionError(f"source binding target not found: {target}")
    return matches[0]


def _equivalent_binding_targets(
    targets: list[_PromptBindingTarget],
) -> _PromptBindingTarget | None:
    signatures = {_binding_target_signature(target) for target in targets}
    if len(signatures) != 1:
        return None
    return sorted(targets, key=lambda target: target.binding_target_id)[0]


def _binding_target_signature(
    target: _PromptBindingTarget,
) -> tuple[tuple[str, str], ...]:
    ignored = {"binding_target_id", "plan_selection_id", "source_strategy_id"}
    return tuple(
        sorted(
            (key, json.dumps(value, sort_keys=True))
            for key, value in target.payload.items()
            if key not in ignored
        )
    )


def _targets_for_source_role(
    targets: list[_PromptBindingTarget],
    *,
    source_role: str,
) -> list[_PromptBindingTarget]:
    if not source_role:
        return []
    role_terms = {source_role, f"{source_role}_set"}
    return [target for target in targets if target.requirement_id in role_terms]


def _binding_target_by_id(prompt: str) -> dict[str, _PromptBindingTarget]:
    return {target.binding_target_id: target for target in _binding_targets(prompt)}


def _binding_targets(prompt: str) -> tuple[_PromptBindingTarget, ...]:
    try:
        payload = _prompt_json_section(prompt, label="Binding plan families")
    except (AssertionError, ValueError):
        return ()
    return tuple(
        _prompt_binding_target(target)
        for fact in (payload.get("bindings_by_requested_fact") or {}).values()
        for shape in (fact.get("plan_shapes") or {}).values()
        for targets in (shape.get("role_targets") or {}).values()
        for target in targets
    )


def _prompt_binding_target(raw_value: Any) -> _PromptBindingTarget:
    if not isinstance(raw_value, dict):
        raise AssertionError("binding target must be an object")
    payload = dict(raw_value)
    return _PromptBindingTarget(
        binding_target_id=_required_target_text(payload, "binding_target_id"),
        requested_fact_id=_required_target_text(payload, "requested_fact_id"),
        source_candidate_id=_required_target_text(payload, "source_candidate_id"),
        plan_shape=_required_target_text(payload, "plan_shape"),
        requirement_id=_required_target_text(payload, "requirement_id"),
        answer_output_ids=_target_answer_output_ids(payload),
        payload=payload,
    )


def _required_target_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"binding target requires {key}")
    return value


def _target_answer_output_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_values = payload.get("required_answer_output_ids")
    if not isinstance(raw_values, list | tuple):
        raise AssertionError("binding target requires required_answer_output_ids")
    output: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str) or not raw_value:
            raise AssertionError("binding target answer_output_ids must be strings")
        output.append(raw_value)
    return tuple(output)


def _required_fixture_text(payload: dict[str, Any], *, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"source binding fixture requires {key}")
    return value


def _optional_fixture_text(payload: dict[str, Any], *, key: str) -> str:
    if key not in payload:
        return ""
    value = payload[key]
    if not isinstance(value, str):
        raise AssertionError(f"source binding fixture {key} must be text")
    return value


def _evidence_ids_referenced_by_text(
    text: str,
    *,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            evidence_id
            for support_set in fulfillment_support_sets
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
            if evidence_id and evidence_id in text
        )
    )


def _choice_reviews_for_param_decision(
    *,
    param_id: str,
    choices: tuple[str, ...],
    role: dict[str, str],
    param_decision: dict[str, Any],
    param_options: dict[str, Any],
    membership_tests: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    choice_set = param_decision.get("population_choice_set")
    include_values = set(choices)
    exclude_values: set[str] = set()
    if isinstance(choice_set, dict):
        include_values = {
            str(value) for value in choice_set.get("include_values") or () if str(value)
        }
        exclude_values = {
            str(value) for value in choice_set.get("exclude_values") or () if str(value)
        }
    else:
        decision_id = str(param_decision.get("param_decision_id") or "")
        bind_decision_ids = dict(param_options.get("bind_decision_ids") or {})
        matched_values = {
            value
            for value, option_decision_id in bind_decision_ids.items()
            if option_decision_id == decision_id
        }
        if matched_values:
            include_values = matched_values
            exclude_values = set(choices) - include_values
    return {
        "controlled_population_role_id": role["role_id"],
        "role_selection_basis": f"{param_id} controls {role['role_text']}.",
        "population_test_basis": _population_test_basis_for_role(
            role=role,
            membership_tests=membership_tests,
        ),
        "choice_reviews": [
            {
                "choice_option_id": choice,
                "choice_domain_meaning": f"{param_id} value {choice}",
                "choice_inclusion_basis": f"{choice} is reviewed for inclusion.",
                "choice_inclusion": (
                    "INCLUDE"
                    if choice not in exclude_values or choice in include_values
                    else "EXCLUDE"
                ),
                "population_test_results": _population_test_results_for_choice(
                    choice=choice,
                    included=choice not in exclude_values or choice in include_values,
                    role=role,
                    membership_tests=membership_tests,
                ),
            }
            for choice in choices
        ],
    }


def _row_predicate_reviews_for_candidate(
    predicate_values: dict[str, tuple[str, ...]],
    *,
    role: dict[str, str],
    membership_tests: tuple[dict[str, str], ...],
) -> dict[str, dict[str, Any]]:
    return {
        predicate_id: {
            "choice_reviews": [
                {
                    "choice_option_id": value,
                    "choice_domain_meaning": f"{predicate_id} value {value}",
                    "population_test_results": _row_predicate_test_results_for_choice(
                        choice=value,
                        included=True,
                        role=role,
                        membership_tests=membership_tests,
                    ),
                }
                for value in values
            ],
        }
        for predicate_id, values in predicate_values.items()
    }


def _row_predicate_test_results_for_choice(
    *,
    choice: str,
    included: bool,
    role: dict[str, str],
    membership_tests: tuple[dict[str, str], ...],
) -> dict[str, dict[str, Any]]:
    tests = membership_tests or (
        {
            "test_id": "subject_identity",
            "test_question": "Does this choice satisfy the requested answer population?",
        },
    )
    return {
        str(test["test_id"]): {
            "test_id": str(test["test_id"]),
            "test_question": test["test_question"],
            "role_scoped_test_question": (
                f"For {role['role_text']}, {test['test_question']}"
            ),
            "because": (
                f"{choice} is {'included in' if included else 'excluded from'} "
                "the requested answer population."
            ),
            "test_effect": "SATISFIES_TEST" if included else "CONFLICTS_WITH_TEST",
        }
        for test in tests
    }


def _row_predicate_review_role(
    candidate_id: str,
    roles: tuple[dict[str, str], ...],
) -> dict[str, str]:
    if roles:
        return roles[0]
    return {
        "role_id": f"row_predicate.{candidate_id}.rows",
        "role_text": f"{candidate_id} response rows",
    }


def _first_population_role(roles: tuple[dict[str, str], ...]) -> dict[str, str]:
    if not roles:
        raise AssertionError("source-binding test payload requires population_roles")
    return roles[0]


def _population_test_results_for_choice(
    *,
    choice: str,
    included: bool,
    role: dict[str, str],
    membership_tests: tuple[dict[str, str], ...],
) -> dict[str, dict[str, Any]]:
    tests = membership_tests or (
        {
            "test_id": "subject_identity",
            "test_question": "Does this choice satisfy the requested answer population?",
        },
    )
    results: dict[str, dict[str, Any]] = {}
    for test in tests:
        test_id = str(test["test_id"])
        result: dict[str, Any] = {
            "population_consequence": (
                f"{choice} is {'included in' if included else 'excluded from'} "
                "the requested answer population."
            ),
        }
        if test.get("kind") == "NORMAL_INSTANCE_GUARD":
            result.update(_normal_instance_guard_result(included=included))
        else:
            result["test_basis"] = (
                f"{choice} is {'included in' if included else 'excluded from'} "
                f"the requested population for {test_id}."
            )
            result["test_effect"] = (
                "SATISFIES_TEST" if included else "CONFLICTS_WITH_TEST"
            )
        results[test_id] = result
    return results


def _population_test_basis_for_role(
    *,
    role: dict[str, str],
    membership_tests: tuple[dict[str, str], ...],
) -> dict[str, dict[str, str]]:
    tests = membership_tests or (
        {
            "test_id": "subject_identity",
            "test_question": "Does this choice satisfy the requested answer population?",
        },
    )
    return {
        str(test["test_id"]): {
            "test_question": str(test["test_question"]),
            "role_scoped_test_question": (
                f"For {role['role_text']}, {test['test_question']}"
            ),
        }
        for test in tests
    }


def _normal_instance_guard_result(*, included: bool) -> dict[str, object]:
    if not included:
        return {
            "role_match_basis": (
                "The choice does not decide the normal-instance guard in this "
                "scripted source-binding payload."
            ),
            "explicit_user_override_evidence": [],
            "explicit_user_override_applies": False,
            "population_consequence": (
                "The choice does not decide the normal-instance guard."
            ),
            "disposition": {
                "matched_excluded_role": "NONE",
                "test_effect": "DOES_NOT_DECIDE_TEST",
            },
        }
    return {
        "role_match_basis": "The choice was compared to the excluded normal-instance roles.",
        "explicit_user_override_evidence": [],
        "explicit_user_override_applies": False,
        "population_consequence": ("The choice satisfies the normal-instance guard."),
        "disposition": {
            "matched_excluded_role": "NONE",
            "test_effect": "SATISFIES_TEST",
        },
    }


def plan_selection_payload_from_fact_plan(
    fact_plan: dict[str, Any],
    *,
    prompt: str = "",
) -> dict[str, Any]:
    outcome = fact_plan.get("outcome")
    if isinstance(outcome, dict) and outcome.get("kind") == "impossible":
        return _direct_source_alignment_payload_from_prompt(prompt)
    if not isinstance(outcome, dict) or outcome.get("kind") != "fact_plan":
        return _direct_source_alignment_payload_from_prompt(prompt)
    return _source_alignment_payload_from_fact_plan(fact_plan, prompt=prompt)


def _plan_selection_shape_for_pattern(pattern: str) -> str:
    if pattern == "grouped_rows":
        return "list_rows"
    return pattern


def _plan_selection_source_strategies_for_fact(
    candidate_groups: Any,
    *,
    requested_fact_id: str,
) -> tuple[dict[str, Any], ...]:
    for group in candidate_groups:
        if not isinstance(group, dict):
            continue
        if str(group.get("requested_fact_id") or "") != requested_fact_id:
            continue
        source_strategies = tuple(
            source_strategy
            for source_strategy in group.get("source_strategies") or ()
            if isinstance(source_strategy, dict)
            and str(source_strategy.get("source_strategy_id") or "")
        )
        if source_strategies:
            return source_strategies
    raise AssertionError(f"plan selection candidate missing for {requested_fact_id}")


def _plan_selection_source_strategy_for_answer(
    answer: dict[str, Any],
    *,
    plan_shape: str,
    source_strategies: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    matching_shape = tuple(
        source_strategy
        for source_strategy in source_strategies
        if str(source_strategy.get("plan_shape") or "") == plan_shape
    )
    if not matching_shape:
        raise AssertionError(f"plan selection candidate missing shape: {plan_shape}")
    answer_value_ids = _answer_value_ids(answer)
    if answer_value_ids:
        for source_strategy in matching_shape:
            member_value_ids = _source_strategy_member_values(
                source_strategy,
                key="value_id",
            )
            if answer_value_ids <= member_value_ids:
                return source_strategy
    answer_relation_ids = _answer_memory_relation_ids(answer)
    if answer_relation_ids:
        for source_strategy in matching_shape:
            if answer_relation_ids <= _source_strategy_member_relation_ids(
                source_strategy
            ):
                return source_strategy
    answer_calendar_ids = _answer_calendar_ids(answer)
    if answer_calendar_ids:
        for source_strategy in matching_shape:
            if answer_calendar_ids <= _source_strategy_member_values(
                source_strategy,
                key="calendar_id",
            ):
                return source_strategy
    answer_read_ids = _answer_read_ids(answer)
    answer_field_ids = _answer_field_ids(answer)
    answer_metric_field_id = _answer_metric_field_id(answer)
    if _answer_uses_aggregate_choice(answer):
        for source_strategy in matching_shape:
            if _aggregate_candidate_matches_answer(source_strategy, answer):
                return source_strategy
        raise AssertionError(
            "plan selection fixture has no current-contract aggregate candidate"
        )
    if answer_metric_field_id:
        for source_strategy in matching_shape:
            if answer_metric_field_id in _source_strategy_role_field_ids(
                source_strategy
            ).get("metric_measure", set()):
                return source_strategy
    if answer_read_ids:
        for source_strategy in matching_shape:
            member_read_ids = _source_strategy_member_values(
                source_strategy,
                key="read_id",
            )
            if (
                answer_read_ids <= member_read_ids
                and _source_strategy_has_member_kind(
                    source_strategy,
                    kind="same_scope_api_read",
                )
                and (
                    not answer_field_ids
                    or answer_field_ids
                    <= _source_strategy_member_field_ids(source_strategy)
                )
            ):
                return source_strategy
        for source_strategy in matching_shape:
            member_read_ids = _source_strategy_member_values(
                source_strategy,
                key="read_id",
            )
            if answer_read_ids <= member_read_ids and (
                not answer_field_ids
                or answer_field_ids
                <= _source_strategy_member_field_ids(source_strategy)
            ):
                return source_strategy
        read_matches = tuple(
            source_strategy
            for source_strategy in matching_shape
            if answer_read_ids
            <= _source_strategy_member_values(source_strategy, key="read_id")
        )
        if len(read_matches) == 1:
            return read_matches[0]
    if answer_field_ids:
        for source_strategy in matching_shape:
            if answer_field_ids <= _source_strategy_member_field_ids(source_strategy):
                return source_strategy
    raise AssertionError(
        "plan selection fixture could not prove a source strategy match for "
        f"shape={plan_shape}, reads={sorted(answer_read_ids)}, "
        f"fields={sorted(answer_field_ids)}"
    )


def _aggregate_candidate_matches_answer(
    source_strategy: dict[str, Any],
    answer: dict[str, Any],
) -> bool:
    answer_read_ids = _answer_read_ids(answer)
    if answer_read_ids and not (
        answer_read_ids
        <= _source_strategy_member_values(source_strategy, key="read_id")
    ):
        return False
    role_field_ids = _source_strategy_role_field_ids(source_strategy)
    metric_key = _aggregate_choice_metric_field_id_for_answer(answer)
    metric_fields = set(role_field_ids.get("metric_measure") or ())
    row_count_fields = set(role_field_ids.get("row_count_basis") or ())
    if metric_key and not (
        metric_key in metric_fields
        or _metric_key_matches_row_count_field(
            metric_key,
            row_count_fields=row_count_fields,
        )
    ):
        return False
    group_fields = set(role_field_ids.get("group_key") or ())
    expected_group_fields = _aggregate_choice_group_field_ids_for_answer(
        answer,
        candidate_field_ids=_source_strategy_member_field_ids(source_strategy),
    )
    if expected_group_fields and not expected_group_fields <= group_fields:
        return False
    return bool(metric_fields or row_count_fields or group_fields)


def _source_strategy_role_field_ids(
    source_strategy: dict[str, Any],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for member in source_strategy.get("source_members") or ():
        if not isinstance(member, dict):
            continue
        field_ids = _source_member_response_field_ids(member)
        for role_name in ("metric_measure", "group_key", "row_count_basis"):
            output.setdefault(role_name, set()).update(field_ids)
    return output


def _metric_key_matches_row_count_field(
    metric_key: str,
    *,
    row_count_fields: set[str],
) -> bool:
    if metric_key in row_count_fields:
        return True
    prefix = "count_records_"
    return (
        metric_key.startswith(prefix) and metric_key[len(prefix) :] in row_count_fields
    )


def _first_plan_selection_payload_from_prompt(prompt: str) -> dict[str, Any]:
    return _direct_source_alignment_payload_from_prompt(prompt)


def _direct_source_alignment_payload_from_prompt(prompt: str) -> dict[str, Any]:
    candidates = _source_alignment_candidates_from_prompt(prompt)
    return _source_alignment_payload_from_prompt(
        prompt,
        aligned_source_candidate_ids=frozenset(
            candidate["source_candidate_id"]
            for fact_candidates in candidates.values()
            for candidate in fact_candidates
        ),
    )


def _source_alignment_payload_from_prompt(
    prompt: str,
    *,
    aligned_source_candidate_ids: frozenset[str],
) -> dict[str, Any]:
    if not prompt:
        raise AssertionError("source alignment requires prompt")
    reviews: dict[str, dict[str, dict[str, str]]] = {}
    for requested_fact_id, candidates in _source_alignment_candidates_from_prompt(
        prompt
    ).items():
        fact_reviews: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            source_candidate_id = candidate["source_candidate_id"]
            if not source_candidate_id:
                continue
            aligned = source_candidate_id in aligned_source_candidate_ids
            fact_reviews[source_candidate_id] = {
                "source_candidate_id": source_candidate_id,
                "basis": (
                    "The fixture marks the shown source candidate as aligned."
                    if aligned
                    else "The fixture marks the shown source candidate as not aligned."
                ),
                "source_alignment": "DIRECT" if aligned else "NOT_ALIGNED",
            }
        if requested_fact_id:
            reviews[requested_fact_id] = fact_reviews
    return {
        "outcome": {
            "kind": "source_alignment_reviews",
            "reviews_by_requested_fact": reviews,
        }
    }


def _source_alignment_payload_from_fact_plan(
    fact_plan: dict[str, Any],
    *,
    prompt: str,
) -> dict[str, Any]:
    source_candidates_by_fact = _source_alignment_candidates_from_prompt(prompt)
    answers = tuple(
        {
            **answer,
            "requested_fact_id": _canonical_prompt_requested_fact_id(
                prompt,
                requested_fact_id=str(answer.get("requested_fact_id") or ""),
            ),
        }
        for answer in (fact_plan.get("outcome") or {}).get("answers") or ()
        if isinstance(answer, dict)
    )
    reviews: dict[str, dict[str, dict[str, str]]] = {}
    for requested_fact_id, candidates in source_candidates_by_fact.items():
        fact_answers = tuple(
            answer
            for answer in answers
            if str(answer.get("requested_fact_id") or "") == requested_fact_id
        )
        direct_ids = {
            candidate["source_candidate_id"]
            for candidate in candidates
            if any(
                _source_alignment_candidate_matches_answer(candidate, answer)
                for answer in fact_answers
            )
        }
        if not direct_ids:
            direct_ids.update(
                candidate["source_candidate_id"] for candidate in candidates
            )
        reviews[requested_fact_id] = {
            candidate["source_candidate_id"]: {
                "source_candidate_id": candidate["source_candidate_id"],
                "basis": (
                    "The fixture aligns this source with the authored fact plan."
                    if candidate["source_candidate_id"] in direct_ids
                    else "The fixture marks this source as not aligned with the authored fact plan."
                ),
                "source_alignment": (
                    "DIRECT"
                    if candidate["source_candidate_id"] in direct_ids
                    else "NOT_ALIGNED"
                ),
            }
            for candidate in candidates
        }
    if reviews:
        return {
            "outcome": {
                "kind": "source_alignment_reviews",
                "reviews_by_requested_fact": reviews,
            }
        }
    return _direct_source_alignment_payload_from_prompt(prompt)


def _source_alignment_candidates_from_prompt(
    prompt: str,
) -> dict[str, tuple[dict[str, str], ...]]:
    xml_text = _prompt_text_section(prompt, label="Source alignment reviews")
    root = ElementTree.fromstring(xml_text)
    output: dict[str, tuple[dict[str, str], ...]] = {}
    for fact in root.findall("requested_fact"):
        requested_fact_id = str(fact.attrib.get("id") or "")
        candidates: list[dict[str, str]] = []
        for source in fact.findall("./source_candidates/source_candidate"):
            source_candidate_id = str(source.attrib.get("id") or "")
            if not source_candidate_id:
                continue
            candidate = {"source_candidate_id": source_candidate_id}
            for key in ("kind", "read"):
                value = str(source.attrib.get(key) or "")
                if value:
                    candidate["read_id" if key == "read" else key] = value
            field_ids = tuple(
                dict.fromkeys(
                    str(field.attrib.get("name") or field.attrib.get("id") or "")
                    for field in source.findall(".//field")
                    if str(field.attrib.get("name") or field.attrib.get("id") or "")
                )
            )
            if field_ids:
                candidate["field_ids"] = " ".join(field_ids)
            api_read = source.find("api_read")
            if api_read is not None:
                candidate.setdefault("read_id", str(api_read.attrib.get("read") or ""))
            source_node = source.find("source")
            if source_node is not None:
                for key, attr in (
                    ("value_id", "value"),
                    ("source_relation_id", "relation"),
                    ("memory_relation_id", "memory_relation"),
                    ("calendar_id", "calendar"),
                    ("kind", "kind"),
                ):
                    value = str(source_node.attrib.get(attr) or "")
                    if value:
                        candidate[key] = value
            candidates.append(candidate)
        if requested_fact_id:
            output[requested_fact_id] = tuple(candidates)
    return output


def _source_alignment_candidate_matches_answer(
    candidate: dict[str, str],
    answer: dict[str, Any],
) -> bool:
    answer_field_ids = _answer_source_field_ids(answer)
    candidate_field_ids = set(str(candidate.get("field_ids") or "").split())
    if (
        answer_field_ids
        and candidate_field_ids
        and not answer_field_ids <= candidate_field_ids
    ):
        return False
    for source in _answer_source_dicts(answer):
        if source.get("kind") and candidate.get("kind"):
            if str(source["kind"]) != candidate["kind"]:
                continue
            if str(source["kind"]) == "same_scope_api_read":
                return True
        if (
            source.get("source_candidate_id")
            and source.get("source_candidate_id") == candidate["source_candidate_id"]
        ):
            return True
        if candidate.get("read_id") and source.get("read_id") == candidate["read_id"]:
            return True
        if (
            candidate.get("value_id")
            and source.get("value_id") == candidate["value_id"]
        ):
            return True
        if candidate.get("source_relation_id") and (
            source.get("relation_id") == candidate["source_relation_id"]
            or source.get("source_relation_id") == candidate["source_relation_id"]
        ):
            return True
        if candidate.get("memory_relation_id") and (
            source.get("relation_id") == candidate["memory_relation_id"]
            or source.get("memory_relation_id") == candidate["memory_relation_id"]
        ):
            return True
        if (
            candidate.get("calendar_id")
            and source.get("calendar_id") == candidate["calendar_id"]
        ):
            return True
    return False


def _answer_source_field_ids(answer: dict[str, Any]) -> set[str]:
    output: set[str] = set()
    for item in answer.get("output_fields") or ():
        if isinstance(item, dict) and str(item.get("field_id") or ""):
            output.add(str(item["field_id"]))
    output_field = answer.get("output_field")
    if isinstance(output_field, dict) and str(output_field.get("field_id") or ""):
        output.add(str(output_field["field_id"]))
    metric = answer.get("metric")
    if isinstance(metric, dict) and str(metric.get("field_id") or ""):
        output.add(str(metric["field_id"]))
    return output


def _answer_source_dicts(answer: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    for key in ("source", "source_hint"):
        value = answer.get(key)
        if isinstance(value, dict):
            output.append(value)
    for key in ("candidate", "observed", "left", "right"):
        value = answer.get(key)
        if isinstance(value, dict) and isinstance(value.get("source"), dict):
            output.append(value["source"])
    if answer.get("pattern") == "computed_scalar":
        for scalar_input in answer.get("scalar_inputs") or ():
            if isinstance(scalar_input, dict):
                if scalar_input.get("value_id"):
                    output.append({"value_id": scalar_input.get("value_id")})
                if scalar_input.get("source"):
                    output.append(scalar_input["source"])
    return tuple(output)


def _answer_value_ids(answer: dict[str, Any]) -> set[str]:
    return {
        str(scalar_input.get("value_id") or "")
        for scalar_input in answer.get("scalar_inputs") or ()
        if isinstance(scalar_input, dict) and str(scalar_input.get("value_id") or "")
    }


def _answer_read_ids(answer: Any) -> set[str]:
    output: set[str] = set()
    if not isinstance(answer, dict):
        return output
    source = answer.get("source") or answer.get("source_hint")
    if isinstance(source, dict) and str(source.get("read_id") or ""):
        output.add(str(source["read_id"]))
    for key in (
        "candidate",
        "observed",
        "left",
        "right",
    ):
        output |= _answer_read_ids(answer.get(key))
    for key in (
        "source",
        "sources",
        "scalar_inputs",
        "output_fields",
        "group_fields",
    ):
        value = answer.get(key)
        if isinstance(value, list):
            for item in value:
                output |= _answer_read_ids(item)
    return output


def _answer_memory_relation_ids(answer: Any) -> set[str]:
    output: set[str] = set()
    if not isinstance(answer, dict):
        return output
    source = answer.get("source")
    if isinstance(source, dict) and str(source.get("memory_relation_id") or ""):
        output.add(str(source["memory_relation_id"]))
    for key in (
        "candidate",
        "observed",
        "left",
        "right",
    ):
        output |= _answer_memory_relation_ids(answer.get(key))
    for key in (
        "source",
        "sources",
        "scalar_inputs",
        "output_fields",
        "group_fields",
    ):
        value = answer.get(key)
        if isinstance(value, list):
            for item in value:
                output |= _answer_memory_relation_ids(item)
    return output


def _answer_calendar_ids(answer: Any) -> set[str]:
    output: set[str] = set()
    if not isinstance(answer, dict):
        return output
    source = answer.get("source")
    if isinstance(source, dict) and str(source.get("calendar_id") or ""):
        output.add(str(source["calendar_id"]))
    for key in (
        "candidate",
        "observed",
        "left",
        "right",
    ):
        output |= _answer_calendar_ids(answer.get(key))
    for key in (
        "source",
        "sources",
        "scalar_inputs",
        "output_fields",
        "group_fields",
    ):
        value = answer.get(key)
        if isinstance(value, list):
            for item in value:
                output |= _answer_calendar_ids(item)
    return output


def _source_strategy_member_values(
    source_strategy: dict[str, Any],
    *,
    key: str,
) -> set[str]:
    return {
        str(member.get(key) or "")
        for member in source_strategy.get("source_members") or ()
        if isinstance(member, dict) and str(member.get(key) or "")
    }


def _source_strategy_member_field_ids(source_strategy: dict[str, Any]) -> set[str]:
    return {
        str(field_id)
        for member in source_strategy.get("source_members") or ()
        if isinstance(member, dict)
        for field_id in (
            tuple(member.get("field_ids") or ())
            + tuple(_source_member_response_field_ids(member))
        )
        if str(field_id)
    }


def _source_member_response_field_ids(member: dict[str, Any]) -> set[str]:
    return {
        str(field.get("field_id") or field.get("name") or "")
        for row in member.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
        and str(field.get("field_id") or field.get("name") or "")
    }


def _source_strategy_member_relation_ids(source_strategy: dict[str, Any]) -> set[str]:
    return {
        relation_id
        for member in source_strategy.get("source_members") or ()
        if isinstance(member, dict)
        for relation_id in (
            str(member.get("memory_relation_id") or ""),
            str(member.get("source_relation_id") or ""),
        )
        if relation_id
    }


def _source_strategy_has_member_kind(
    source_strategy: dict[str, Any],
    *,
    kind: str,
) -> bool:
    return any(
        isinstance(member, dict) and member.get("kind") == kind
        for member in source_strategy.get("source_members") or ()
    )


def bound_fact_plan_payload_from_fact_plan(
    fact_plan: dict[str, Any],
    *,
    prompt: str = "",
    provider_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = fact_plan.get("outcome")
    if isinstance(outcome, dict) and outcome.get("kind") != "fact_plan":
        return json.loads(json.dumps(fact_plan))
    replacements = _bound_source_replacements_from_prompt(prompt)
    if not replacements:
        raise AssertionError("bound fact plan payload requires bound source prompt")
    output = json.loads(json.dumps(fact_plan))
    for answer in output["outcome"]["answers"]:
        replace_answer_sources(answer, replacements=replacements)
        replace_aggregate_choice_selection(answer, prompt=prompt)
        replace_answer_metric(answer, prompt=prompt)
        remove_raw_field_labels(answer)
        remove_disallowed_fixture_answer_fields(
            answer,
            provider_schema=provider_schema,
        )
    return output


def remove_disallowed_fixture_answer_fields(
    answer: dict[str, Any],
    *,
    provider_schema: dict[str, Any] | None,
) -> None:
    """Keep fake model payloads aligned with the current provider schema."""

    allowed_properties = _allowed_answer_properties_for_schema(
        provider_schema,
        answer=answer,
    )
    answer.pop("aggregate_choice", None)
    if not allowed_properties:
        return
    for field in ("output_fields",):
        if field not in allowed_properties:
            answer.pop(field, None)


def _allowed_answer_properties_for_schema(
    provider_schema: dict[str, Any] | None,
    *,
    answer: dict[str, Any],
) -> set[str]:
    if not isinstance(provider_schema, dict):
        return set()
    requested_fact_id = str(answer.get("requested_fact_id") or "")
    pattern = str(answer.get("pattern") or "")
    matches: list[set[str]] = []
    for schema in _iter_json_object_schemas(provider_schema):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            continue
        if "requested_fact_id" not in properties:
            continue
        if not _schema_property_accepts(
            properties.get("requested_fact_id"),
            requested_fact_id,
        ):
            continue
        if pattern and not _schema_property_accepts(properties.get("pattern"), pattern):
            continue
        matches.append(set(properties))
    allowed: set[str] = set()
    for match in matches:
        allowed.update(match)
    return allowed


def _iter_json_object_schemas(schema: Any):
    if not isinstance(schema, dict):
        return
    if isinstance(schema.get("properties"), dict):
        yield schema
    for key in ("oneOf", "anyOf", "allOf"):
        for item in schema.get(key) or ():
            yield from _iter_json_object_schemas(item)
    if isinstance(schema.get("items"), dict):
        yield from _iter_json_object_schemas(schema["items"])
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for item in properties.values():
            yield from _iter_json_object_schemas(item)


def _schema_property_accepts(schema: Any, value: str) -> bool:
    if not isinstance(schema, dict):
        return True
    enum = schema.get("enum")
    if isinstance(enum, list):
        return value in {str(item) for item in enum}
    const = schema.get("const")
    if const is not None:
        return value == str(const)
    return True


def extract_source_bindings(
    fact_plan: dict[str, Any],
    *,
    prompt: str = "",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    bindings: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    source_param_decision_options = _source_candidate_param_decision_options(prompt)
    source_field_ids = _source_candidate_field_ids(prompt)
    source_evidence_items = _source_candidate_evidence_items(prompt)
    source_fulfillment_support_sets = _source_candidate_fulfillment_support_sets(prompt)

    def add_source(
        source: dict[str, Any],
        *,
        answer: dict[str, Any],
        source_role: str = "",
    ) -> str:
        requested_fact_id = _canonical_prompt_requested_fact_id(
            prompt,
            requested_fact_id=str(answer["requested_fact_id"]),
        )
        key = json.dumps(
            {
                "requested_fact_id": requested_fact_id,
                "plan_shape": answer.get("pattern"),
                "source_role": source_role,
                "source": source,
            },
            sort_keys=True,
        )
        source_candidate_id = _source_candidate_id_for_requested_fact(
            source,
            prompt=prompt,
            requested_fact_id=requested_fact_id,
        )
        if source_candidate_id not in source_field_ids:
            raise AssertionError(
                f"source candidate evidence missing for {source_candidate_id}"
            )
        binding_target = _binding_target_for_candidate(
            prompt,
            requested_fact_id=requested_fact_id,
            source_candidate_id=source_candidate_id,
            source_role=source_role,
            plan_shape=_required_prompt_text(answer, key="pattern"),
            allow_equivalent_targets=True,
        )
        binding_target_id = binding_target.binding_target_id
        requires_fulfillment = binding_target.requires_answer_fulfillment
        candidate_evidence_ids = source_field_ids[source_candidate_id]
        selected_evidence_ids: tuple[str, ...] = ()
        if requires_fulfillment:
            selected_evidence_ids = _answer_evidence_ids_for_answer(
                answer,
                source=source,
                candidate_evidence_ids=candidate_evidence_ids,
                candidate_evidence_items=source_evidence_items.get(
                    source_candidate_id,
                    (),
                ),
            )
        if key in replacements:
            if requires_fulfillment:
                _append_source_fulfillments(
                    bindings,
                    source_binding_id=replacements[key],
                    answer=answer,
                    source=source,
                    evidence_ids=selected_evidence_ids,
                    fulfillment_support_sets=source_fulfillment_support_sets.get(
                        source_candidate_id,
                        (),
                    ),
                    source_candidate_id=source_candidate_id,
                    prompt=prompt,
                )
            return replacements[key]
        binding_id = f"sb_{len(bindings) + 1}"
        answer_population = _answer_population(
            prompt,
            source_candidate_id,
            binding_target=binding_target,
        )
        item = {
            "binding_target_id": binding_target_id,
            "answer_population": answer_population,
            "fulfillment_decisions": (
                _source_fulfillments(
                    answer=answer,
                    source=source,
                    evidence_ids=selected_evidence_ids,
                    fulfillment_support_sets=source_fulfillment_support_sets.get(
                        source_candidate_id,
                        (),
                    ),
                    source_candidate_id=source_candidate_id,
                    prompt=prompt,
                )
                if requires_fulfillment
                else {}
            ),
            "param_decisions": _param_decisions_for_prompt(
                _source_param_decision_items(
                    source,
                    param_values=source_param_decision_options.get(
                        source_candidate_id,
                        {},
                    ),
                    population_intent_text=answer_population["intent_text"],
                ),
                prompt=prompt,
            ),
            "resolved_input_applications": (
                _unambiguous_resolved_input_applications(binding_target)
            ),
            "row_predicate_reviews": dict(source.get("row_predicate_reviews") or {}),
        }
        bindings.append(item)
        replacements[key] = binding_id
        return binding_id

    for answer in fact_plan["outcome"]["answers"]:
        for source_role, source in _answer_source_entries(answer):
            add_source(source, answer=answer, source_role=source_role)
        if answer.get("pattern") == "computed_scalar":
            for index, scalar_input in enumerate(
                answer.get("scalar_inputs") or (),
                start=1,
            ):
                source = _source_for_scalar_value(
                    str(scalar_input["value_id"]),
                    prompt=prompt,
                )
                add_source(
                    source,
                    answer=answer,
                    source_role=f"value_{index}",
                )
    return bindings, replacements


def _unambiguous_resolved_input_applications(
    target: _PromptBindingTarget,
) -> list[dict[str, object]]:
    surface = target.payload.get("resolved_input_application")
    raw_options = (
        surface.get("application_options") if isinstance(surface, dict) else ()
    )
    options = tuple(option for option in raw_options or () if isinstance(option, dict))
    options_by_target: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for option in options:
        target_key = (
            str(option.get("target_kind") or ""),
            str(option.get("target_id") or ""),
        )
        options_by_target.setdefault(target_key, []).append(option)
    return [
        {
            "target_kind": target_kind,
            "target_id": target_id,
            "value_id": str(option["value_id"]),
            "value_component": str(option["value_component"]),
            "match_basis_explanation": (
                "Apply the only shown resolved input option for this target."
            ),
            "population_test_results": _satisfying_application_test_results(option),
        }
        for (target_kind, target_id), target_options in options_by_target.items()
        if len(target_options) == 1
        for option in target_options
    ]


def _satisfying_application_test_results(
    option: dict[str, Any],
) -> dict[str, dict[str, str]]:
    raw_basis = option.get("population_test_basis")
    basis = raw_basis if isinstance(raw_basis, dict) else {}
    return {
        str(test_id): {
            "test_id": str(test_id),
            "test_question": str(test_basis["test_question"]),
            "role_scoped_test_question": str(
                test_basis["role_scoped_test_question"]
            ),
            "because": "The selected application enforces this constraint.",
            "test_effect": "SATISFIES_TEST",
        }
        for test_id, test_basis in basis.items()
        if isinstance(test_basis, dict)
    }


def _source_candidate_id_for_requested_fact(
    source: dict[str, Any],
    *,
    prompt: str,
    requested_fact_id: str,
) -> str:
    if prompt:
        payload = _source_candidate_prompt_payload(prompt)
        for candidate in _all_source_candidates(payload):
            matched = _source_candidate_id_for_candidate(source, candidate)
            if matched:
                return matched
        matched = _source_candidate_id_from_prompt(source, payload=payload)
        if matched:
            return matched
    return source_candidate_id_for_source(source, prompt=prompt)


def _source_candidate_id_for_candidate(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    candidate_id = str(candidate.get("source_candidate_id") or "")
    if not candidate_id:
        return ""
    kind = source.get("kind")
    if kind == "read" and _candidate_read_id(candidate) == source.get("read_id"):
        return candidate_id
    if kind == "memory_relation" and candidate.get("memory_relation_id") == source.get(
        "memory_relation_id"
    ):
        return candidate_id
    if kind == "calendar" and candidate.get("calendar_id") == source.get("calendar_id"):
        return candidate_id
    if kind == "value" and candidate.get("value_id") == source.get("value_id"):
        return candidate_id
    return ""


def _canonical_prompt_requested_fact_id(
    prompt: str,
    *,
    requested_fact_id: str,
) -> str:
    if not prompt:
        return requested_fact_id
    try:
        requested_facts = (
            _prompt_json_section(
                prompt,
                label="Requested facts",
            ).get("requested_facts")
            or ()
        )
    except (AssertionError, ValueError):
        return requested_fact_id
    for fact in requested_facts:
        if not isinstance(fact, dict):
            continue
        if str(fact.get("requested_fact_id") or "") == requested_fact_id:
            return requested_fact_id
        if str(fact.get("evidence_ref") or "") == f"requested_fact:{requested_fact_id}":
            return str(fact.get("requested_fact_id") or requested_fact_id)
    facts = tuple(fact for fact in requested_facts if isinstance(fact, dict))
    if len(facts) == 1:
        return str(facts[0].get("requested_fact_id") or requested_fact_id)
    return requested_fact_id


def _requested_fact_id_for_source_candidate(
    prompt: str,
    *,
    source_candidate_id: str,
    default: str,
) -> str:
    if not prompt or not source_candidate_id:
        return default
    try:
        payload = _source_candidate_prompt_payload(prompt)
    except (AssertionError, ValueError):
        return default
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if (
                isinstance(candidate, dict)
                and str(candidate.get("source_candidate_id") or "")
                == source_candidate_id
            ):
                return str(fact_sources.get("requested_fact_id") or default)
    return default


def _answer_sources(answer: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(source for _role, source in _answer_source_entries(answer))


def _answer_source_entries(
    answer: dict[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    output: list[tuple[str, dict[str, Any]]] = []
    source = answer.get("source") or answer.get("source_hint")
    if isinstance(source, dict) and source.get("kind") != "values":
        output.append(("", source))
    for key in ("candidate", "observed", "left", "right"):
        nested = answer.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("source"), dict):
            output.append((key, nested["source"]))
    return tuple(output)


def _source_for_scalar_value(value_id: str, *, prompt: str) -> dict[str, Any]:
    if not prompt:
        return {"kind": "value", "value_id": value_id}
    payload = _source_candidate_prompt_payload(prompt)
    for candidate in _all_source_candidates(payload):
        if candidate.get("kind") == "value" and candidate.get("value_id") == value_id:
            return {
                "kind": "value",
                "source_candidate_id": str(candidate["source_candidate_id"]),
                "value_id": value_id,
            }
    for candidate in _all_source_candidates(payload):
        relation_id = str(candidate.get("memory_relation_id") or "")
        if (
            candidate.get("kind") == "prior_answer_rows"
            and relation_id
            and value_id.startswith(f"{relation_id}.value.")
        ):
            return {
                "kind": "memory_relation",
                "memory_relation_id": relation_id,
            }
    return {"kind": "value", "value_id": value_id}


def _bound_source_replacements_from_prompt(prompt: str) -> dict[str, str]:
    if "Bound sources:\n" not in prompt:
        return {}
    payload = _prompt_json_section(prompt, label="Bound sources")
    output: dict[str, str] = {}
    read_counts: dict[str, int] = {}
    sources = tuple(
        source
        for source in payload.get("bound_sources") or ()
        if isinstance(source, dict)
    )
    for source in sources:
        read_id = str(source.get("read_id") or "")
        if read_id:
            read_counts[read_id] = read_counts.get(read_id, 0) + 1
    for source in sources:
        source_binding_id = str(source.get("source_binding_id") or "")
        if not source_binding_id:
            continue
        for key in _bound_source_keys(source, read_counts=read_counts):
            output.setdefault(json.dumps(key, sort_keys=True), source_binding_id)
    return output


def _bound_source_keys(
    source: dict[str, Any],
    *,
    read_counts: dict[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    if source.get("value_id"):
        return ({"kind": "value", "value_id": source["value_id"]},)
    if source.get("read_id"):
        key: dict[str, Any] = {"kind": "read", "read_id": source["read_id"]}
        bound_params = source.get("bound_params") or ()
        if bound_params:
            key["param_bindings"] = [
                {"param_id": item["param_id"], "value": item["value"]}
                for item in bound_params
                if isinstance(item, dict)
                and item.get("param_id")
                and item.get("value") is not None
            ]
        keys = [key]
        if (
            read_counts
            and read_counts.get(str(source["read_id"])) == 1
            and bound_params
        ):
            keys.append({"kind": "read", "read_id": source["read_id"]})
        return tuple(keys)
    if source.get("memory_relation_id"):
        return (
            {
                "kind": "memory_relation",
                "memory_relation_id": source["memory_relation_id"],
            },
        )
    if source.get("calendar_id"):
        return ({"kind": "calendar", "calendar_id": source["calendar_id"]},)
    return ()


def _append_source_fulfillments(
    bindings: list[dict[str, Any]],
    *,
    source_binding_id: str,
    answer: dict[str, Any],
    source: dict[str, Any],
    evidence_ids: tuple[str, ...],
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    source_candidate_id: str,
    prompt: str,
) -> None:
    index = int(source_binding_id.removeprefix("sb_")) - 1
    item = bindings[index]
    existing = set((item.get("fulfillment_decisions") or {}).keys())
    for answer_output_id, fulfillment in _source_fulfillments(
        answer=answer,
        source=source,
        evidence_ids=evidence_ids,
        fulfillment_support_sets=fulfillment_support_sets,
        source_candidate_id=source_candidate_id,
        prompt=prompt,
    ).items():
        if answer_output_id in existing:
            continue
        item.setdefault("fulfillment_decisions", {})[answer_output_id] = fulfillment


def _source_fulfillments(
    *,
    answer: dict[str, Any],
    source: dict[str, Any],
    evidence_ids: tuple[str, ...],
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    source_candidate_id: str,
    prompt: str,
) -> dict[str, dict[str, Any]]:
    answer_output_ids = tuple(
        str(item) for item in answer.get("answer_output_ids") or ()
    )
    if not answer_output_ids and answer.get("answer_output_id"):
        answer_output_ids = (str(answer["answer_output_id"]),)
    if not answer_output_ids and _answer_uses_aggregate_choice(answer):
        answer_output_ids = tuple(
            dict.fromkeys(
                str(support_set.get("answer_output_id") or "")
                for support_set in fulfillment_support_sets
                if isinstance(support_set, dict)
                and str(support_set.get("answer_output_id") or "")
            )
        )
    if not answer_output_ids:
        answer_output_ids = ("answer_1",)
    answer_output_ids = _answer_output_ids_for_source(
        answer,
        source=source,
        answer_output_ids=answer_output_ids,
    )
    answer_output_ids = tuple(
        _canonical_prompt_answer_output_id(
            prompt,
            requested_fact_id=str(answer.get("requested_fact_id") or ""),
            answer_output_id=answer_output_id,
        )
        for answer_output_id in answer_output_ids
    )
    output: dict[str, dict[str, Any]] = {}
    for answer_output_id in answer_output_ids:
        answer_evidence_ids = _evidence_ids_for_answer_output(
            answer,
            source=source,
            answer_output_id=answer_output_id,
            answer_output_ids=answer_output_ids,
            evidence_ids=evidence_ids,
            fulfillment_support_sets=fulfillment_support_sets,
            prompt=prompt,
        )
        evidence_text = ", ".join(answer_evidence_ids)
        support_set_id = _fulfillment_choice_id_for_evidence(
            answer_output_id=answer_output_id,
            evidence_ids=answer_evidence_ids,
            fulfillment_support_sets=fulfillment_support_sets,
            source_candidate_id=source_candidate_id,
        )
        output[answer_output_id] = {
            "match_basis_explanation": (
                f"{answer_output_id} is fulfilled by {evidence_text} because "
                "the selected source evidence provides the requested output."
            ),
            "fulfillment_choice_id": support_set_id,
        }
    return output


def _answer_output_ids_for_source(
    answer: dict[str, Any],
    *,
    source: dict[str, Any],
    answer_output_ids: tuple[str, ...],
) -> tuple[str, ...]:
    output_fields = tuple(
        field for field in answer.get("output_fields") or () if isinstance(field, dict)
    )
    if not output_fields or not any(field.get("side") for field in output_fields):
        return answer_output_ids
    selected: list[str] = []
    for index, field in enumerate(output_fields):
        side = str(field.get("side") or "")
        scoped = answer.get(side)
        if (
            isinstance(scoped, dict)
            and scoped.get("source") == source
            and index < len(answer_output_ids)
        ):
            selected.append(answer_output_ids[index])
    return tuple(selected)


def _evidence_ids_for_answer_output(
    answer: dict[str, Any],
    *,
    source: dict[str, Any],
    answer_output_id: str,
    answer_output_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    prompt: str,
) -> tuple[str, ...]:
    if _answer_uses_count_record_support(answer):
        selected = _row_population_evidence_for_answer_output(
            answer_output_id=answer_output_id,
            fulfillment_support_sets=fulfillment_support_sets,
        )
        if selected:
            return selected
    output_fields = _answer_output_field_ids_for_source(
        answer,
        source=source,
        answer_output_id=answer_output_id,
        answer_output_ids=answer_output_ids,
    )
    if _answer_uses_aggregate_choice(answer):
        selected = _aggregate_choice_evidence_for_answer_output(
            answer,
            answer_output_id=answer_output_id,
            answer_output_ids=answer_output_ids,
            fulfillment_support_sets=fulfillment_support_sets,
            candidate_evidence_ids=evidence_ids,
        )
        if selected:
            return selected
    for field_id in output_fields:
        selected = _support_set_evidence_for_field(
            field_id,
            answer_output_id=answer_output_id,
            fulfillment_support_sets=fulfillment_support_sets,
            candidate_evidence_ids=evidence_ids,
        )
        if selected:
            return selected
    if not output_fields:
        selected = _selected_source_evidence_for_answer_output(
            answer_output_id=answer_output_id,
            fulfillment_support_sets=fulfillment_support_sets,
            candidate_evidence_ids=evidence_ids,
        )
        if selected:
            return selected
    if _answer_uses_row_population_support(answer):
        selected = _row_population_evidence_for_answer_output(
            answer_output_id=answer_output_id,
            fulfillment_support_sets=fulfillment_support_sets,
        )
        if selected:
            return selected
    raise AssertionError(
        f"fixture answer output has no current-contract support set: {answer_output_id}"
    )


def _answer_uses_count_record_support(answer: dict[str, Any]) -> bool:
    metric = answer.get("metric")
    return (
        str(answer.get("pattern") or "") == "aggregate_scalar"
        and isinstance(metric, dict)
        and str(metric.get("kind") or "") == "count_records"
    )


def _answer_output_field_ids_for_output(
    answer: dict[str, Any],
    *,
    answer_output_id: str,
    answer_output_ids: tuple[str, ...],
) -> list[str]:
    output_fields = [
        str(item.get("field_id") or "")
        for item in answer.get("output_fields") or ()
        if isinstance(item, dict) and str(item.get("field_id") or "")
    ]
    if len(output_fields) == len(answer_output_ids):
        try:
            return [output_fields[answer_output_ids.index(answer_output_id)]]
        except ValueError:
            return []
    output_field = answer.get("output_field")
    if isinstance(output_field, dict) and str(output_field.get("field_id") or ""):
        return [str(output_field["field_id"])]
    metric = answer.get("metric")
    if isinstance(metric, dict) and str(metric.get("field_id") or ""):
        return [str(metric["field_id"])]
    return output_fields


def _answer_output_field_ids_for_source(
    answer: dict[str, Any],
    *,
    source: dict[str, Any],
    answer_output_id: str,
    answer_output_ids: tuple[str, ...],
) -> list[str]:
    for scoped_key in ("candidate", "observed", "left", "right"):
        scoped = answer.get(scoped_key)
        if not isinstance(scoped, dict) or scoped.get("source") != source:
            continue
        output_fields = _field_ids_from_items(scoped.get("output_fields"))
        if len(output_fields) == len(answer_output_ids):
            try:
                return [output_fields[answer_output_ids.index(answer_output_id)]]
            except ValueError:
                return []
        if output_fields:
            return output_fields
        identity_fields = _field_ids_from_items(scoped.get("identity_fields"))
        if identity_fields:
            return identity_fields
    return _answer_output_field_ids_for_output(
        answer,
        answer_output_id=answer_output_id,
        answer_output_ids=answer_output_ids,
    )


def _answer_uses_row_population_support(answer: dict[str, Any]) -> bool:
    if _answer_uses_count_record_support(answer):
        return True
    return str(answer.get("pattern") or "") in {
        "direct_field_value",
        "grouped_rows",
        "joined_rows",
        "list_rows",
    }


def _row_population_evidence_for_answer_output(
    *,
    answer_output_id: str,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        selected = tuple(
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for item in slot.get("row_count_basis_evidence") or ()
            if isinstance(item, dict)
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id
        )
        if selected:
            return selected
    return ()


def _selected_source_evidence_for_answer_output(
    *,
    answer_output_id: str,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    candidate_evidence_ids: tuple[str, ...],
) -> tuple[str, ...]:
    expected = set(candidate_evidence_ids)
    if not expected:
        return ()
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        support_set_evidence_ids = {
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
        }
        if expected <= support_set_evidence_ids:
            return candidate_evidence_ids
    return ()


def _aggregate_choice_evidence_for_answer_output(
    answer: dict[str, Any],
    *,
    answer_output_id: str,
    answer_output_ids: tuple[str, ...],
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    candidate_evidence_ids: tuple[str, ...],
) -> tuple[str, ...]:
    candidate_ids = set(candidate_evidence_ids)
    metric_key = _aggregate_choice_metric_field_id_for_answer(answer)
    expected_group_fields = _aggregate_choice_group_field_ids(answer)
    output_index = (
        answer_output_ids.index(answer_output_id)
        if answer_output_id in answer_output_ids
        else 0
    )
    group_output_count = max(1, len(expected_group_fields))
    if expected_group_fields and output_index < group_output_count:
        for support_set in fulfillment_support_sets:
            if str(support_set.get("answer_output_id") or "") != answer_output_id:
                continue
            evidence_items = tuple(
                item
                for slot in support_set.get("fulfillment_slots") or ()
                if isinstance(slot, dict)
                for key in ("entity_evidence", "value_evidence")
                for item in slot.get(key) or ()
                if isinstance(item, dict)
            )
            group_field_ids = {
                str(field_id)
                for item in evidence_items
                for field_id in _evidence_field_ids(item)
            }
            if not expected_group_fields <= group_field_ids:
                continue
            selected = tuple(
                str(item.get("evidence_id") or "")
                for item in evidence_items
                if str(item.get("evidence_id") or "") in candidate_ids
            )
            if selected:
                return selected
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        evidence_items = tuple(
            item
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for item in _slot_evidence_items(slot)
            if isinstance(item, dict)
        )
        metric_field_ids = {
            str(item.get("field_id") or "")
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for item in slot.get("metric_measure_evidence") or ()
            if isinstance(item, dict)
        }
        row_count_field_ids = {
            str(item.get("field_id") or "")
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for item in slot.get("row_count_basis_evidence") or ()
            if isinstance(item, dict)
        }
        has_metric = bool(
            metric_key
            and (
                metric_key in metric_field_ids
                or _metric_key_matches_row_count_field(
                    metric_key,
                    row_count_fields=row_count_field_ids,
                )
            )
        )
        if not has_metric:
            continue
        group_field_ids = {
            str(field_id)
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for key in ("entity_evidence", "value_evidence")
            for item in slot.get(key) or ()
            if isinstance(item, dict)
            for field_id in _evidence_field_ids(item)
        }
        selected = tuple(
            str(item.get("evidence_id") or "")
            for item in evidence_items
            if str(item.get("evidence_id") or "") in candidate_ids
        )
        if selected:
            return selected
    return ()


def _support_set_evidence_for_field(
    field_id: str,
    *,
    answer_output_id: str,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    candidate_evidence_ids: tuple[str, ...],
) -> tuple[str, ...]:
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        selected = tuple(
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for item in _slot_evidence_items(slot)
            if isinstance(item, dict)
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id in candidate_evidence_ids
            and (
                str(item.get("field_id") or "") == field_id
                or str(item.get("field_id") or "").rsplit(".", 1)[-1] == field_id
                or evidence_id == field_id
                or evidence_id.rsplit(".", 1)[-1] == field_id
            )
        )
        if selected:
            return selected
    return ()


def _first_support_set_evidence_subset(
    *,
    answer_output_id: str,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    candidate_evidence_ids: tuple[str, ...],
) -> tuple[str, ...]:
    candidate_ids = set(candidate_evidence_ids)
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        selected = tuple(
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
            if evidence_id in candidate_ids
        )
        if selected:
            return selected
    return candidate_evidence_ids


def _single_support_set_evidence_subset(
    evidence_ids: tuple[str, ...],
    *,
    answer_output_id: str,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    expected = set(evidence_ids)
    if not expected:
        return evidence_ids
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        support_set_evidence_ids = {
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
        }
        if expected <= support_set_evidence_ids:
            return evidence_ids
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        support_set_evidence_ids = {
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
        }
        subset = tuple(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in support_set_evidence_ids
        )
        if subset:
            return subset
    return evidence_ids


def _evidence_subset_has_support_set(
    evidence_ids: tuple[str, ...],
    *,
    answer_output_id: str,
    fulfillment_support_sets: tuple[dict[str, Any], ...],
) -> bool:
    expected = set(evidence_ids)
    return any(
        expected
        <= {
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
        }
        for support_set in fulfillment_support_sets
        if str(support_set.get("answer_output_id") or "") == answer_output_id
    )


def _fulfillment_choice_id_for_evidence(
    *,
    answer_output_id: str,
    evidence_ids: tuple[str, ...],
    fulfillment_support_sets: tuple[dict[str, Any], ...],
    source_candidate_id: str,
) -> str:
    del source_candidate_id
    expected = set(evidence_ids)
    matches: list[dict[str, Any]] = []
    for support_set in fulfillment_support_sets:
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        support_set_evidence_ids = {
            evidence_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for evidence_id in _slot_evidence_ids(slot)
        }
        if expected <= support_set_evidence_ids:
            matches.append(support_set)
    if matches:
        return str(
            max(
                matches,
                key=lambda item: _support_set_match_score(
                    item,
                    expected_evidence_ids=expected,
                ),
            ).get("fulfillment_choice_id")
            or ""
        )
    raise AssertionError(
        f"fulfillment support set not found for {answer_output_id}:{sorted(expected)}"
    )


def _slot_evidence_ids(slot: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for item in _slot_evidence_items(slot)
        for evidence_id in (str(item.get("evidence_id") or ""),)
        if evidence_id
    )


def _metric_fit_contract_from_prompt(
    prompt: str,
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    bases: dict[str, dict[str, dict[str, str]]] = {}
    interpretations: dict[str, dict[str, dict[str, str]]] = {}
    surface = _prompt_json_section(prompt, label="Metric fit candidates")
    for requested_fact in surface.get("requested_fact_metric_fit_surface") or ():
        if not isinstance(requested_fact, dict):
            continue
        requested_fact_id = str(requested_fact.get("requested_fact_id") or "")
        if not requested_fact_id:
            continue
        metric_candidates = tuple(
            candidate
            for candidate in requested_fact.get("metric_candidates") or ()
            if isinstance(candidate, dict)
        )
        has_declared_measure = any(
            str(candidate.get("field_type") or "") != "row_population"
            for candidate in metric_candidates
        )
        for candidate in metric_candidates:
            if not isinstance(candidate, dict):
                continue
            evidence_id = str(candidate.get("metric_evidence_id") or "")
            if not evidence_id:
                continue
            fits = not (
                has_declared_measure
                and str(candidate.get("field_type") or "") == "row_population"
            )
            fit_text = "fitting" if fits else "not fitting"
            bases.setdefault(requested_fact_id, {})[evidence_id] = {
                "metric_meaning": (
                    f"{evidence_id} is treated as metric evidence in this test fixture."
                ),
                "fit_basis": (
                    f"{evidence_id} is treated as {fit_text} the requested answer in this "
                    "test fixture."
                ),
            }
            interpretations.setdefault(requested_fact_id, {})[evidence_id] = {
                "interpretation": (
                    "FITS_REQUESTED_ANSWER" if fits else "DOES_NOT_FIT_REQUESTED_ANSWER"
                ),
            }
    return {
        "metric_fit_bases": bases,
        "fit_basis_interpretations": interpretations,
    }


def _slot_evidence_items(slot: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for key in (
            "metric_measure_evidence",
            "value_evidence",
            "row_count_basis_evidence",
            "entity_evidence",
        )
        for item in slot.get(key) or ()
        if isinstance(item, dict)
    )


def _canonical_prompt_answer_output_id(
    prompt: str,
    *,
    requested_fact_id: str,
    answer_output_id: str,
) -> str:
    if not prompt:
        return answer_output_id
    try:
        requested_facts = (
            _prompt_json_section(
                prompt,
                label="Requested facts",
            ).get("requested_facts")
            or ()
        )
    except (AssertionError, ValueError):
        return answer_output_id
    matching_facts = [
        fact
        for fact in requested_facts
        if isinstance(fact, dict)
        and (
            str(fact.get("evidence_ref") or "") == f"requested_fact:{requested_fact_id}"
            or str(fact.get("requested_fact_id") or "") == requested_fact_id
        )
    ]
    facts = matching_facts or [
        fact for fact in requested_facts if isinstance(fact, dict)
    ]
    for fact in facts:
        outputs = [
            output
            for output in fact.get("answer_outputs") or ()
            if isinstance(output, dict)
        ]
        if not outputs:
            continue
        for output in outputs:
            if str(output.get("id") or "") == answer_output_id:
                return answer_output_id
        if len(outputs) == 1:
            return str(outputs[0].get("id") or answer_output_id)
    return answer_output_id


def _answer_evidence_ids_for_answer(
    answer: dict[str, Any],
    *,
    source: dict[str, Any],
    candidate_evidence_ids: tuple[str, ...],
    candidate_evidence_items: tuple[tuple[str, str], ...] = (),
) -> tuple[str, ...]:
    answer_field_ids = _source_scoped_answer_field_ids(
        answer,
        source=source,
    ) or _answer_field_ids(answer)
    if not answer_field_ids:
        return candidate_evidence_ids
    selected_from_items = [
        evidence_id
        for evidence_id, field_id in candidate_evidence_items
        if evidence_id in answer_field_ids
        or field_id in answer_field_ids
        or field_id.rsplit(".", 1)[-1] in answer_field_ids
    ]
    if selected_from_items:
        return tuple(selected_from_items)
    selected: list[str] = []
    for evidence_id in candidate_evidence_ids:
        field_id = evidence_id.rsplit(".", 1)[-1]
        if evidence_id in answer_field_ids or field_id in answer_field_ids:
            selected.append(evidence_id)
    return tuple(selected) or candidate_evidence_ids


def _source_scoped_answer_field_ids(
    answer: dict[str, Any],
    *,
    source: dict[str, Any],
) -> set[str]:
    for key in ("candidate", "observed", "left", "right"):
        scoped = answer.get(key)
        if not isinstance(scoped, dict):
            continue
        if scoped.get("source") != source:
            continue
        field_ids = set(_field_ids_from_items(scoped.get("output_fields")))
        if field_ids:
            return field_ids
        field_ids = _field_ids_from_values(scoped.get("identity_fields"))
        if field_ids:
            return field_ids
    return set()


def _answer_field_ids(answer: dict[str, Any]) -> set[str]:
    field_ids: set[str] = set()
    for key in ("output_fields", "group_fields"):
        field_ids.update(_field_ids_from_items(answer.get(key)))
    field_ids.update(_aggregate_choice_group_field_ids(answer))
    output_field = answer.get("output_field")
    if isinstance(output_field, dict) and output_field.get("field_id"):
        field_ids.add(str(output_field["field_id"]))
    metric = answer.get("metric")
    if isinstance(metric, dict) and metric.get("field_id"):
        field_ids.add(str(metric["field_id"]))
    aggregate_metric_field_id = _aggregate_choice_metric_field_id_for_answer(answer)
    if aggregate_metric_field_id:
        field_ids.add(aggregate_metric_field_id)
    for scalar_input in answer.get("scalar_inputs") or ():
        if not isinstance(scalar_input, dict):
            continue
        value_id = str(scalar_input.get("value_id") or "")
        if value_id:
            field_ids.add(value_id.rsplit(".", 1)[-1])
    return field_ids


def _answer_metric_field_id(answer: dict[str, Any]) -> str:
    metric = answer.get("metric")
    if isinstance(metric, dict) and metric.get("field_id"):
        return str(metric["field_id"])
    return _aggregate_choice_metric_field_id_for_answer(answer)


def _field_ids_from_items(raw_items: object) -> tuple[str, ...]:
    if not isinstance(raw_items, (list, tuple, set)):
        return ()
    field_ids: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            field_id = str(item.get("field_id") or "")
        else:
            field_id = str(item or "")
        if field_id:
            field_ids.append(field_id)
    return tuple(dict.fromkeys(field_ids))


def _field_ids_from_values(raw_items: object) -> set[str]:
    return {str(item) for item in raw_items or () if str(item)}


def _source_param_decision_items(
    source: dict[str, Any],
    *,
    param_values: dict[str, dict[str, Any]],
    population_intent_text: str,
) -> dict[str, dict[str, Any]]:
    bindings = {
        str(item.get("param_id") or ""): str(item.get("value") or "")
        for item in source.get("param_bindings") or ()
        if isinstance(item, dict)
    }
    output: dict[str, dict[str, Any]] = {}
    for param_id, options in param_values.items():
        value = bindings.get(param_id)
        if options.get("decision_surface") == "population_choice_set":
            choices = tuple(str(choice) for choice in options.get("choices") or ())
            include_values = (value,) if value else choices[:1]
            output[param_id] = _test_param_decision(
                options=options,
                population_intent=population_intent_text,
                match_basis_explanation=(
                    f"{param_id} values match {population_intent_text} because they are the selected source argument scope."
                ),
                population_choice_set={
                    "include_values": list(include_values),
                    "exclude_values": [
                        choice for choice in choices if choice not in include_values
                    ],
                },
            )
            continue
        selected_option = _selected_bind_option(
            param_id=param_id,
            value=value or "",
            options=options,
        )
        if selected_option is not None:
            output[param_id] = _test_param_decision(
                options=options,
                population_intent=population_intent_text,
                match_basis_explanation=(
                    f"{selected_option['meaning']} This matches {population_intent_text} because it is the selected source argument scope."
                ),
                param_decision_id=selected_option["param_decision_id"],
            )
            continue
        inferred_option = _single_or_preferred_bind_option(
            param_id=param_id,
            options=options,
        )
        if inferred_option is not None:
            output[param_id] = _test_param_decision(
                options=options,
                population_intent=population_intent_text,
                match_basis_explanation=(
                    f"{inferred_option['meaning']} This matches "
                    f"{population_intent_text} because it is the resolved value "
                    "selected for this param."
                ),
                param_decision_id=inferred_option["param_decision_id"],
            )
            continue
        if not value:
            meaning = str(
                options.get("omit_meaning") or f"do not filter {param_id}"
            )
            non_bind_decision_id = str(options.get("non_bind_decision_id") or "")
            if not non_bind_decision_id:
                continue
            output[param_id] = _test_param_decision(
                options=options,
                population_intent=(
                    f"{param_id} is not part of the requested population"
                ),
                match_basis_explanation=(
                    f"{meaning} This matches {param_id} is not part of the "
                    "requested population because it is the selected source "
                    "argument scope."
                ),
                param_decision_id=non_bind_decision_id,
            )
            continue
        meaning = str(options.get("omit_meaning") or f"do not filter {param_id}")
        non_bind_decision_id = str(options.get("non_bind_decision_id") or "")
        if not non_bind_decision_id:
            continue
        output[param_id] = _test_param_decision(
            options=options,
            population_intent=(f"{param_id} is not part of the requested population"),
            match_basis_explanation=(
                f"{meaning} This matches {param_id} is not part of the requested population because it is the selected source argument scope."
            ),
            param_decision_id=non_bind_decision_id,
        )
    return output


def _test_param_decision(
    *,
    options: dict[str, Any],
    population_intent: str,
    match_basis_explanation: str,
    param_decision_id: str = "",
    population_choice_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "population_intent": population_intent,
        "match_basis_explanation": match_basis_explanation,
    }
    if population_choice_set is not None:
        output["population_choice_set"] = population_choice_set
    if param_decision_id:
        output["param_decision_id"] = param_decision_id
    return output


def _single_or_preferred_bind_option(
    *,
    param_id: str,
    options: dict[str, Any],
) -> dict[str, str] | None:
    candidates = tuple(
        option
        for option in options.get("bind_options") or ()
        if isinstance(option, dict)
    )
    if len(candidates) == 1:
        return {
            "meaning": str(candidates[0].get("meaning") or ""),
            "param_decision_id": str(candidates[0].get("param_decision_id") or ""),
        }
    preferred_component = _preferred_time_component_for_test_param(param_id)
    for candidate in candidates:
        if str(candidate.get("value_component") or "") == preferred_component:
            return {
                "meaning": str(candidate.get("meaning") or ""),
                "param_decision_id": str(candidate.get("param_decision_id") or ""),
            }
    return None


def _selected_bind_option(
    *,
    param_id: str,
    value: str,
    options: dict[str, Any],
) -> dict[str, str] | None:
    if not value:
        return None
    candidates = tuple(
        option
        for option in options.get("bind_options") or ()
        if isinstance(option, dict) and option.get("value") == value
    )
    if len(candidates) == 1:
        return {
            "meaning": str(candidates[0].get("meaning") or ""),
            "param_decision_id": str(candidates[0].get("param_decision_id") or ""),
        }
    preferred_component = _preferred_time_component_for_test_param(param_id)
    for candidate in candidates:
        if str(candidate.get("value_component") or "") == preferred_component:
            return {
                "meaning": str(candidate.get("meaning") or ""),
                "param_decision_id": str(candidate.get("param_decision_id") or ""),
            }
    return None


def _preferred_time_component_for_test_param(param_id: str) -> str:
    if param_id in {"interval_start", "start_date", "start_time", "from"}:
        return "start"
    if param_id in {"interval_end", "end_date", "end_time", "to"}:
        return "end"
    return "instant"


def _param_decisions_for_prompt(
    param_decisions: dict[str, dict[str, str]],
    *,
    prompt: str,
) -> dict[str, dict[str, str]] | list[dict[str, str]]:
    if "use param_decisions as an array of decision objects" not in prompt:
        return param_decisions
    return [
        {"param_id": param_id, **decision}
        for param_id, decision in param_decisions.items()
    ]


def _answer_population(
    prompt: str,
    source_candidate_id: str,
    *,
    binding_target: _PromptBindingTarget | None = None,
) -> dict[str, Any]:
    intent_text = _current_question_text(prompt) or "sales"
    target = binding_target or _only_target_for_candidate(
        prompt,
        source_candidate_id=source_candidate_id,
    )
    return {
        "population_binding_id": _source_population_binding_id(
            prompt,
            source_candidate_id,
        ),
        "intent_text": intent_text,
        "match_basis_explanation": f"{intent_text} defines the source population",
        "population_test_results": (
            satisfying_source_population_test_results(target) if target else {}
        ),
    }


def _only_target_for_candidate(
    prompt: str,
    *,
    source_candidate_id: str,
) -> _PromptBindingTarget | None:
    matches = tuple(
        target
        for target in _binding_targets(prompt)
        if target.source_candidate_id == source_candidate_id
    )
    return matches[0] if len(matches) == 1 else None


def _source_population_binding_id(prompt: str, source_candidate_id: str) -> str:
    if not prompt:
        return f"pop.{source_candidate_id}.candidate_population"
    payload = _source_candidate_prompt_payload(prompt)
    for candidate in _all_source_candidates(payload):
        if str(candidate.get("source_candidate_id") or "") != source_candidate_id:
            continue
        for binding in (
            _candidate_binding_surface(candidate).get("population_bindings") or ()
        ):
            if not isinstance(binding, dict):
                continue
            binding_id = str(binding.get("population_binding_id") or "")
            if binding_id:
                return binding_id
    return f"pop.{source_candidate_id}.candidate_population"


def _current_question_text(prompt: str) -> str:
    marker = "Current question:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def _source_candidate_param_decision_options(
    prompt: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_param_options(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_param_options(output, candidate)
    return output


def _source_candidate_field_ids(prompt: str) -> dict[str, tuple[str, ...]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, tuple[str, ...]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_field_ids(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_field_ids(output, candidate)
    return output


def _source_candidate_optional_param_ids(prompt: str) -> dict[str, tuple[str, ...]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, tuple[str, ...]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_optional_param_ids(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_optional_param_ids(output, candidate)
    return output


def _source_candidate_finite_choice_values(
    prompt: str,
) -> dict[str, dict[str, tuple[str, ...]]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_finite_choice_values(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_finite_choice_values(output, candidate)
    return output


def _source_candidate_row_predicate_values(
    prompt: str,
) -> dict[str, dict[str, tuple[str, ...]]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_row_predicate_values(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_row_predicate_values(output, candidate)
    return output


def _source_candidate_population_roles(
    prompt: str,
) -> dict[str, tuple[dict[str, str], ...]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, tuple[dict[str, str], ...]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_population_roles(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_population_roles(output, candidate)
    return output


def _requested_fact_membership_tests(
    prompt: str,
) -> dict[str, tuple[dict[str, str], ...]]:
    if not prompt:
        return {}
    payload = _prompt_json_section(prompt, label="Requested facts")
    output: dict[str, tuple[dict[str, str], ...]] = {}
    for fact in payload.get("requested_facts") or ():
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("requested_fact_id") or "")
        answer_request = fact.get("answer_request")
        if not fact_id or not isinstance(answer_request, dict):
            continue
        answer_population = answer_request.get("answer_population")
        if not isinstance(answer_population, dict):
            continue
        tests = tuple(
            {
                "test_id": _source_binding_membership_test_key(test),
                "test_question": str(test.get("test_question") or ""),
                "kind": str(test.get("kind") or ""),
            }
            for test in answer_population.get("membership_tests") or ()
            if isinstance(test, dict)
            and str(test.get("test_id") or "")
            and str(test.get("test_question") or "")
        )
        if tests:
            output[fact_id] = tests
    return output


def _source_binding_membership_test_key(test: dict[str, Any]) -> str:
    kind = str(test.get("kind") or "")
    if kind == "EXPLICIT_USER_CONSTRAINT":
        return f"{kind.lower()}:{str(test.get('test_id') or '')}"
    if kind in {"SUBJECT_IDENTITY", "NORMAL_INSTANCE_GUARD", "RAW_RECORD_GUARD"}:
        return kind.lower()
    return str(test.get("test_id") or "")


def _source_candidate_default_included_optional_param_ids(
    prompt: str,
) -> dict[str, tuple[str, ...]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, tuple[str, ...]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_default_included_optional_param_ids(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_default_included_optional_param_ids(output, candidate)
    return output


def _source_candidate_evidence_items(
    prompt: str,
) -> dict[str, tuple[tuple[str, str], ...]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, tuple[tuple[str, str], ...]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_evidence_items(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_evidence_items(output, candidate)
    return output


def _source_candidate_fulfillment_support_sets(
    prompt: str,
) -> dict[str, tuple[dict[str, Any], ...]]:
    if not prompt:
        return {}
    payload = _source_candidate_prompt_payload(prompt)
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ):
        for candidate in payload.get(key) or ():
            if isinstance(candidate, dict):
                _add_candidate_fulfillment_support_sets(output, candidate)
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for candidate in _source_options_for_fact_sources(fact_sources):
            if isinstance(candidate, dict):
                _add_candidate_fulfillment_support_sets(output, candidate)
    return output


def _source_candidate_prompt_payload(prompt: str) -> dict[str, Any]:
    for label in ("Candidate evidence sources", "Chosen evidence sources"):
        if f"{label}:\n" in prompt:
            text = _prompt_text_section(prompt, label=label)
            if text.startswith("<candidate_evidence_sources>"):
                return _candidate_evidence_sources_payload_from_xml(text)
            payload = _prompt_json_section(prompt, label=label)
            if "chosen_source_candidates" in payload:
                return {
                    "requested_fact_sources": [
                        {
                            "source_contexts": [
                                {
                                    "source_options": (
                                        payload.get("chosen_source_candidates") or []
                                    )
                                }
                            ]
                        }
                    ]
                }
            return payload
    if "Selected source invocations:\n" in prompt:
        payload = _prompt_json_section(prompt, label="Selected source invocations")
        return {
            "requested_fact_sources": [
                {
                    "source_contexts": [
                        {"source_options": payload.get("source_invocations") or []}
                    ]
                }
            ]
        }
    return {}


def _candidate_evidence_sources_payload_from_xml(text: str) -> dict[str, Any]:
    root = ElementTree.fromstring(text)
    payload: dict[str, Any] = {"requested_fact_sources": []}
    for fact_node in root.findall("requested_fact"):
        fact_payload = {
            "requested_fact_id": fact_node.attrib.get("id", ""),
            "source_contexts": [],
        }
        for context_node in fact_node.findall("source_context"):
            fact_payload["source_contexts"].append(
                {
                    "context_id": context_node.attrib.get("id", ""),
                    "kind": context_node.attrib.get("kind", ""),
                    "source_options": tuple(
                        _source_candidate_from_xml(candidate_node)
                        for candidate_node in context_node
                        if candidate_node.tag in {"api_read", "source"}
                    ),
                }
            )
        payload["requested_fact_sources"].append(fact_payload)
    for tag, key in (
        ("memory_sources", "memory_source_candidates"),
        ("utility_sources", "utility_source_candidates"),
        ("value_sources", "value_source_candidates"),
    ):
        candidates = tuple(
            _source_candidate_from_xml(candidate_node)
            for group_node in root.findall(tag)
            for candidate_node in group_node
            if candidate_node.tag in {"api_read", "source"}
        )
        if candidates:
            payload[key] = candidates
    return payload


def _source_candidate_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    attrs = node.attrib
    candidate = {
        "source_candidate_id": attrs.get("id", ""),
        "kind": attrs.get("kind", ""),
    }
    for xml_key, payload_key in (
        ("read", "read_id"),
        ("row_source", "row_source_id"),
        ("value", "value_id"),
        ("relation", "source_relation_id"),
        ("memory_relation", "memory_relation_id"),
        ("field", "source_field_id"),
        ("calendar", "calendar_id"),
        ("cardinality", "cardinality"),
    ):
        if attrs.get(xml_key):
            candidate[payload_key] = attrs[xml_key]
    description = node.findtext("description", default="").strip()
    if description:
        candidate["description"] = description
    for key, value in _source_binding_surface_from_xml(node).items():
        if value:
            candidate[key] = value
    response_rows = tuple(
        _response_row_from_xml(row) for row in node.findall("response/row")
    )
    if response_rows:
        candidate["response_rows"] = response_rows
    row_predicates = tuple(
        _row_predicate_from_xml(predicate)
        for predicate in node.findall("row_predicates/predicate")
    )
    if row_predicates:
        candidate["row_predicates"] = row_predicates
    return {key: value for key, value in candidate.items() if value != ""}


def _source_binding_surface_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    return {
        "fields": tuple(
            _field_item_from_xml(field)
            for fields_node in node.findall("fields")
            for field in fields_node.findall("field")
        ),
        "evidence_items": tuple(
            _field_item_from_xml(evidence)
            for evidence_node in node.findall("evidence_items")
            for evidence in evidence_node.findall("evidence")
        ),
        "params": tuple(
            _binding_param_from_xml(param)
            for param in node.findall("binding_params/param")
        ),
        "population_bindings": tuple(
            dict(population.attrib)
            for population in node.findall("population_bindings/population")
        ),
        "population_roles": tuple(
            _population_role_from_xml(role)
            for role in node.findall("population_roles/role")
        ),
        "fulfillment_support_sets": tuple(
            _fulfillment_choice_from_xml(choice)
            for choice in node.findall("fulfillment_choices/choice")
        ),
    }


def _field_item_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    return {
        "field_id": node.attrib.get("name", ""),
        "id": node.attrib.get("name", ""),
        "evidence_id": node.attrib.get("id", ""),
        "field_path": node.attrib.get("path", ""),
        "path": node.attrib.get("path", ""),
        "type": node.attrib.get("type", ""),
    }


def _response_row_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    return {
        "path": node.attrib.get("path", ""),
        "cardinality": node.attrib.get("cardinality", ""),
        "evidence_token": node.attrib.get("evidence_token", ""),
        "fields": tuple(
            {
                "field_id": field.attrib.get("name", ""),
                "path": field.attrib.get("path", ""),
                "type": field.attrib.get("type", ""),
                "evidence_token": field.attrib.get("evidence_token", ""),
            }
            for field in node.findall("field")
        ),
    }


def _row_predicate_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    return {
        "predicate_id": node.attrib.get("id", ""),
        "field_id": node.attrib.get("field", ""),
        "field_path": node.attrib.get("path", ""),
        "row_path_id": node.attrib.get("row", ""),
        "type": node.attrib.get("type", ""),
        "operator": node.attrib.get("operator", ""),
        "default": node.attrib.get("default", ""),
        "allowed_values": tuple(
            str(value.text or "").strip()
            for value in node.findall("values/value")
            if str(value.text or "").strip()
        ),
    }


def _binding_param_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    param = {
        "param_id": node.attrib.get("param_id", ""),
        "name": node.attrib.get("name", ""),
        "type": node.attrib.get("type", ""),
        "required": node.attrib.get("required") == "True",
        "decision_surface": node.attrib.get("decision_surface", ""),
        "choices": tuple(
            choice.attrib.get("value", "")
            for choice in node.findall("choices/choice")
            if choice.attrib.get("value")
        ),
        "decision_options": tuple(
            _decision_option_from_xml(option)
            for option in node.findall("decision_options/option")
        ),
    }
    population_contract = node.find("population_contract")
    if population_contract is not None:
        param["population_contract"] = dict(population_contract.attrib)
    return param


def _decision_option_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    return {
        "param_decision_id": node.attrib.get("id", ""),
        "decision": node.attrib.get("decision", ""),
        "value": node.attrib.get("value", ""),
        "value_component": node.attrib.get("value_component", ""),
        "meaning": node.attrib.get("meaning", ""),
    }


def _population_role_from_xml(node: ElementTree.Element) -> dict[str, str]:
    return {
        "role_id": node.attrib.get("id", ""),
        "row_path_id": node.attrib.get("row_path", ""),
        "role_kind": node.attrib.get("kind", ""),
        "role_text": node.attrib.get("text", ""),
    }


def _fulfillment_choice_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    return {
        "fulfillment_choice_id": node.attrib.get("id", ""),
        "answer_output_id": node.attrib.get("answer_output", ""),
        "fulfillment_slots": (
            {
                "metric_measure_evidence": tuple(
                    _fulfillment_evidence_from_xml(evidence)
                    for evidence in node.findall("evidence")
                    if evidence.attrib.get("kind") == "metric"
                ),
                "row_count_basis_evidence": tuple(
                    _fulfillment_evidence_from_xml(evidence)
                    for evidence in node.findall("evidence")
                    if evidence.attrib.get("kind") == "row_count_basis"
                ),
                "value_evidence": tuple(
                    _fulfillment_evidence_from_xml(evidence)
                    for evidence in node.findall("evidence")
                    if evidence.attrib.get("kind") == "value"
                ),
                "entity_evidence": tuple(
                    _fulfillment_evidence_from_xml(evidence)
                    for evidence in node.findall("evidence")
                    if evidence.attrib.get("kind")
                    in {"candidate_key", "entity_reference"}
                ),
            },
        ),
    }


def _fulfillment_evidence_from_xml(node: ElementTree.Element) -> dict[str, Any]:
    output = {
        "evidence_id": node.attrib.get("evidence_id", ""),
        "field_id": node.attrib.get("field", ""),
        "label": node.attrib.get("label", ""),
        "row_path_id": node.attrib.get("row_path", ""),
        "type": node.attrib.get("type", ""),
        "entity_key": node.attrib.get("entity_key", ""),
    }
    if node.attrib.get("kind") in {"candidate_key", "entity_reference"}:
        output["components"] = [
            {"component_id": field_id.strip(), "field_id": field_id.strip()}
            for field_id in node.attrib.get("field", "").split(",")
            if field_id.strip()
        ]
    return output


def _evidence_field_ids(item: dict[str, Any]) -> tuple[str, ...]:
    component_field_ids = tuple(
        str(component.get("field_id") or "")
        for component in item.get("components") or ()
        if isinstance(component, dict) and str(component.get("field_id") or "")
    )
    if component_field_ids:
        return component_field_ids
    field_id = str(item.get("field_id") or "")
    return (field_id,) if field_id else ()


def _source_options_for_fact_sources(
    fact_sources: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        candidate
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
    )


def _add_candidate_param_options(
    output: dict[str, dict[str, dict[str, Any]]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    output[source_candidate_id] = {
        param_id: _param_options(item)
        for item in _candidate_binding_surface(candidate).get("params") or ()
        if isinstance(item, dict)
        for param_id in (str(item.get("param_id") or ""),)
        if param_id and _param_options(item)
    }


def _param_options(param: dict[str, Any]) -> dict[str, Any]:
    if param.get("decision_surface") == "population_choice_set":
        choices = tuple(
            str(choice or "").strip()
            for choice in param.get("choices") or ()
            if str(choice or "").strip()
        )
        if choices:
            return {
                "required": param.get("required") is True,
                "decision_surface": "population_choice_set",
                "choices": choices,
            }
    decision_options = param.get("decision_options")
    bind_options = tuple(
        item
        for item in decision_options or ()
        if isinstance(item, dict)
        and item.get("decision") == "bind"
        and str(item.get("value") or "")
        and str(item.get("param_decision_id") or "")
    )
    bind_meanings = {
        str(item.get("value") or ""): str(item.get("meaning") or "")
        for item in bind_options
        if str(item.get("value") or "") and str(item.get("meaning") or "")
    }
    bind_decision_ids = {
        str(item.get("value") or ""): str(item.get("param_decision_id") or "")
        for item in bind_options
    }
    if not bind_meanings:
        return {}
    non_bind_options = [
        item
        for item in decision_options or ()
        if isinstance(item, dict) and item.get("decision") != "bind"
    ]
    omit_option = non_bind_options[0] if non_bind_options else {}
    return {
        "required": param.get("required") is True,
        "bind_meanings": bind_meanings,
        "bind_decision_ids": bind_decision_ids,
        "bind_options": bind_options,
        "omit_decision": str(omit_option.get("decision") or ""),
        "omit_meaning": str(omit_option.get("meaning") or ""),
        "non_bind_decision_id": str(omit_option.get("param_decision_id") or ""),
        "normal_instance_role_profiles": tuple(
            item
            for item in param.get("normal_instance_role_profiles") or ()
            if isinstance(item, dict)
        ),
    }


def _add_candidate_field_ids(
    output: dict[str, tuple[str, ...]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    fields = tuple(
        str(item.get("field_id") or item.get("id") or "")
        for item in (
            _candidate_binding_surface(candidate).get("fields")
            or candidate.get("fields")
            or candidate.get("columns")
            or ()
        )
        if isinstance(item, dict) and str(item.get("field_id") or item.get("id") or "")
    )
    evidence_items = tuple(
        str(item.get("evidence_id") or "")
        for item in _candidate_binding_surface(candidate).get("evidence_items") or ()
        if isinstance(item, dict) and str(item.get("evidence_id") or "")
    )
    support_set_evidence_items = tuple(
        str(item.get("evidence_id") or "")
        for support_set in _candidate_binding_surface(candidate).get(
            "fulfillment_support_sets"
        )
        or ()
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
        for key in (
            "metric_measure_evidence",
            "value_evidence",
            "row_count_basis_evidence",
            "entity_evidence",
        )
        for item in slot.get(key) or ()
        if isinstance(item, dict) and str(item.get("evidence_id") or "")
    )
    value_id = str(candidate.get("value_id") or "")
    output[source_candidate_id] = (
        evidence_items
        or support_set_evidence_items
        or fields
        or ((value_id,) if value_id else ())
    )


def _add_candidate_optional_param_ids(
    output: dict[str, tuple[str, ...]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    output[source_candidate_id] = tuple(
        str(item.get("param_id") or "")
        for item in _candidate_binding_surface(candidate).get("params") or ()
        if isinstance(item, dict)
        and str(item.get("param_id") or "")
        and item.get("required") is not True
    )


def _add_candidate_finite_choice_values(
    output: dict[str, dict[str, tuple[str, ...]]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    choice_values = {
        param_id: tuple(
            str(choice) for choice in item.get("choices") or () if str(choice)
        )
        for item in _candidate_binding_surface(candidate).get("params") or ()
        if isinstance(item, dict)
        and item.get("choices")
        and isinstance(item.get("population_contract"), dict)
        for param_id in (str(item.get("param_id") or ""),)
        if param_id
    }
    if choice_values:
        output[source_candidate_id] = choice_values


def _add_candidate_row_predicate_values(
    output: dict[str, dict[str, tuple[str, ...]]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    predicate_values = {
        predicate_id: tuple(
            str(value) for value in item.get("allowed_values") or () if str(value)
        )
        for item in candidate.get("row_predicates") or ()
        if isinstance(item, dict)
        for predicate_id in (str(item.get("predicate_id") or ""),)
        if predicate_id
    }
    if predicate_values:
        output[source_candidate_id] = predicate_values


def _add_candidate_population_roles(
    output: dict[str, tuple[dict[str, str], ...]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    binding_surface = _candidate_binding_surface(candidate)
    roles = tuple(
        {
            "role_id": str(item.get("role_id") or ""),
            "row_path_id": str(item.get("row_path_id") or ""),
            "role_kind": str(item.get("role_kind") or ""),
            "role_text": str(item.get("role_text") or ""),
        }
        for item in binding_surface.get("population_roles") or ()
        if isinstance(item, dict)
        and str(item.get("role_id") or "")
        and str(item.get("row_path_id") or "")
        and str(item.get("role_kind") or "")
        and str(item.get("role_text") or "")
    )
    if roles:
        output[source_candidate_id] = roles


def _add_candidate_default_included_optional_param_ids(
    output: dict[str, tuple[str, ...]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    output[source_candidate_id] = tuple(
        str(item.get("param_id") or "")
        for item in _candidate_binding_surface(candidate).get("params") or ()
        if isinstance(item, dict)
        and str(item.get("param_id") or "")
        and item.get("required") is not True
        and str(item.get("type") or "") != "choice"
    )


def _add_candidate_evidence_items(
    output: dict[str, tuple[tuple[str, str], ...]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    evidence_items = tuple(
        (str(item.get("evidence_id") or ""), str(item.get("field_id") or ""))
        for item in _candidate_binding_surface(candidate).get("evidence_items") or ()
        if isinstance(item, dict)
        and str(item.get("evidence_id") or "")
        and str(item.get("field_id") or "")
    )
    if not evidence_items:
        evidence_items = tuple(
            (str(item.get("evidence_id") or ""), str(item.get("field_id") or ""))
            for support_set in _candidate_binding_surface(candidate).get(
                "fulfillment_support_sets"
            )
            or ()
            if isinstance(support_set, dict)
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for key in (
                "metric_measure_evidence",
                "value_evidence",
                "row_count_basis_evidence",
                "entity_evidence",
            )
            for item in slot.get(key) or ()
            if isinstance(item, dict)
            and str(item.get("evidence_id") or "")
            and str(item.get("field_id") or "")
        )
    if evidence_items:
        output[source_candidate_id] = evidence_items


def _add_candidate_fulfillment_support_sets(
    output: dict[str, tuple[dict[str, Any], ...]],
    candidate: dict[str, Any],
) -> None:
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    if not source_candidate_id:
        return
    support_sets = tuple(
        support_set
        for support_set in _candidate_binding_surface(candidate).get(
            "fulfillment_support_sets"
        )
        or ()
        if isinstance(support_set, dict)
        and str(support_set.get("fulfillment_choice_id") or "")
    )
    if support_sets:
        output[source_candidate_id] = support_sets


def _candidate_binding_surface(candidate: dict[str, Any]) -> dict[str, Any]:
    surface = candidate.get("binding_surface")
    if isinstance(surface, dict):
        return surface
    if candidate.get("kind") not in {"new_api_read", "same_scope_api_read"}:
        return candidate
    output = {
        key: candidate[key]
        for key in (
            "applied_filters",
            "bound_params",
            "source_invocations",
            "population_bindings",
            "params",
            "population_roles",
        )
        if key in candidate
    }
    if "fulfillment_support_sets" in candidate:
        output["fulfillment_support_sets"] = candidate["fulfillment_support_sets"]
    elif "fulfillment_choices" in candidate:
        output["fulfillment_support_sets"] = candidate["fulfillment_choices"]
    fields = [
        field
        for row in candidate.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    ]
    if fields:
        output["fields"] = fields
    return output


def _prompt_json_section(prompt: str, *, label: str) -> dict[str, Any]:
    return prompt_section_payload(prompt, label)


def _prompt_text_section(prompt: str, *, label: str) -> str:
    marker = f"{label}:\n"
    start = prompt.index(marker) + len(marker)
    next_section = prompt.find("\n\n", start)
    if next_section < 0:
        return prompt[start:].strip()
    return prompt[start:next_section].strip()


def replace_answer_sources(
    answer: dict[str, Any],
    *,
    replacements: dict[str, str],
) -> None:
    source = answer.pop("source", None) or answer.pop("source_hint", None)
    if isinstance(source, dict):
        if source.get("kind") != "values":
            answer["source_binding_id"] = replacements[
                json.dumps(source, sort_keys=True)
            ]
    for key in ("candidate", "observed", "left", "right"):
        if isinstance(answer.get(key), dict) and "source" in answer[key]:
            answer[key]["source_binding_id"] = replacements[
                json.dumps(answer[key].pop("source"), sort_keys=True)
            ]
def remove_raw_field_labels(answer: dict[str, Any]) -> None:
    """Emit the current fact-plan contract from older concise test fixtures."""

    for key in ("group_fields", "output_fields"):
        if isinstance(answer.get(key), list):
            _remove_labels(answer[key])
    if isinstance(answer.get("output_field"), dict):
        answer["output_field"].pop("label", None)
    for key in ("candidate", "left", "right"):
        operand = answer.get(key)
        if not isinstance(operand, dict):
            continue
        if isinstance(operand.get("fields"), list):
            _remove_labels(operand["fields"])
        if isinstance(operand.get("output_fields"), list):
            _remove_labels(operand["output_fields"])


def _answer_uses_aggregate_choice(answer: dict[str, Any]) -> bool:
    if str(answer.get("pattern") or "") != "aggregate_by_group":
        return False
    return isinstance(answer.get("aggregate_choice"), dict) or (
        isinstance(answer.get("group"), dict)
        and isinstance(answer.get("metric"), dict)
        and isinstance(answer.get("function"), dict)
    )


def _remove_labels(items: list[dict[str, Any]]) -> None:
    for item in items:
        if isinstance(item, dict):
            item.pop("label", None)


def replace_answer_metric(answer: dict[str, Any], *, prompt: str = "") -> None:
    metric = answer.get("metric")
    if not isinstance(metric, dict):
        return
    source_binding_id = str(answer.get("source_binding_id") or "")
    if str(answer.get("pattern") or "") == "aggregate_by_group":
        return
    if str(answer.get("pattern") or "") == "aggregate_scalar":
        replacement = _matching_scalar_aggregate_selection(
            prompt,
            source_binding_id=source_binding_id,
            metric=metric,
        )
        if replacement:
            answer.update(replacement)
            return
        if "id" in metric and isinstance(answer.get("function"), dict):
            return
        raise AssertionError("fact-plan fixture must select exact scalar choices")


def _matching_scalar_aggregate_selection(
    prompt: str,
    *,
    source_binding_id: str,
    metric: dict[str, Any],
) -> dict[str, Any]:
    if not prompt:
        return {}
    try:
        choices = _scalar_aggregate_choices_from_prompt(prompt)
    except ValueError:
        return {}
    metric_function = str(metric.get("function") or "")
    for choice in choices:
        if source_binding_id and choice["source_binding_id"] != source_binding_id:
            continue
        selected_metric = _matching_metric_candidate(choice, _metric_field_id(metric))
        if not selected_metric and str(metric.get("kind") or "") == "count_records":
            selected_metric = _matching_metric_candidate(choice, "")
        selected_function = _matching_function_candidate(
            choice,
            metric_function,
            metric=selected_metric,
        )
        if not selected_metric or not selected_function:
            continue
        return {
            "metric": {
                "selection_basis": "Selected from scalar aggregate operation choices.",
                "id": selected_metric["id"],
                "kind": selected_metric["kind"],
                **(
                    {"field_id": selected_metric["field"]}
                    if selected_metric.get("field")
                    else {}
                ),
            },
            "function": {
                "selection_basis": "Selected from scalar aggregate operation choices.",
                "id": selected_function["id"],
                "value": selected_function["value"],
            },
        }
    return {}


def _metric_field_id(metric: dict[str, Any]) -> str:
    return str(metric.get("field_id") or metric.get("input_field") or "")


def replace_aggregate_choice_selection(
    answer: dict[str, Any],
    *,
    prompt: str,
) -> None:
    if str(answer.get("pattern") or "") != "aggregate_by_group":
        return
    aggregate_choice = (
        answer.get("aggregate_choice")
        if isinstance(answer.get("aggregate_choice"), dict)
        else None
    )
    if not aggregate_choice:
        return
    try:
        grouped_choices = _grouped_aggregate_choices_from_prompt(prompt)
    except (AssertionError, ValueError):
        return
    replacement = _matching_grouped_aggregate_selection(
        grouped_choices,
        answer=answer,
        aggregate_choice=aggregate_choice,
    )
    if not replacement:
        return
    answer.update(replacement)
    answer.pop("aggregate_choice", None)


def _matching_grouped_aggregate_selection(
    choices: tuple[dict[str, Any], ...],
    *,
    answer: dict[str, Any],
    aggregate_choice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_field_id = _aggregate_choice_metric_field_id_for_answer(answer)
    metric_function = ""
    expected_group_fields = _aggregate_choice_group_field_ids_for_answer(
        answer,
        candidate_field_ids=set(),
    )
    if aggregate_choice:
        metric_field_id = str(aggregate_choice.get("metric_field_id") or "")
        metric_function = str(aggregate_choice.get("metric_function") or "")
        expected_group_fields = {
            str(field_id)
            for field_id in aggregate_choice.get("group_field_ids") or ()
            if str(field_id)
        }
    source_binding_id = str(answer.get("source_binding_id") or "")
    for choice in choices:
        if source_binding_id and choice["source_binding_id"] != source_binding_id:
            continue
        group = _matching_group_candidate(choice, expected_group_fields)
        metric = _matching_metric_candidate(choice, metric_field_id)
        function = _matching_function_candidate(choice, metric_function, metric=metric)
        if group and metric and function:
            return {
                "source_binding_id": choice["source_binding_id"],
                "metric": {
                    "selection_basis": "Selected from grouped aggregate operation choices.",
                    "id": metric["id"],
                    "kind": metric["kind"],
                    **({"field_id": metric["field"]} if metric.get("field") else {}),
                },
                "function": {
                    "selection_basis": "Selected from grouped aggregate operation choices.",
                    "id": function["id"],
                    "value": function["value"],
                },
            }
    return {}


def _matching_group_candidate(
    choice: dict[str, Any],
    expected_group_fields: set[str],
) -> dict[str, Any]:
    for group in choice["groups"]:
        group_fields = set(group.get("fields") or ())
        if not expected_group_fields or group_fields == expected_group_fields:
            return group
    return {}


def _matching_metric_candidate(
    choice: dict[str, Any],
    metric_field_id: str,
) -> dict[str, str]:
    for metric in choice["metrics"]:
        if metric["kind"] == "count_records" and not metric_field_id:
            return metric
        if metric_field_id and metric.get("field") == metric_field_id:
            return metric
    return {}


def _matching_function_candidate(
    choice: dict[str, Any],
    metric_function: str,
    *,
    metric: dict[str, str],
) -> dict[str, str]:
    allowed = set(str(metric.get("allowed_functions") or "").split())
    for function in choice["functions"]:
        if function["value"] not in allowed:
            continue
        if not metric_function or function["value"] == metric_function:
            return function
    return {}


def _grouped_aggregate_choices_from_prompt(prompt: str) -> tuple[dict[str, Any], ...]:
    section = _prompt_text_section(
        prompt,
        label="Grouped aggregate operation choices",
    )
    choices: list[dict[str, Any]] = []
    for source_match in re.finditer(
        r'<source_binding id="([^"]+)" read="([^"]*)">(.*?)</source_binding>',
        section,
        re.S,
    ):
        body = source_match.group(3)
        choices.append(
            {
                "source_binding_id": source_match.group(1),
                "groups": tuple(
                    {
                        "fields": tuple(
                            field_id
                            for field_id in attrs.get("fields", "").split()
                            if field_id
                        ),
                    }
                    for attrs in _xml_tag_attrs(body, "group")
                ),
                "metrics": tuple(
                    {
                        "id": attrs.get("id", ""),
                        "kind": attrs.get("kind", ""),
                        "field": attrs.get("field", ""),
                        "allowed_functions": attrs.get("allowed_functions", ""),
                    }
                    for attrs in _xml_tag_attrs(body, "metric")
                ),
                "functions": tuple(
                    {
                        "id": attrs.get("id", ""),
                        "value": attrs.get("value", ""),
                    }
                    for attrs in _xml_tag_attrs(body, "function")
                ),
            }
        )
    return tuple(choices)


def _scalar_aggregate_choices_from_prompt(prompt: str) -> tuple[dict[str, Any], ...]:
    section = _prompt_text_section(
        prompt,
        label="Scalar aggregate operation choices",
    )
    choices: list[dict[str, Any]] = []
    for source_match in re.finditer(
        r'<source_binding id="([^"]+)" read="([^"]*)">(.*?)</source_binding>',
        section,
        re.S,
    ):
        body = source_match.group(3)
        choices.append(
            {
                "source_binding_id": source_match.group(1),
                "metrics": tuple(
                    {
                        "id": attrs.get("id", ""),
                        "kind": attrs.get("kind", ""),
                        "field": attrs.get("field", ""),
                        "allowed_functions": attrs.get("allowed_functions", ""),
                    }
                    for attrs in _xml_tag_attrs(body, "metric")
                ),
                "functions": tuple(
                    {
                        "id": attrs.get("id", ""),
                        "value": attrs.get("value", ""),
                    }
                    for attrs in _xml_tag_attrs(body, "function")
                ),
            }
        )
    return tuple(choices)


def _xml_tag_attrs(text: str, tag: str) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            attr_match.group(1): attr_match.group(2)
            for attr_match in re.finditer(
                r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', match.group(1)
            )
        }
        for match in re.finditer(rf"<{tag}\s+([^>]*)/?>", text)
    )


def _aggregate_choice_metric_field_id_for_answer(answer: dict[str, Any]) -> str:
    metric = answer.get("metric")
    if isinstance(metric, dict) and str(metric.get("field_id") or ""):
        return str(metric["field_id"])
    aggregate_choice = answer.get("aggregate_choice")
    if isinstance(aggregate_choice, dict):
        return str(aggregate_choice.get("metric_field_id") or "")
    return ""


def _aggregate_choice_group_field_ids(answer: dict[str, Any]) -> set[str]:
    group = answer.get("group")
    if isinstance(group, dict) and str(group.get("field_id") or ""):
        return {str(group["field_id"])}
    aggregate_choice = answer.get("aggregate_choice")
    if not isinstance(aggregate_choice, dict):
        return set()
    return {
        str(field_id)
        for field_id in aggregate_choice.get("group_field_ids") or ()
        if str(field_id)
    }


def _aggregate_choice_group_field_ids_for_answer(
    answer: dict[str, Any],
    *,
    candidate_field_ids: set[str],
) -> set[str]:
    del candidate_field_ids
    return _aggregate_choice_group_field_ids(answer)


def source_candidate_id_for_source(source: dict[str, Any], *, prompt: str = "") -> str:
    if prompt:
        payload = _source_candidate_prompt_payload(prompt)
        matched = _source_candidate_id_from_prompt(source, payload=payload)
        if matched:
            return matched
    kind = source.get("kind")
    if source.get("source_candidate_id"):
        return str(source["source_candidate_id"])
    if kind == "read":
        return source["read_id"]
    if kind == "same_scope_api_read":
        return source["read_id"]
    if kind == "memory_relation":
        return source["memory_relation_id"]
    if kind == "calendar":
        return source["calendar_id"]
    if kind == "value":
        return source["value_id"]
    raise AssertionError(f"unsupported source kind: {kind}")


def _source_candidate_id_from_prompt(
    source: dict[str, Any],
    *,
    payload: dict[str, Any],
) -> str:
    for candidate in _all_source_candidates(payload):
        candidate_id = str(candidate.get("source_candidate_id") or "")
        if not candidate_id:
            continue
        kind = source.get("kind")
        if kind == "read" and _candidate_read_id(candidate) == source.get("read_id"):
            return candidate_id
        if (
            kind == "same_scope_api_read"
            and candidate.get("kind") == "same_scope_api_read"
            and _candidate_read_id(candidate) == source.get("read_id")
        ):
            return candidate_id
        if kind == "memory_relation" and candidate.get(
            "memory_relation_id"
        ) == source.get("memory_relation_id"):
            return candidate_id
        if kind == "calendar" and candidate.get("calendar_id") == source.get(
            "calendar_id"
        ):
            return candidate_id
        if kind == "value" and candidate.get("value_id") == source.get("value_id"):
            return candidate_id
    return ""


def _candidate_read_id(candidate: dict[str, Any]) -> str:
    read_id = str(candidate.get("read_id") or "")
    if read_id:
        return read_id
    read_contract = candidate.get("read_contract")
    if isinstance(read_contract, dict):
        return str(read_contract.get("read_id") or "")
    return ""


def _all_source_candidates(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    for key in (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
        "source_invocations",
    ):
        output.extend(
            candidate
            for candidate in payload.get(key) or ()
            if isinstance(candidate, dict)
        )
    for fact_sources in payload.get("requested_fact_sources") or ():
        if isinstance(fact_sources, dict):
            output.extend(_source_options_for_fact_sources(fact_sources))
    return tuple(output)
