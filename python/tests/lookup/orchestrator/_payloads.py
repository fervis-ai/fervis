from dataclasses import replace
import re
from typing import cast, Iterable
from xml.etree import ElementTree

from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole

from tests.lookup.orchestrator._plans import *  # noqa: F403
from tests.lookup.prompt_sections import prompt_section_payload
from tests.testkit.question_contract_provider import (
    ProviderQuestionInputOwnership,
    provider_answer_population,
    provider_question_input_ownership,
)


def _question_contract_decision(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_basis": "The current wording supports the selected outcome.",
        "outcome": outcome,
    }


def _without_empty_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value}


def _answer_subject_payload(subject: str) -> dict[str, object]:
    return {
        "subject_text": subject,
        "instance_interpretation": {"kind": "NORMAL_BUSINESS_INSTANCE"},
    }


@dataclass(frozen=True)
class ReadEligibilityRetentionSpec:
    requested_fact_id: str
    read_id: str
    source_candidate_id: str = ""
    row_path_ids: tuple[str, ...] = ()
    answer_value_fields: tuple[str, ...] = ()
    measured_value_fields: tuple[str, ...] = ()
    group_key_fields: tuple[str, ...] = ()
    population_scope_fields: tuple[str, ...] = ()
    known_input_resolver_results: tuple[tuple[str, str], ...] = ()


def read_eligibility_response_from_prompt(
    prompt: str,
    *,
    retention_specs: tuple[ReadEligibilityRetentionSpec, ...],
) -> dict[str, Any]:
    return {
        "answer": json.dumps(
            {
                "tool": "submit_read_eligibility",
                "arguments": read_eligibility_payload_from_prompt(
                    prompt,
                    retention_specs=retention_specs,
                ),
            },
            default=str,
        ),
        "usage": {
            "inputTokens": 1,
            "outputTokens": 1,
            "thinkingTokens": 0,
            "costUsd": 0,
        },
    }


def read_eligibility_response_from_fact_plan(
    prompt: str,
    fact_plan_payload: dict[str, Any],
) -> dict[str, Any]:
    return read_eligibility_response_from_prompt(
        prompt,
        retention_specs=_read_eligibility_retention_specs_from_fact_plan(
            fact_plan_payload
        ),
    )


def read_eligibility_response_for_retained_fields(
    prompt: str,
    *,
    requested_fact_id: str = "fact_1",
    read_id: str = "",
    source_candidate_id: str = "",
    row_path_ids: tuple[str, ...] = (),
    answer_value_fields: tuple[str, ...] = (),
    measured_value_fields: tuple[str, ...] = (),
    group_key_fields: tuple[str, ...] = (),
    population_scope_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not (read_id or source_candidate_id):
        read_id, source_candidate_id = _unique_read_candidate_for_declared_retention(
            prompt,
            requested_fact_id=requested_fact_id,
            field_identifiers=(
                *answer_value_fields,
                *measured_value_fields,
                *group_key_fields,
                *population_scope_fields,
                *row_path_ids,
            ),
        )
    return read_eligibility_response_from_prompt(
        prompt,
        retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id=requested_fact_id,
                read_id=read_id,
                source_candidate_id=source_candidate_id,
                row_path_ids=row_path_ids,
                answer_value_fields=answer_value_fields,
                measured_value_fields=measured_value_fields,
                group_key_fields=group_key_fields,
                population_scope_fields=population_scope_fields,
            ),
        ),
    )


def _unique_read_candidate_for_declared_retention(
    prompt: str,
    *,
    requested_fact_id: str,
    field_identifiers: tuple[str, ...],
) -> tuple[str, str]:
    candidate_groups = _read_eligibility_prompt_json_section(
        prompt,
        label="Candidate API reads",
    )["requested_fact_read_candidates"]
    matches: list[dict[str, Any]] = []
    for group in candidate_groups:
        if str(group.get("requested_fact_id") or "") != requested_fact_id:
            continue
        for card in group.get("read_candidates") or ():
            if not isinstance(card, dict):
                continue
            try:
                for identifier in field_identifiers:
                    if identifier:
                        _resolve_field_path(card, identifier)
            except AssertionError:
                continue
            matches.append(card)
    if len(matches) != 1:
        raise AssertionError(
            "read eligibility fixture declared fields must resolve to exactly "
            f"one candidate; matched {len(matches)}"
        )
    return str(matches[0].get("read_id") or ""), str(
        matches[0].get("source_candidate_id") or ""
    )


def read_eligibility_payload_from_prompt(
    prompt: str,
    *,
    retention_specs: tuple[ReadEligibilityRetentionSpec, ...],
) -> dict[str, Any]:
    candidate_groups = _read_eligibility_prompt_json_section(
        prompt,
        label="Candidate API reads",
    )["requested_fact_read_candidates"]
    retention_specs_by_fact: dict[str, list[ReadEligibilityRetentionSpec]] = (
        defaultdict(list)
    )
    for spec in retention_specs:
        retention_specs_by_fact[
            _canonical_read_eligibility_requested_fact_id(
                prompt,
                requested_fact_id=spec.requested_fact_id,
            )
        ].append(spec)
    requested_fact_assessments: dict[str, dict[str, Any]] = {}
    for group in candidate_groups:
        requested_fact_id = group["requested_fact_id"]
        fact_retention_specs = tuple(retention_specs_by_fact.get(requested_fact_id, ()))
        canonical_inputs = _canonical_inputs_for_fact(
            group,
            specs=fact_retention_specs,
        )
        cards_by_source_id = {
            str(card.get("source_candidate_id") or ""): card
            for card in group.get("read_candidates") or ()
            if isinstance(card, dict)
        }
        cards_by_read_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for card in group.get("read_candidates") or ():
            if isinstance(card, dict):
                cards_by_read_id[str(card.get("read_id") or "")].append(card)
        retention_specs_by_source_id: dict[str, list[ReadEligibilityRetentionSpec]] = (
            defaultdict(list)
        )
        for spec in fact_retention_specs:
            card = _read_eligibility_card_for_retention_spec(
                spec,
                cards_by_source_id=cards_by_source_id,
                cards_by_read_id=cards_by_read_id,
            )
            retention_specs_by_source_id[str(card["source_candidate_id"])].append(spec)
        read_candidate_reviews: dict[str, dict[str, Any]] = {}
        for card in group.get("read_candidates") or ():
            if not isinstance(card, dict):
                continue
            source_candidate_id = str(card.get("source_candidate_id") or "")
            if source_candidate_id in retention_specs_by_source_id:
                merged_spec = _merged_read_eligibility_retention_spec(
                    requested_fact_id=requested_fact_id,
                    read_id=str(card.get("read_id") or ""),
                    source_candidate_id=source_candidate_id,
                    specs=tuple(retention_specs_by_source_id[source_candidate_id]),
                )
                read_candidate_reviews[source_candidate_id] = (
                    _retention_review_payload_from_spec(
                        merged_spec,
                        card=card,
                    )
                )
                continue
            read_candidate_reviews[source_candidate_id] = {
                "retention_basis": (
                    "This test fixture did not declare this read candidate "
                    "as retained for the requested fact."
                ),
                "retention_decision": "DROP",
            }
        requested_fact_assessments[requested_fact_id] = {
            "read_candidate_reviews": read_candidate_reviews,
            "canonical_inputs": {
                binding["known_input_id"]: {
                    key: value
                    for key, value in binding.items()
                    if key not in {"known_input_id", "canonical_result"}
                }
                for binding in canonical_inputs
            },
        }
    return {
        "requested_fact_assessments": requested_fact_assessments,
    }


def _cards_by_read_id(
    cards: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        output[str(card.get("read_id") or "")].append(card)
    return output


def _canonical_read_eligibility_requested_fact_id(
    prompt: str,
    *,
    requested_fact_id: str,
) -> str:
    try:
        requested_facts = (
            _read_eligibility_prompt_json_section(
                prompt,
                label="Requested facts",
            ).get("requested_facts")
            or ()
        )
    except (AssertionError, ValueError):
        return requested_fact_id
    if len(requested_facts) == 1 and isinstance(requested_facts[0], dict):
        return str(requested_facts[0].get("requested_fact_id") or requested_fact_id)
    for fact in requested_facts:
        if not isinstance(fact, dict):
            continue
        if str(fact.get("requested_fact_id") or "") == requested_fact_id:
            return requested_fact_id
        if str(fact.get("evidence_ref") or "") == f"requested_fact:{requested_fact_id}":
            return str(fact.get("requested_fact_id") or requested_fact_id)
    return requested_fact_id


def _merged_read_eligibility_retention_spec(
    *,
    requested_fact_id: str,
    read_id: str,
    source_candidate_id: str,
    specs: tuple[ReadEligibilityRetentionSpec, ...],
) -> ReadEligibilityRetentionSpec:
    return ReadEligibilityRetentionSpec(
        requested_fact_id=requested_fact_id,
        read_id=read_id,
        source_candidate_id=source_candidate_id,
        row_path_ids=_unique(
            row_path_id for spec in specs for row_path_id in spec.row_path_ids
        ),
        answer_value_fields=_unique(
            field for spec in specs for field in spec.answer_value_fields
        ),
        measured_value_fields=_unique(
            field for spec in specs for field in spec.measured_value_fields
        ),
        group_key_fields=_unique(
            field for spec in specs for field in spec.group_key_fields
        ),
        population_scope_fields=_unique(
            field for spec in specs for field in spec.population_scope_fields
        ),
        known_input_resolver_results=tuple(
            dict.fromkeys(
                binding
                for spec in specs
                for binding in spec.known_input_resolver_results
            )
        ),
    )


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _read_eligibility_retention_specs_from_fact_plan(
    fact_plan_payload: dict[str, Any],
) -> tuple[ReadEligibilityRetentionSpec, ...]:
    outcome = fact_plan_payload.get("outcome")
    if not isinstance(outcome, dict):
        return ()
    specs: list[ReadEligibilityRetentionSpec] = []
    for answer in outcome.get("answers") or ():
        if not isinstance(answer, dict):
            continue
        requested_fact_id = str(answer.get("requested_fact_id") or "")
        pattern = str(answer.get("pattern") or "")
        if pattern == "set_difference":
            specs.extend(
                _read_eligibility_retention_specs_from_set_difference_answer(
                    answer,
                    requested_fact_id=requested_fact_id,
                )
            )
            continue
        if pattern == "joined_rows":
            specs.extend(
                _read_eligibility_retention_specs_from_joined_rows_answer(
                    answer,
                    requested_fact_id=requested_fact_id,
                )
            )
            continue
        source = answer.get("source") or {}
        if not isinstance(source, dict) or source.get("kind") != "read":
            continue
        read_id = str(source.get("read_id") or "")
        if not read_id:
            continue
        output_fields = tuple(
            str(field.get("field_id") or "")
            for field in answer.get("output_fields") or ()
            if isinstance(field, dict) and str(field.get("field_id") or "")
        )
        output_field = answer.get("output_field")
        if isinstance(output_field, dict) and str(output_field.get("field_id") or ""):
            output_fields = (*output_fields, str(output_field["field_id"]))
        group_fields = tuple(
            str(field.get("field_id") or "")
            for field in answer.get("group_fields") or ()
            if isinstance(field, dict) and str(field.get("field_id") or "")
        )
        metric = answer.get("metric") or {}
        metric_field = (
            str(metric.get("field_id") or "") if isinstance(metric, dict) else ""
        )
        measured_fields = (
            (metric_field,)
            if isinstance(metric, dict)
            and metric.get("kind") == "aggregate_field"
            and metric_field
            else ()
        )
        row_path_fields = _row_path_fields_for_answer(
            pattern=pattern,
            output_fields=output_fields,
            measured_fields=measured_fields,
            group_fields=group_fields,
        )
        specs.append(
            ReadEligibilityRetentionSpec(
                requested_fact_id=requested_fact_id,
                read_id=read_id,
                row_path_ids=row_path_fields,
                answer_value_fields=output_fields,
                measured_value_fields=measured_fields,
                group_key_fields=group_fields,
            )
        )
    return tuple(specs)


def _row_path_fields_for_answer(
    *,
    pattern: str,
    output_fields: tuple[str, ...],
    measured_fields: tuple[str, ...],
    group_fields: tuple[str, ...],
) -> tuple[str, ...]:
    if pattern not in {
        "list_rows",
        "grouped_rows",
        "aggregate_scalar",
        "aggregate_by_group",
        "aggregate_by_group",
    }:
        return ()
    return _unique(
        (
            *output_fields,
            *measured_fields,
            *group_fields,
        )
    )


def _read_eligibility_retention_specs_from_set_difference_answer(
    answer: dict[str, Any],
    *,
    requested_fact_id: str,
) -> tuple[ReadEligibilityRetentionSpec, ...]:
    candidate = answer.get("candidate") or {}
    observed = answer.get("observed") or {}
    candidate_source = candidate.get("source") or {}
    observed_source = observed.get("source") or {}
    specs: list[ReadEligibilityRetentionSpec] = []
    if isinstance(candidate_source, dict) and candidate_source.get("kind") == "read":
        candidate_identity_fields = tuple(candidate.get("identity_fields") or ())
        specs.append(
            ReadEligibilityRetentionSpec(
                requested_fact_id=requested_fact_id,
                read_id=str(candidate_source.get("read_id") or ""),
                population_scope_fields=candidate_identity_fields,
                answer_value_fields=tuple(
                    str(field.get("field_id") or "")
                    for field in candidate.get("output_fields") or ()
                    if isinstance(field, dict) and str(field.get("field_id") or "")
                ),
            )
        )
    if isinstance(observed_source, dict) and observed_source.get("kind") == "read":
        observed_identity_fields = tuple(observed.get("identity_fields") or ())
        specs.append(
            ReadEligibilityRetentionSpec(
                requested_fact_id=requested_fact_id,
                read_id=str(observed_source.get("read_id") or ""),
                population_scope_fields=observed_identity_fields,
            )
        )
    return tuple(specs)


def _read_eligibility_retention_specs_from_joined_rows_answer(
    answer: dict[str, Any],
    *,
    requested_fact_id: str,
) -> tuple[ReadEligibilityRetentionSpec, ...]:
    output_fields_by_side: dict[str, list[str]] = defaultdict(list)
    for field in answer.get("output_fields") or ():
        if not isinstance(field, dict):
            continue
        side = str(field.get("side") or "")
        field_id = str(field.get("field_id") or "")
        if side and field_id:
            output_fields_by_side[side].append(field_id)
    specs: list[ReadEligibilityRetentionSpec] = []
    for side in ("left", "right"):
        relation = answer.get(side) or {}
        source = relation.get("source") or {}
        if not isinstance(source, dict) or source.get("kind") != "read":
            continue
        field_ids = tuple(
            str(field.get("field_id") or "")
            for field in relation.get("fields") or ()
            if isinstance(field, dict) and str(field.get("field_id") or "")
        )
        output_field_ids = tuple(output_fields_by_side.get(side) or ())
        join_field_ids = tuple(
            str(key.get(f"{side}_field_id") or "")
            for key in answer.get("join_keys") or ()
            if isinstance(key, dict) and str(key.get(f"{side}_field_id") or "")
        )
        specs.append(
            ReadEligibilityRetentionSpec(
                requested_fact_id=requested_fact_id,
                read_id=str(source.get("read_id") or ""),
                row_path_ids=join_field_ids or field_ids,
                answer_value_fields=output_field_ids,
            )
        )
    return tuple(specs)


def _read_eligibility_card_for_retention_spec(
    spec: ReadEligibilityRetentionSpec,
    *,
    cards_by_source_id: dict[str, dict[str, Any]],
    cards_by_read_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if spec.source_candidate_id:
        card = cards_by_source_id.get(spec.source_candidate_id)
        if card is None:
            raise AssertionError(
                f"read eligibility fixture source candidate not shown: "
                f"{spec.source_candidate_id}"
            )
        if str(card.get("read_id") or "") != spec.read_id:
            raise AssertionError(
                "read eligibility fixture source candidate/read mismatch: "
                f"{spec.source_candidate_id} != {spec.read_id}"
            )
        return card
    cards = cards_by_read_id.get(spec.read_id, [])
    if len(cards) > 1:
        identifiers = (
            *spec.row_path_ids,
            *spec.answer_value_fields,
            *spec.measured_value_fields,
            *spec.group_key_fields,
            *spec.population_scope_fields,
        )
        matching_cards = tuple(
            card
            for card in cards
            if _read_eligibility_card_has_identifiers(card, identifiers)
        )
        if len(matching_cards) == 1:
            return matching_cards[0]
    if len(cards) != 1:
        raise AssertionError(
            f"read eligibility fixture read_id must resolve to exactly one candidate: "
            f"{spec.read_id}"
        )
    return cards[0]


def _read_eligibility_card_has_identifiers(
    card: dict[str, Any],
    identifiers: tuple[str, ...],
) -> bool:
    for identifier in identifiers:
        if not identifier:
            continue
        try:
            _resolve_row_path_id(card, identifier)
        except AssertionError:
            try:
                _resolve_field_path(card, identifier)
            except AssertionError:
                return False
    return True


def _retention_review_payload_from_spec(
    spec: ReadEligibilityRetentionSpec,
    *,
    card: dict[str, Any],
) -> dict[str, Any]:
    field_tokens_by_path = {
        str(field.get("path") or ""): str(field.get("evidence_token") or "")
        for field in _response_row_fields(card)
    }
    row_tokens_by_id = {
        _response_row_id(row): str(row.get("evidence_token") or "")
        for row in _response_rows(card)
    }
    row_tokens: list[str] = []
    field_tokens: list[str] = []

    for row_identifier in spec.row_path_ids:
        row_path_id = _resolve_row_path_id(card, row_identifier)
        token = row_tokens_by_id.get(row_path_id, "")
        if not token:
            raise AssertionError(
                f"read eligibility fixture row path missing token: {row_path_id}"
            )
        row_tokens.append(token)

    for identifiers in (
        spec.answer_value_fields,
        spec.measured_value_fields,
        spec.group_key_fields,
        spec.population_scope_fields,
    ):
        for identifier in identifiers:
            field_path = _resolve_field_path(card, identifier)
            token = field_tokens_by_path.get(field_path, "")
            if not token:
                raise AssertionError(
                    f"read eligibility fixture field missing token: {field_path}"
                )
            field_tokens.append(token)
    row_tokens = list(dict.fromkeys(row_tokens))
    field_tokens = list(dict.fromkeys(field_tokens))
    return {
        "relevant_row_path_tokens": row_tokens,
        "relevant_field_tokens": field_tokens,
        "retention_basis": (
            "This fixture retained the read because it exposes row or field "
            "evidence that may be useful for the requested fact."
        ),
        "retention_decision": "RETAIN",
    }


def _canonical_inputs_for_fact(
    group: dict[str, Any],
    *,
    specs: tuple[ReadEligibilityRetentionSpec, ...],
) -> list[dict[str, Any]]:
    requested_results = dict(
        binding for spec in specs for binding in spec.known_input_resolver_results
    )
    selected: list[dict[str, str]] = []
    for known_input in group.get("known_inputs") or ():
        if not isinstance(known_input, dict):
            continue
        options = tuple(
            option
            for option in known_input.get("canonical_options") or ()
            if isinstance(option, dict)
        )
        if not options:
            continue
        known_input_id = str(known_input.get("id") or "")
        requested_result = requested_results.get(known_input_id, "")
        matches = tuple(
            option
            for option in options
            if not requested_result
            or str(option.get("result") or "") == requested_result
        )
        if len(matches) != 1:
            raise AssertionError(
                "fixture canonical selection must match one shown option"
            )
        option = matches[0]
        canonical_option_assessments = {
            str(candidate["id"]): (
                f"{candidate['result']}: this fixture assessed the shown "
                "canonical option against the retained read evidence."
            )
            for candidate in options
        }
        selection: dict[str, Any] = {
            "known_input_id": known_input_id,
            "interpretation_question": str(
                known_input.get("interpretation_question") or ""
            ),
            "canonical_option_assessments": canonical_option_assessments,
            "because": (
                "This fixture selects the declared canonical result for "
                "the named input in the requested fact."
            ),
            "canonical_option_id": str(option.get("id") or ""),
            "canonical_result": str(option.get("result") or ""),
        }
        resolver_option_ids = tuple(option.get("resolver_option_ids") or ())
        selection["resolver_option_assessments"] = {
            str(option_id): (
                "This fixture assessed the shown lookup request parameters and "
                "returned identity verification fields for this resolver route."
            )
            for option_id in resolver_option_ids
        }
        if resolver_option_ids:
            selection["resolver_option_id"] = str(resolver_option_ids[0])
        selected.append(selection)
    return selected


def _resolve_field_path(card: dict[str, Any], identifier: str) -> str:
    matches = tuple(
        str(field.get("path") or "")
        for field in _response_row_fields(card)
        if _field_matches_identifier(field, identifier)
    )
    unique_matches = tuple(dict.fromkeys(path for path in matches if path))
    if len(unique_matches) != 1:
        raise AssertionError(
            f"read eligibility fixture field must resolve to exactly one path: "
            f"{identifier}"
        )
    return unique_matches[0]


def _field_matches_identifier(field: dict[str, Any], identifier: str) -> bool:
    field_path = str(field.get("path") or "")
    field_id = str(field.get("field_id") or "")
    return identifier in {
        field_path,
        field_id,
        field_path.rsplit(".", 1)[-1],
    }


def _resolve_row_path_id(card: dict[str, Any], identifier: str) -> str:
    row_matches = tuple(
        _response_row_id(row)
        for row in _response_rows(card)
        if identifier
        in {
            _response_row_id(row),
            str(row.get("path") or ""),
        }
    )
    if row_matches:
        unique_row_matches = tuple(dict.fromkeys(row_matches))
        if len(unique_row_matches) == 1:
            return unique_row_matches[0]
    field_path = _resolve_field_path(card, identifier)
    row_path_id = _response_row_id_for_field_path(card, field_path)
    if not row_path_id:
        raise AssertionError(
            f"read eligibility fixture field has no row path: {identifier}"
        )
    return row_path_id


def _response_rows(card: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        row for row in card.get("response_rows") or () if isinstance(row, dict)
    )


def _response_row_fields(card: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        field
        for row in _response_rows(card)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    )


def _response_row_id(row: dict[str, Any]) -> str:
    token = str(row.get("evidence_token") or "")
    marker = ".row."
    if marker in token:
        return token.split(marker, 1)[1]
    return str(row.get("path") or "")


def _response_row_id_for_field_path(
    card: dict[str, Any],
    field_path: str,
) -> str:
    for row in _response_rows(card):
        if any(
            isinstance(field, dict) and str(field.get("path") or "") == field_path
            for field in row.get("fields") or ()
        ):
            return _response_row_id(row)
    return ""


def _read_eligibility_prompt_json_section(
    prompt: str,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        return prompt_section_payload(prompt, label)
    except (AssertionError, ValueError):
        context = _read_eligibility_xml_context(prompt)
        if label == "Candidate API reads":
            return {
                "requested_fact_read_candidates": [
                    {
                        "requested_fact_id": fact.attrib.get("id", ""),
                        "known_inputs": [
                            _read_eligibility_xml_known_input(known_input)
                            for known_input in fact.findall(
                                "./known_inputs/known_input"
                            )
                        ],
                        "read_candidates": [
                            _read_eligibility_xml_card(card)
                            for card in fact.findall("./candidate_api_reads/api_read")
                        ],
                    }
                    for fact in context.findall("./requested_fact")
                ]
            }
        if label == "Requested facts":
            return {
                "requested_facts": [
                    {"requested_fact_id": fact.attrib.get("id", "")}
                    for fact in context.findall("./requested_fact")
                ]
            }
        raise


def _read_eligibility_xml_context(prompt: str) -> ElementTree.Element:
    start = prompt.index("<read_eligibility_context>")
    end_tag = "</read_eligibility_context>"
    end = prompt.index(end_tag, start) + len(end_tag)
    return ElementTree.fromstring(prompt[start:end])


def _read_eligibility_xml_card(card: ElementTree.Element) -> dict[str, Any]:
    return {
        "source_candidate_id": card.attrib.get("id", ""),
        "read_id": card.attrib.get("read", ""),
        "endpoint_name": card.attrib.get("endpoint", ""),
        "row_source_id": card.attrib.get("row_source", ""),
        "docstring": card.findtext("./description", default=""),
        "canonical_targets": [
            {
                "id": target.attrib.get("id", ""),
                "known_input": target.attrib.get("known_input", ""),
                "accepts": target.attrib.get("accepts", ""),
            }
            for target in card.findall("./canonical_targets/canonical_target")
        ],
        "response_rows": [
            row
            for element in card.findall("./response/row")
            for row in _read_eligibility_xml_rows(element)
        ],
    }


def _read_eligibility_xml_known_input(
    known_input: ElementTree.Element,
) -> dict[str, Any]:
    return {
        "id": known_input.attrib.get("id", ""),
        "interpretation_question": known_input.findtext(
            "./interpretation_question",
            default="",
        ),
        "canonical_options": [
            {
                "id": option.attrib.get("id", ""),
                "result": option.attrib.get("result", ""),
                "resolver_option_ids": [
                    resolver.attrib.get("option_id", "")
                    for resolver in option.findall("./resolver")
                ],
            }
            for option in known_input.findall("./canonical_options/canonical_option")
        ],
    }


def _read_eligibility_xml_rows(
    element: ElementTree.Element,
) -> list[dict[str, Any]]:
    row = {
        "path": element.attrib.get("path", ""),
        "cardinality": element.attrib.get("cardinality", ""),
        "evidence_token": element.attrib.get("evidence_token", ""),
        "fields": [
            {
                "field_id": field.attrib.get("name", ""),
                "path": field.attrib.get("path", ""),
                "type": field.attrib.get("type", ""),
                "evidence_token": field.attrib.get("evidence_token", ""),
            }
            for field in element.findall("./field")
        ],
    }
    children = [
        child
        for element_child in element.findall("./row")
        for child in _read_eligibility_xml_rows(element_child)
    ]
    return [row, *children]


def _planner_prompt_json_section(prompt: str, *, label: str) -> dict[str, Any]:
    return _read_eligibility_prompt_json_section(prompt, label=label)


def _offered_conversation_resolution_tool_names(
    tool_specs: tuple[Any, ...],
) -> tuple[str, ...]:
    return tuple(
        tool.name
        for tool in tool_specs
        if tool.name in CONVERSATION_RESOLUTION_TOOL_NAMES
    )


def _select_conversation_resolution_tool_name(
    tool_specs: tuple[Any, ...],
    *,
    responses: dict[str, dict[str, Any]] | None = None,
) -> str:
    offered = _offered_conversation_resolution_tool_names(tool_specs)
    if not offered:
        return ""
    for name in offered:
        if responses and name in responses:
            return name
    return offered[0]


@dataclass(frozen=True)
class _QuestionContractIdMap:
    requested_fact_ids: dict[str, str]
    answer_output_ids: dict[tuple[str, str], str]
    known_input_ids: dict[str, str]


def _question_contract_id_map(contract: QuestionContract) -> _QuestionContractIdMap:
    requested_fact_ids: dict[str, str] = {}
    answer_output_ids: dict[tuple[str, str], str] = {}
    known_input_ids: dict[str, str] = {}
    for fact_index, fact in enumerate(contract.requested_facts, start=1):
        canonical_fact_id = f"fact_{fact_index}"
        requested_fact_ids[fact.id] = canonical_fact_id
        for output_index, output in enumerate(fact.answer_outputs, start=1):
            answer_output_ids[(fact.id, output.id)] = f"answer_{output_index}"
        counters = {
            "entity": 0,
            "time": 0,
            "limit": 0,
        }
        for known in fact.known_inputs:
            if known.id in known_input_ids:
                continue
            label = _known_input_label(known)
            if not label:
                continue
            counters[label] += 1
            known_input_ids[known.id] = f"{canonical_fact_id}_{label}_{counters[label]}"
    return _QuestionContractIdMap(
        requested_fact_ids=requested_fact_ids,
        answer_output_ids=answer_output_ids,
        known_input_ids=known_input_ids,
    )


def _question_contract_for_arguments(
    arguments: dict[str, Any],
    *,
    description: str = "",
) -> QuestionContract:
    outcome = arguments.get("outcome")
    if not isinstance(outcome, dict):
        return _default_question_contract(description=description)

    outputs_by_fact: dict[str, list[str]] = defaultdict(list)
    for answer in outcome.get("answers") or ():
        if not isinstance(answer, dict):
            continue
        fact_id = str(answer.get("requested_fact_id") or "").strip()
        if not fact_id:
            continue
        output_ids = answer.get("answer_output_ids") or ()
        for output_id in output_ids:
            output_text = str(output_id or "").strip()
            if output_text and output_text not in outputs_by_fact[fact_id]:
                outputs_by_fact[fact_id].append(output_text)
        outputs_by_fact.setdefault(fact_id, [])

    for fulfillment in outcome.get("fulfillment") or ():
        if not isinstance(fulfillment, dict):
            continue
        fact_id = str(fulfillment.get("requested_fact_id") or "").strip()
        output_id = str(fulfillment.get("answer_output_id") or "").strip()
        if not fact_id:
            continue
        if output_id and output_id not in outputs_by_fact[fact_id]:
            outputs_by_fact[fact_id].append(output_id)
        else:
            outputs_by_fact.setdefault(fact_id, [])

    for blocked in outcome.get("blocked_facts") or ():
        if isinstance(blocked, dict):
            fact_id = str(blocked.get("requested_fact_id") or "").strip()
            if fact_id:
                outputs_by_fact.setdefault(fact_id, [])

    for missing in outcome.get("missing_catalog_inputs") or ():
        if isinstance(missing, dict):
            fact_id = str(missing.get("requested_fact_id") or "").strip()
            if fact_id:
                outputs_by_fact.setdefault(fact_id, [])

    if not outputs_by_fact:
        return _default_question_contract(description=description)

    return QuestionContract(
        requested_facts=tuple(
            RequestedFact(
                id=fact_id,
                description=description or fact_id,
                answer_expression=_requested_fact_answer_expression(
                    RequestedFactAnswerExpressionFamily(
                        _answer_expression_family_by_fact_id_from_fact_plan(
                            arguments
                        ).get(fact_id, "scalar_aggregate")
                    ),
                ),
                answer_subject=RequestedFactAnswerSubject(
                    subject_text=description or fact_id
                ),
                answer_outputs=tuple(
                    RequestedFactAnswerOutput(id=output_id, role="ANSWER_VALUE")
                    for output_id in (output_ids or ["answer"])
                ),
            )
            for fact_id, output_ids in outputs_by_fact.items()
        )
    )


def _question_contract_with_answer_expression_from_fact_plan(
    contract: QuestionContract,
    fact_plan: dict[str, Any],
) -> QuestionContract:
    family_by_fact_id = _answer_expression_family_by_fact_id_from_fact_plan(fact_plan)
    if not family_by_fact_id:
        return contract
    requested_facts = tuple(
        (
            replace(
                fact,
                answer_expression=_requested_fact_answer_expression(
                    RequestedFactAnswerExpressionFamily(family),
                    known_inputs=fact.known_inputs,
                ),
            )
            if (
                family := family_by_fact_id.get(fact.id)
                or family_by_fact_id.get(f"fact_{index}")
            )
            else fact
        )
        for index, fact in enumerate(contract.requested_facts, start=1)
    )
    if requested_facts == contract.requested_facts:
        return contract
    return replace(contract, requested_facts=requested_facts)


def _question_contract_payload(
    contract: QuestionContract,
) -> dict[str, Any]:
    id_map = _question_contract_id_map(contract)
    payload = {
        "kind": "question_contract",
        "answer_requests_count": len(contract.requested_facts),
        "question_inputs": _question_inputs_payload(contract, id_map=id_map),
        "answer_requests": [
            _question_contract_answer_request_payload(
                fact,
                id_map=id_map,
            )
            for fact in contract.requested_facts
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }
    return payload


def _question_contract_answer_request_payload(
    fact: RequestedFact,
    *,
    id_map: _QuestionContractIdMap,
) -> dict[str, Any]:
    subject_text = (
        fact.answer_subject.subject_text
        if fact.answer_subject is not None
        else fact.description
    )
    ownership = _provider_question_input_ownership(fact, id_map=id_map)
    payload = {
        "answer_fact": fact.description,
        "answer_expression": _answer_expression_payload(fact),
        "answer_subject": _answer_subject_payload(subject_text),
        "answer_population": _answer_population_payload(
            fact,
            subject_text=subject_text,
            ownership=ownership,
        ),
        "answer_outputs": [
            output.to_answer_request_dict() for output in fact.answer_outputs
        ],
        "question_input_uses": list(ownership.question_input_uses),
    }
    return payload


def _provider_question_input_ownership(
    fact: RequestedFact,
    *,
    id_map: _QuestionContractIdMap,
) -> ProviderQuestionInputOwnership:
    def provider_ref(input_ref: str) -> str:
        return id_map.known_input_ids.get(input_ref, input_ref)

    expression = fact.answer_expression
    group_refs = (
        tuple(provider_ref(ref) for ref in expression.group_key.question_input_refs)
        if expression is not None and expression.group_key is not None
        else ()
    )
    population_refs = {
        test.id: tuple(provider_ref(ref) for ref in test.owned_question_input_refs)
        for test in (
            fact.answer_population.membership_tests
            if fact.answer_population is not None
            else ()
        )
        if test.owned_question_input_refs
    }
    result_limit_ref = (
        provider_ref(expression.limit_input_ref)
        if expression is not None and expression.limit_input_ref
        else ""
    )
    owned_refs = {
        *group_refs,
        *(ref for refs in population_refs.values() for ref in refs),
        *([result_limit_ref] if result_limit_ref else []),
    }
    unowned_refs = tuple(
        provider_ref(known.id)
        for known in fact.known_inputs
        if known.id in id_map.known_input_ids
        and provider_ref(known.id) not in owned_refs
    )
    if unowned_refs:
        raise ValueError(
            "test Question Contract fixture has unowned inputs: "
            + ", ".join(unowned_refs)
        )
    return provider_question_input_ownership(
        group_key_input_refs=group_refs,
        population_input_refs_by_test_id=population_refs,
        result_limit_input_ref=result_limit_ref,
    )


def _answer_expression_payload(
    fact: RequestedFact,
) -> dict[str, Any]:
    expression = fact.answer_expression
    if expression is None:
        return {"family": "scalar_aggregate"}
    payload: dict[str, Any] = {"family": expression.family.value}
    if expression.group_key is not None:
        payload["group_key"] = expression.group_key.to_answer_request_dict()
    if expression.ordering_direction is not None:
        payload["ordering"] = {
            "basis": expression.ordering_basis,
            "direction": expression.ordering_direction.value,
        }
    if expression.selection_kind is not None:
        payload["selection"] = {"kind": expression.selection_kind.value}
    return payload


def _question_inputs_payload(
    contract: QuestionContract,
    *,
    id_map: _QuestionContractIdMap,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in contract.requested_facts:
        for known in fact.known_inputs:
            input_ref = id_map.known_input_ids.get(known.id)
            if not input_ref or input_ref in seen:
                continue
            seen.add(input_ref)
            output.append(_known_input_payload(known, input_ref=input_ref))
    return output


def _known_input_payload(
    known: RequestedFactKnownInput,
    *,
    input_ref: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ref": input_ref,
        "source": known.source.value,
        "kind": known.kind.value,
        "inventory_check": {
            "why_this_is_an_input": f"{known.text} is a declared question input"
        },
    }
    if known.kind == KnownInputKind.ROW_SET_REFERENCE:
        payload["reference_text"] = known.text
        payload["occurrence"] = known.occurrence
        payload["resolved_input_ref"] = known.resolved_input_ref
        return payload
    payload["value_source_text"] = known.text
    payload["role"] = known.role.value if known.role else ""
    payload["operand_text"] = known.resolved_value_text
    if known.field_label_text:
        payload["field_label_text"] = known.field_label_text
    if known.value_meaning_hint:
        payload["value_meaning_hint"] = known.value_meaning_hint
    return payload


def _known_input_label(known: RequestedFactKnownInput) -> str:
    if known.is_reference_value:
        return "entity"
    if known.is_time_value:
        return "time"
    if known.is_result_limit:
        return "limit"
    return ""


def _question_contract_response(
    *,
    subject: str = "answer",
    answer_subject: str | None = None,
    answer_expression_family: str = "scalar_aggregate",
    parts: tuple[str, ...] = ("answer",),
    answer_output_role: str | None = None,
    answer_output_roles: tuple[str, ...] | None = None,
    demand_text: str = "How",
    question_inputs: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    del demand_text
    input_payloads: list[dict[str, Any]] = []
    input_refs: list[str] = []
    counters = {
        "entity": 0,
        "time": 0,
        "number": 0,
        "limit": 0,
        "input": 0,
    }
    for item in question_inputs:
        input_ref = _question_input_ref_for_response_item(
            item,
            counters=counters,
        )
        input_payloads.append(
            _question_input_from_response_item(
                item,
                input_ref,
            )
        )
        input_refs.append(input_ref)
    default_output_role = answer_output_role or _answer_output_role_for_family(
        answer_expression_family
    )
    output_roles = answer_output_roles or tuple(default_output_role for _ in parts)
    if len(output_roles) != len(parts):
        raise ValueError("answer output roles must match answer outputs")
    answer_expression: dict[str, Any] = {"family": answer_expression_family}
    input_refs = [str(item["input_ref"]) for item in input_payloads]
    result_limit_refs = [
        str(item["input_ref"])
        for item in input_payloads
        if item.get("role") == LiteralInputRole.RESULT_LIMIT.value
    ]
    if answer_expression_family == "list_rows":
        answer_expression["selection"] = {"kind": "all_results"}
    population_refs = tuple(ref for ref in input_refs if ref not in result_limit_refs)
    ownership = provider_question_input_ownership(
        population_input_refs_by_test_id=(
            {
                f"input_constraint_{index}": (input_ref,)
                for index, input_ref in enumerate(population_refs, start=1)
            }
        ),
        result_limit_input_ref=result_limit_refs[0] if result_limit_refs else "",
    )
    answer_request = {
        "answer_fact": subject,
        "answer_expression": answer_expression,
        "answer_subject": _answer_subject_payload(answer_subject or subject),
        "answer_population": _provider_answer_population_payload(
            _answer_population_payload_from_text(
                description=subject,
                subject_text=answer_subject or subject,
            ),
            ownership=ownership,
        ),
        "answer_outputs": [
            {"description": part, "role": role}
            for part, role in zip(parts, output_roles, strict=True)
        ],
        "question_input_uses": list(ownership.question_input_uses),
    }
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": input_payloads,
        "answer_requests": [answer_request],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }


def _answer_output_role_for_family(answer_expression_family: str) -> str:
    if answer_expression_family in {
        "scalar_aggregate",
        "grouped_aggregate",
        "computed_scalar",
    }:
        return "MEASURED_VALUE"
    return "ANSWER_VALUE"


def _answer_population_payload(
    fact: RequestedFact,
    *,
    subject_text: str,
    ownership: ProviderQuestionInputOwnership,
) -> dict[str, Any]:
    if fact.answer_population is not None:
        payload = fact.answer_population.to_question_contract_dict()
    else:
        instance_interpretation = (
            fact.answer_subject.instance_interpretation
            if fact.answer_subject is not None
            else RequestedFactAnswerSubject(
                subject_text=subject_text
            ).instance_interpretation
        )
        payload = default_answer_population(
            subject_text=subject_text,
            instance_interpretation=instance_interpretation,
        ).to_question_contract_dict()
    return _provider_answer_population_payload(payload, ownership=ownership)


def _provider_answer_population_payload(
    payload: dict[str, Any],
    *,
    ownership: ProviderQuestionInputOwnership,
) -> dict[str, Any]:
    return provider_answer_population(payload, ownership=ownership)


def _answer_population_payload_from_text(
    *,
    description: str,
    subject_text: str,
) -> dict[str, Any]:
    payload = default_answer_population(
        subject_text=subject_text,
        instance_interpretation=RequestedFactAnswerSubject(
            subject_text=subject_text
        ).instance_interpretation,
    ).to_question_contract_dict()
    return _provider_answer_population_payload(
        payload,
        ownership=provider_question_input_ownership(),
    )


def _question_input_ref_for_response_item(
    item: dict[str, Any],
    *,
    counters: dict[str, int],
) -> str:
    provided = str(item.get("input_ref") or "").strip()
    if provided:
        return provided
    resolved_kind = KnownInputKind(str(item["kind"]))
    resolved_role = (
        LiteralInputRole(str(item["role"]))
        if resolved_kind == KnownInputKind.LITERAL
        else None
    )
    resolved_label = {
        LiteralInputRole.REFERENCE_VALUE: "entity",
        LiteralInputRole.TIME_VALUE: "time",
        LiteralInputRole.RESULT_LIMIT: "limit",
    }.get(resolved_role, "input")
    if resolved_kind == KnownInputKind.ROW_SET_REFERENCE:
        resolved_label = "row_set"
    counters[resolved_label] = counters.get(resolved_label, 0) + 1
    index = counters[resolved_label]
    return f"fact_1_{resolved_label}_{index}"


def _question_input_from_response_item(
    item: dict[str, Any],
    input_ref: str,
) -> dict[str, Any]:
    kind = KnownInputKind(str(item["kind"]))
    source_text = str(
        item.get("value_source_text")
        or item.get("source_text")
        or item.get("reference_text")
        or ""
    )
    output: dict[str, Any] = {
        "input_ref": input_ref,
        "kind": kind.value,
        "source": str(
            item.get("source")
            or (
                "conversation_resolution"
                if kind == KnownInputKind.ROW_SET_REFERENCE
                else "question_context"
            )
        ),
        "inventory_check": {
            "why_this_is_an_input": f"{source_text} is a declared question input"
        },
    }
    if kind == KnownInputKind.ROW_SET_REFERENCE:
        output["reference_text"] = source_text
        output["occurrence"] = int(item.get("occurrence") or 1)
        output["resolved_input_ref"] = str(item["resolved_input_ref"])
        return output
    output["value_source_text"] = source_text
    role = LiteralInputRole(str(item["role"]))
    output["role"] = role.value
    output["operand_text"] = str(
        item.get("operand_text")
        or item.get("resolved_value_text")
        or item.get("value_text")
        or source_text
    )
    if item.get("resolved_input_ref"):
        output["resolved_input_ref"] = str(item["resolved_input_ref"])
    if item.get("field_label_text"):
        output["field_label_text"] = str(item["field_label_text"])
    if item.get("value_meaning_hint"):
        output["value_meaning_hint"] = str(item["value_meaning_hint"])
    return output


def _answer_expression_family_by_fact_id_from_fact_plan(
    fact_plan: dict[str, Any],
) -> dict[str, str]:
    outcome = fact_plan.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("kind") != "fact_plan":
        return {}
    output: dict[str, str] = {}
    for index, answer in enumerate(outcome.get("answers") or (), start=1):
        if not isinstance(answer, dict):
            continue
        requested_fact_id = str(answer.get("requested_fact_id") or f"fact_{index}")
        pattern = str(answer.get("pattern") or "")
        output[requested_fact_id] = _answer_expression_family_for_pattern(pattern)
    return output


def _answer_expression_family_for_pattern(pattern: str) -> str:
    if pattern in {"list_rows", "grouped_rows", "joined_rows"}:
        return "list_rows"
    if pattern == "direct_field_value":
        return "scalar_value"
    if pattern == "aggregate_scalar":
        return "scalar_aggregate"
    if pattern == "aggregate_by_group":
        return "grouped_aggregate"
    if pattern == "aggregate_by_group":
        return "grouped_aggregate"
    if pattern == "computed_scalar":
        return "computed_scalar"
    if pattern == "set_difference":
        return "set_difference"
    return "scalar_aggregate"


def _current_question_from_prompt(prompt: str) -> str:
    marker = "Current question:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def _conversation_resolution_payload_from_prompt(prompt: str) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    return {
        "kind": "conversation_resolution",
        "current_question_text": current_question,
        "outcome": {
            "kind": "resolved",
            "resolution_basis": "The current question is context-free.",
            "contextualized_question": current_question,
            "clauses": [
                {
                    "current_clause_text": current_question,
                    "occurrence": 1,
                    "resolved_text": current_question,
                    "retained_frame_parts": [],
                    "values": [],
                }
            ],
        },
    }


def _conversation_resolution_payload_using_memory(
    prompt: str,
    *,
    contextualized_question: str | None = None,
    actual_text: str,
    source_kind: str = "",
    retained_part_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    resolved_question = contextualized_question or current_question
    source = (
        _context_source_by_kind(prompt, source_kind)
        if source_kind
        else _first_context_source(prompt)
    )
    return _conversation_resolution_clause_payload(
        prompt=prompt,
        current_question=current_question,
        contextualized_question=resolved_question,
        actual_text=actual_text,
        selected_sources=(source,),
        retained_part_ids=retained_part_ids,
    )


def _conversation_resolution_payload_using_memories(
    prompt: str,
    *,
    contextualized_question: str,
    memories: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    actual_text = str(memories[0].get("actual_text") or "that") if memories else "that"
    selected_sources = _context_sources_from_prompt(prompt)
    return _conversation_resolution_clause_payload(
        prompt=prompt,
        current_question=current_question,
        contextualized_question=contextualized_question,
        actual_text=actual_text,
        selected_sources=selected_sources,
    )


def _conversation_resolution_payload_from_response(
    prompt: str,
    response: Any,
) -> dict[str, Any]:
    if response is None:
        return _conversation_resolution_payload_from_prompt(prompt)
    if callable(response):
        payload = response(prompt)
        if not isinstance(payload, dict):
            raise AssertionError("conversation-resolution builder must return dict")
        return payload
    if isinstance(response, dict):
        return dict(response)
    raise AssertionError("conversation-resolution response must be a dict or callable")


def _memory_kind_for_test_id(memory_id: str) -> str:
    if ".prior_request." in memory_id:
        return "prior_answer_request"
    if ".relation." in memory_id:
        return "row_set"
    if ".entity." in memory_id:
        return "entity_identity"
    if ".outcome." in memory_id:
        return "clarification_response"
    if ".value." in memory_id:
        return "scalar_value"
    return ""


def _context_sources_from_prompt(prompt: str) -> tuple[dict[str, Any], ...]:
    return _items_from_prompt_section(
        prompt,
        marker="Context sources:\n",
        key="context_sources",
    )


def _first_context_source(prompt: str) -> dict[str, Any]:
    sources = _context_sources_from_prompt(prompt)
    if not sources:
        return {
            "source_id": "current_question",
            "kind": "current_question",
            "text": _current_question_from_prompt(prompt),
        }
    return sources[0]


def _context_source_by_kind(prompt: str, kind: str) -> dict[str, Any]:
    for source in _context_sources_from_prompt(prompt):
        if source.get("kind") == kind:
            return source
    return _first_context_source(prompt)


def _items_from_prompt_section(
    prompt: str,
    *,
    marker: str,
    key: str,
) -> tuple[dict[str, Any], ...]:
    if marker not in prompt:
        return ()
    start = prompt.index(marker) + len(marker)
    end = prompt.find("\n\n", start)
    if end < 0:
        end = len(prompt)
    payload = json.loads(prompt[start:end])
    items = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return ()
    return tuple(item for item in items if isinstance(item, dict))


def _conversation_resolution_clause_payload(
    *,
    prompt: str,
    current_question: str,
    contextualized_question: str,
    actual_text: str,
    selected_sources: tuple[dict[str, Any], ...],
    retained_part_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    current_text = actual_text if actual_text in current_question else current_question
    context_frames = tuple(
        _items_from_prompt_section(
            prompt,
            marker="Available context frames:\n",
            key="available_context_frames",
        )
    )
    values = _resolved_values_for_sources(
        selected_sources=selected_sources,
        context_frames=context_frames,
        current_text=current_text,
    )
    retained_frame_parts = _retained_frame_part_refs(
        context_frames=context_frames,
        part_ids=retained_part_ids,
    )
    return {
        "kind": "conversation_resolution",
        "current_question_text": current_question,
        "outcome": {
            "kind": "resolved",
            "resolution_basis": (
                "The selected visible context supplies the meaning omitted by the "
                "current clause."
            ),
            "contextualized_question": contextualized_question,
            "clauses": [
                {
                    "current_clause_text": current_text,
                    "occurrence": 1,
                    "resolved_text": contextualized_question,
                    "retained_frame_parts": retained_frame_parts,
                    "values": values,
                }
            ],
        },
    }


def _resolved_values_for_sources(
    *,
    selected_sources: tuple[dict[str, Any], ...],
    context_frames: tuple[dict[str, Any], ...],
    current_text: str,
) -> list[dict[str, Any]]:
    values = [
        {
            "value_id": f"context_value_{index}",
            "resolved_text": str(anchor["text"]),
            "frame_parameter": {"kind": "none"},
            "sources": [
                {
                    "kind": "context_anchor",
                    "source_id": str(source["source_id"]),
                    "anchor_id": str(anchor["anchor_id"]),
                }
            ],
        }
        for index, (source, anchor) in enumerate(
            (
                (source, anchor)
                for source in selected_sources
                for anchor in source.get("meaning_anchors") or ()
                if isinstance(anchor, dict)
                and anchor.get("anchor_id")
                and anchor.get("text")
            ),
            start=1,
        )
    ]
    if values:
        values[0]["sources"].insert(
            0,
            {"kind": "current_span", "text": current_text, "occurrence": 1},
        )
        return values
    return []


def _retained_frame_part_refs(
    *,
    context_frames: tuple[dict[str, Any], ...],
    part_ids: tuple[str, ...],
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for part_id in part_ids:
        matches = [
            (str(frame["frame_id"]), str(part["part_id"]))
            for frame in context_frames
            for part in frame.get("parts") or ()
            if str(part.get("part_id") or "") == part_id
        ]
        if len(matches) != 1:
            raise AssertionError(f"retained frame part is not unique: {part_id}")
        frame_id, matched_part_id = matches[0]
        refs.append(
            {
                "kind": "frame_part",
                "frame_id": frame_id,
                "part_id": matched_part_id,
            }
        )
    return refs


def _answer_output_support_role(answer_output: dict[str, Any]) -> str:
    return str(answer_output.get("role") or "ROW_COUNT")


def _query_enrichment_payload_from_prompt(prompt: str) -> dict[str, Any]:
    requested_facts = _planner_prompt_json_section(prompt, label="Requested facts")[
        "requested_facts"
    ]
    entity_targets = _planner_prompt_json_section(prompt, label="Entity targets")[
        "entity_targets"
    ]
    vocabulary = _planner_prompt_json_section(prompt, label="API catalog vocabulary")[
        "api_catalog_vocabulary"
    ]
    resource_names = tuple(str(item) for item in vocabulary.get("resource_names") or ())
    return {
        "requested_fact_resource_name_matches": [
            {
                "requested_fact_id": fact["requested_fact_id"],
                "answer_output_resource_lineage": (
                    [
                        {
                            "answer_output_id": str(answer_output["answer_output_id"]),
                            "support_role": _answer_output_support_role(answer_output),
                            "source_text": str(
                                fact.get("requested_fact_description")
                                or "requested fact"
                            ),
                            "matching_resource_names": list(
                                _matching_resource_names_for_fact(
                                    fact,
                                    resource_names=resource_names,
                                )
                            ),
                        }
                        for answer_output in fact.get("answer_outputs") or []
                    ]
                    if _matching_resource_names_for_fact(
                        fact,
                        resource_names=resource_names,
                    )
                    else []
                ),
            }
            for fact in requested_facts
        ],
        "entity_target_catalog_search_terms": [
            {
                "target_id": target["target_id"],
                "catalog_search_terms": [
                    {
                        "basis": (
                            f"{resource_name} can identify "
                            f"{target['resolved_value_text']} because "
                            "value_meaning_hint is "
                            f"{target['value_meaning_hint']}."
                        ),
                        "term": resource_name,
                    }
                    for resource_name in resource_names
                ][:5],
            }
            for target in entity_targets
        ],
    }


def _matching_resource_names_for_fact(
    fact: dict[str, Any],
    *,
    resource_names: tuple[str, ...],
) -> tuple[str, ...]:
    fact_terms = _query_enrichment_terms(
        " ".join(
            (
                str(fact.get("requested_fact_id") or ""),
                str(fact.get("requested_fact_description") or ""),
                " ".join(
                    str(output.get("description") or "")
                    for output in fact.get("answer_outputs") or ()
                    if isinstance(output, dict)
                ),
            )
        )
    )
    return tuple(
        resource_name
        for resource_name in resource_names
        if fact_terms.intersection(_query_enrichment_terms(resource_name))
    )


def _query_enrichment_terms(text: str) -> set[str]:
    return {
        term.removesuffix("s")
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 1
    }


def _query_enrichment_payload(
    *term_groups: tuple[str, ...],
    requested_fact_id: str = "fact_1",
    entity_target_catalog_search_terms: list[dict[str, Any]] | None = None,
    support_role: str = "ROW_COUNT",
) -> dict[str, Any]:
    return {
        "requested_fact_resource_name_matches": [
            {
                "requested_fact_id": requested_fact_id,
                "answer_output_resource_lineage": [
                    {
                        "answer_output_id": "answer_1",
                        "support_role": support_role,
                        "source_text": f"source phrase {index}",
                        "matching_resource_names": list(terms),
                    }
                    for index, terms in enumerate(term_groups, start=1)
                ],
            }
        ],
        "entity_target_catalog_search_terms": entity_target_catalog_search_terms or [],
    }


def _plan_payload(
    plan: FactPlan,
    *,
    question_contract: QuestionContract,
) -> dict[str, Any]:
    from dataclasses import asdict

    if isinstance(plan.outcome, AnswerProgram):
        return _canonicalized_plan_payload(
            _pattern_payload_from_answer_plan(plan.outcome),
            question_contract=question_contract,
        )
    payload = _without_none(asdict(plan))
    payload.pop("bindings", None)
    outcome = payload.get("outcome")
    if isinstance(outcome, dict):
        outcome.pop("values", None)
        if "kind" not in outcome:
            outcome["kind"] = "fact_plan"
    return _canonicalized_plan_payload(payload, question_contract=question_contract)


def _pattern_payload_from_answer_plan(plan: AnswerProgram) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "fact_plan",
            "answers": [_pattern_answer_from_answer_plan(plan)],
        }
    }


def _pattern_answer_from_answer_plan(plan: AnswerProgram) -> dict[str, Any]:
    relation_by_id = {relation.id: relation for relation in plan.relations}
    if len(plan.operations) == 1 and isinstance(plan.operations[0].spec, AggregateSpec):
        aggregate = plan.operations[0]
        metric = aggregate.spec.aggregations[0]
        pattern = (
            "aggregate_by_group" if aggregate.spec.group_by else "aggregate_scalar"
        )
        source_relation = relation_by_id[aggregate.spec.input_relation]
        if metric.function == AggregationFunction.COUNT:
            raise AssertionError(
                "compiled count programs cannot be converted back to model choices"
            )
        metric_payload = {
            "kind": "aggregate_field",
            "function": metric.function.value,
            "field_id": metric.input_field,
            "label": metric.output_field,
        }
        answer = {
            **_pattern_answer_base(plan, source_relation),
            "pattern": pattern,
            "metric": metric_payload,
        }
        if aggregate.spec.group_by:
            answer["group_fields"] = [
                {"field_id": field_id} for field_id in aggregate.spec.group_by
            ]
        return answer
    if len(plan.operations) == 1 and isinstance(plan.operations[0].spec, ProjectSpec):
        project = plan.operations[0]
        result_field_ids = {
            field_id
            for output in plan.result_projection.relation_outputs
            for field_id in (
                tuple(component.field_id for component in output.entity_key.components)
                if output.entity_key is not None
                else (output.field_id,)
            )
        }
        result_output_ids = {
            fulfillment.result_output_id for fulfillment in plan.fulfillment
        }
        projected_result_ids = result_field_ids | result_output_ids
        output_fields = [
            {"field_id": field.expression.field_id}
            for field in project.spec.outputs
            if field.output_field in projected_result_ids
            and isinstance(field.expression, FieldRef)
        ]
        return {
            **_pattern_answer_base(plan, relation_by_id[project.spec.input_relation]),
            "pattern": "list_rows",
            "output_fields": output_fields,
        }
    raise AssertionError("test plan shape is not expressible as a model fact pattern")


def _pattern_answer_base(plan: AnswerProgram, relation: Relation) -> dict[str, Any]:
    read_id = relation.source.read_id
    if not read_id:
        raise AssertionError("model fact pattern requires an API read")
    source: dict[str, Any] = {"kind": "read", "read_id": read_id}
    if relation.source.param_bindings:
        source["param_bindings"] = [
            {
                "param_id": item.param_id,
                "value": cast(
                    ConstantRef,
                    item.value_expr,
                ).value.payload.canonical_value(),
            }
            for item in relation.source.param_bindings
        ]
    return {
        "requested_fact_id": plan.fulfillment[0].requested_fact_id,
        "answer_output_ids": [
            fulfillment.answer_output_id for fulfillment in plan.fulfillment
        ],
        "source": source,
    }


def _canonicalized_plan_payload(
    payload: Any,
    *,
    question_contract: QuestionContract,
) -> Any:
    canonical = _canonicalized_plan_value(
        payload,
        id_map=_question_contract_id_map(question_contract),
    )
    if isinstance(canonical, dict):
        outcome = canonical.get("outcome")
        if isinstance(outcome, dict) and isinstance(outcome.get("fulfillment"), list):
            outcome["fulfillment"] = _dedupe_fulfillment(outcome["fulfillment"])
    return canonical


def _dedupe_fulfillment(items: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            deduped.append(item)
            continue
        key = (
            str(item.get("requested_fact_id") or ""),
            str(item.get("answer_output_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _canonicalized_plan_value(value: Any, *, id_map: _QuestionContractIdMap) -> Any:
    if isinstance(value, dict):
        original_fact_id = value.get("requested_fact_id")
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if key == "requested_fact_id":
                canonical[key] = _canonical_requested_fact_id(item, id_map=id_map)
                continue
            if key == "answer_output_id":
                canonical[key] = _canonical_answer_output_id(
                    item,
                    requested_fact_id=original_fact_id,
                    id_map=id_map,
                )
                continue
            if key == "answer_output_ids" and isinstance(item, list):
                canonical[key] = [
                    _canonical_answer_output_id(
                        output_id,
                        requested_fact_id=original_fact_id,
                        id_map=id_map,
                    )
                    for output_id in item
                ]
                continue
            if key == "known_input_id":
                canonical[key] = _canonical_known_input_id(item, id_map=id_map)
                continue
            if key == "value_id":
                canonical[key] = _canonical_value_id(item, id_map=id_map)
                continue
            canonical[key] = _canonicalized_plan_value(item, id_map=id_map)
        return canonical
    if isinstance(value, list):
        return [_canonicalized_plan_value(item, id_map=id_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalized_plan_value(item, id_map=id_map) for item in value)
    if isinstance(value, str):
        return _canonical_evidence_ref(value, id_map=id_map)
    return value


def _canonical_requested_fact_id(
    value: Any,
    *,
    id_map: _QuestionContractIdMap,
) -> Any:
    if not isinstance(value, str):
        return value
    return id_map.requested_fact_ids.get(value, value)


def _canonical_answer_output_id(
    value: Any,
    *,
    requested_fact_id: Any,
    id_map: _QuestionContractIdMap,
) -> Any:
    if not isinstance(value, str) or not isinstance(requested_fact_id, str):
        return value
    return id_map.answer_output_ids.get((requested_fact_id, value), value)


def _canonical_known_input_id(
    value: Any,
    *,
    id_map: _QuestionContractIdMap,
) -> Any:
    if not isinstance(value, str):
        return value
    return id_map.known_input_ids.get(value, value)


def _canonical_value_id(
    value: Any,
    *,
    id_map: _QuestionContractIdMap,
) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("grounded_"):
        raw = value[len("grounded_") :]
        return f"grounded_{id_map.known_input_ids.get(raw, raw)}"
    mapped = id_map.known_input_ids.get(value)
    if mapped is not None:
        return f"grounded_{mapped}"
    return value


def _canonical_evidence_ref(
    value: str,
    *,
    id_map: _QuestionContractIdMap,
) -> str:
    for original_id, canonical_id in id_map.requested_fact_ids.items():
        if value == requested_fact_evidence_ref(original_id):
            return requested_fact_evidence_ref(canonical_id)
    for original_id, canonical_id in id_map.known_input_ids.items():
        if value == f"known_input:{original_id}":
            return f"known_input:{canonical_id}"
    return value


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_none(item)
            for key, item in value.items()
            if key != "proof_refs"
            and item is not None
            and not (
                key
                in {
                    "calendar_id",
                    "field_id",
                    "input_field",
                    "known_input_id",
                    "memory_relation_id",
                    "output",
                    "output_relation",
                    "read_id",
                    "relation_id",
                    "required_catalog_input_id",
                    "required_catalog_choice_input_id",
                    "right",
                    "row_source_id",
                    "scalar_id",
                    "ref",
                    "resolved_start",
                    "resolved_end",
                }
                and item == ""
            )
            and not (
                key
                in {
                    "available_options",
                    "candidate_refs",
                    "evidence_refs",
                    "nearest_fields",
                    "reviewed_read_ids",
                }
                and item == ()
            )
        }
    if isinstance(value, (list, tuple)):
        return [_without_none(item) for item in value]
    return value


__all__ = tuple(name for name in globals() if not name.startswith("__"))
