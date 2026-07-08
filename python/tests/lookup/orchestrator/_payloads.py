from dataclasses import replace
import re
from typing import Iterable

from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole

from tests.lookup.orchestrator._plans import *  # noqa: F403
from tests.lookup.prompt_sections import prompt_section_payload


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
    requested_fact_assessments: list[dict[str, Any]] = []
    for group in candidate_groups:
        requested_fact_id = group["requested_fact_id"]
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
        for spec in retention_specs_by_fact.get(requested_fact_id, ()):
            card = _read_eligibility_card_for_retention_spec(
                spec,
                cards_by_source_id=cards_by_source_id,
                cards_by_read_id=cards_by_read_id,
            )
            retention_specs_by_source_id[str(card["source_candidate_id"])].append(spec)
        read_candidate_reviews = []
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
                read_candidate_reviews.append(
                    _retention_review_payload_from_spec(
                        merged_spec,
                        card=card,
                    )
                )
                continue
            read_candidate_reviews.append(
                {
                    "source_candidate_id": source_candidate_id,
                    "read_id": str(card.get("read_id") or ""),
                    "relevant_row_path_tokens": [],
                    "relevant_field_tokens": [],
                    "retention_basis": (
                        "This test fixture did not declare this read candidate "
                        "as retained for the requested fact."
                    ),
                    "retention_decision": "DROP",
                }
            )
        requested_fact_assessments.append(
            {
                "requested_fact_id": requested_fact_id,
                "read_candidate_reviews": read_candidate_reviews,
            }
        )
    return {
        "requested_fact_assessments": requested_fact_assessments,
    }


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
            str(metric.get("field_id") or metric.get("record_id_field_id") or "")
            if isinstance(metric, dict)
            else ""
        )
        measured_fields = (
            (metric_field,)
            if isinstance(metric, dict)
            and metric.get("kind") == "aggregate_field"
            and metric_field
            else ()
        )
        count_row_fields = (
            (metric_field,)
            if isinstance(metric, dict)
            and metric.get("kind") == "count_records"
            and metric_field
            else ()
        )
        row_path_fields = _row_path_fields_for_answer(
            pattern=pattern,
            output_fields=output_fields,
            measured_fields=measured_fields,
            group_fields=group_fields,
            count_row_fields=count_row_fields,
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
    count_row_fields: tuple[str, ...],
) -> tuple[str, ...]:
    if pattern not in {
        "list_rows",
        "grouped_rows",
        "aggregate_scalar",
        "aggregate_by_group",
        "ranked_aggregate",
    }:
        return count_row_fields
    return _unique(
        (
            *count_row_fields,
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
        "source_candidate_id": str(card["source_candidate_id"]),
        "read_id": str(card.get("read_id") or ""),
        "relevant_row_path_tokens": row_tokens,
        "relevant_field_tokens": field_tokens,
        "retention_basis": (
            "This fixture retained the read because it exposes row or field "
            "evidence that may be useful for the requested fact."
        ),
        "retention_decision": "RETAIN",
    }


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
    return prompt_section_payload(prompt, label)


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


def _conversation_resolution_tool_name_for_payload(payload: dict[str, Any]) -> str:
    del payload
    return CONVERSATION_RESOLUTION_TOOL_NAME


def _select_conversation_resolution_tool_name(
    tool_specs: tuple[Any, ...],
    *,
    responses: dict[str, dict[str, Any]] | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    offered = _offered_conversation_resolution_tool_names(tool_specs)
    if not offered:
        return ""
    for name in offered:
        if responses and name in responses:
            return name
    wanted = _conversation_resolution_tool_name_for_payload(payload or {})
    return wanted if wanted in offered else offered[0]


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
                    )
                ),
                answer_subject=RequestedFactAnswerSubject(
                    subject_text=description or fact_id
                ),
                answer_outputs=tuple(
                    RequestedFactAnswerOutput(id=output_id)
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
                    RequestedFactAnswerExpressionFamily(family)
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
    *,
    prompt: str = "",
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
    return _question_contract_response_with_prompt_memory(payload, prompt=prompt)


def _question_contract_answer_request_payload(
    fact: RequestedFact,
    *,
    id_map: _QuestionContractIdMap,
) -> dict[str, Any]:
    used_input_refs = {
        id_map.known_input_ids[known.id]
        for known in fact.known_inputs
        if known.id in id_map.known_input_ids
    }
    subject_text = (
        fact.answer_subject.subject_text
        if fact.answer_subject is not None
        else fact.description
    )
    payload = {
        "answer_fact": fact.description,
        "answer_expression": (
            fact.answer_expression.to_answer_request_dict()
            if fact.answer_expression is not None
            else {"family": "scalar_aggregate"}
        ),
        "answer_subject": _answer_subject_payload(subject_text),
        "answer_population": _answer_population_payload(
            fact,
            subject_text=subject_text,
        ),
        "answer_outputs": [
            _answer_output_for_current_contract(output.to_answer_request_dict())
            for output in fact.answer_outputs
        ],
        "used_question_inputs": [
            input_ref
            for input_ref in id_map.known_input_ids.values()
            if input_ref in used_input_refs
        ],
    }
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
    payload["resolved_value_text"] = known.resolved_value_text
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
    answer_request = {
        "answer_fact": subject,
        "answer_expression": {"family": answer_expression_family},
        "answer_subject": _answer_subject_payload(answer_subject or subject),
        "answer_population": _answer_population_payload_from_text(
            description=subject,
            subject_text=answer_subject or subject,
        ),
        "answer_outputs": [{"description": part} for part in parts],
        "used_question_inputs": input_refs,
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


def _answer_population_payload(
    fact: RequestedFact,
    *,
    subject_text: str,
) -> dict[str, Any]:
    if fact.answer_population is not None:
        return fact.answer_population.to_question_contract_dict()
    instance_interpretation = (
        fact.answer_subject.instance_interpretation
        if fact.answer_subject is not None
        else RequestedFactAnswerSubject(
            subject_text=subject_text
        ).instance_interpretation
    )
    return default_answer_population(
        description=fact.description,
        subject_text=subject_text,
        instance_interpretation=instance_interpretation,
    ).to_question_contract_dict()


def _answer_population_payload_from_text(
    *,
    description: str,
    subject_text: str,
) -> dict[str, Any]:
    return default_answer_population(
        description=description,
        subject_text=subject_text,
        instance_interpretation=RequestedFactAnswerSubject(
            subject_text=subject_text
        ).instance_interpretation,
    ).to_question_contract_dict()


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
    output["resolved_value_text"] = str(
        item.get("resolved_value_text") or item.get("value_text") or source_text
    )
    if item.get("field_label_text"):
        output["field_label_text"] = str(item["field_label_text"])
    if item.get("value_meaning_hint"):
        output["value_meaning_hint"] = str(item["value_meaning_hint"])
    return output


def _question_contract_response_with_prompt_memory(
    response: dict[str, Any],
    *,
    prompt: str,
    fact_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = dict(response)
    family_by_fact_id = _answer_expression_family_by_fact_id_from_fact_plan(
        fact_plan or {}
    )
    output.setdefault(
        "question_input_inventory_check",
        {"all_input_like_phrases_declared": True},
    )
    question_inputs = output.get("question_inputs")
    if isinstance(question_inputs, list):
        for item in question_inputs:
            if isinstance(item, dict):
                item.setdefault(
                    "inventory_check",
                    {
                        "why_this_is_an_input": (
                            f"{item.get('reference_text') or 'input'} is a declared question input"
                        )
                    },
                )
    answer_requests = output.get("answer_requests")
    if not isinstance(answer_requests, list):
        return output
    output.setdefault("answer_requests_count", len(answer_requests))
    output["answer_requests"] = [
        _question_contract_answer_request_for_current_contract(
            item,
            prompt=prompt,
            answer_expression_family=(
                family_by_fact_id.get(
                    str(item.get("requested_fact_id") or f"fact_{index}")
                )
                if isinstance(item, dict)
                else None
            ),
        )
        for index, item in enumerate(answer_requests, start=1)
    ]
    return output


def _question_contract_answer_request_for_current_contract(
    item: Any,
    *,
    prompt: str,
    answer_expression_family: str | None = None,
) -> Any:
    if not isinstance(item, dict):
        return item
    output = dict(item)
    if answer_expression_family:
        output["answer_expression"] = _question_contract_answer_expression_payload(
            answer_expression_family
        )
    else:
        output.setdefault("answer_expression", {"family": "scalar_aggregate"})
    if "answer_subject" in output:
        output["answer_subject"] = (
            _question_contract_answer_subject_for_current_contract(
                output.get("answer_subject"),
            )
        )
    if "answer_population" not in output:
        subject_text = str(
            (output.get("answer_subject") or {}).get("subject_text")
            if isinstance(output.get("answer_subject"), dict)
            else output.get("answer_fact")
        )
        output["answer_population"] = _answer_population_payload_from_text(
            description=str(output.get("answer_fact") or subject_text),
            subject_text=subject_text,
        )
    answer_outputs = output.get("answer_outputs")
    if isinstance(answer_outputs, list):
        output["answer_outputs"] = [
            _answer_output_for_current_contract(answer_output)
            for answer_output in answer_outputs
        ]
    return output


def _answer_output_for_current_contract(answer_output: object) -> object:
    if not isinstance(answer_output, dict):
        return answer_output
    if "description" not in answer_output:
        return answer_output
    output: dict[str, object] = {
        "description": answer_output.get("description"),
    }
    if answer_output.get("role"):
        output["role"] = answer_output["role"]
    return output


def _question_contract_answer_expression_payload(family: str) -> dict[str, object]:
    output: dict[str, object] = {"family": family}
    if family == "grouped_aggregate":
        output["group_key"] = {
            "description": "group",
            "domain": "SOURCE_RESULT_VALUES",
        }
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
    if pattern == "ranked_aggregate":
        return "ranked_selection"
    if pattern == "computed_scalar":
        return "computed_scalar"
    if pattern == "set_difference":
        return "set_difference"
    return "scalar_aggregate"


def _question_contract_answer_subject_for_current_contract(
    raw: Any,
) -> Any:
    if isinstance(raw, dict):
        subject_text = str(raw.get("subject_text") or "").strip()
        if subject_text:
            instance = raw.get("instance_interpretation")
            if isinstance(instance, dict) and str(instance.get("kind") or "").strip():
                return {
                    "subject_text": subject_text,
                    "instance_interpretation": {
                        "kind": str(instance.get("kind") or "").strip()
                    },
                }
            return _answer_subject_payload(subject_text)
    return raw


def _current_question_from_prompt(prompt: str) -> str:
    marker = "Current question:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def _conversation_resolution_payload_from_prompt(prompt: str) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    return {
        "kind": "conversation_resolution",
        "status": "standalone",
        "current_question_text": current_question,
        "clause_resolutions": [],
        "unresolved": _resolved_unresolved(),
    }


def _conversation_resolution_payload_using_memory(
    prompt: str,
    *,
    integrated_question: str | None = None,
    actual_text: str,
    source_kind: str = "",
) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    resolved_question = integrated_question or current_question
    source = (
        _context_source_by_kind(prompt, source_kind)
        if source_kind
        else _first_context_source(prompt)
    )
    return _conversation_resolution_clause_payload(
        prompt=prompt,
        current_question=current_question,
        integrated_question=resolved_question,
        actual_text=actual_text,
        selected_sources=(source,),
        force_selected_source=True,
    )


def _conversation_resolution_payload_using_memories(
    prompt: str,
    *,
    integrated_question: str,
    memories: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    current_question = _current_question_from_prompt(prompt)
    actual_text = str(memories[0].get("actual_text") or "that") if memories else "that"
    selected_sources = _context_sources_from_prompt(prompt)
    return _conversation_resolution_clause_payload(
        prompt=prompt,
        current_question=current_question,
        integrated_question=integrated_question,
        actual_text=actual_text,
        selected_sources=selected_sources,
        force_selected_source=bool(selected_sources),
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
        return "clarification_answer"
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
    integrated_question: str,
    actual_text: str,
    selected_sources: tuple[dict[str, Any], ...],
    requested_value_frame: dict[str, Any] | None = None,
    dependencies: list[dict[str, Any]] | None = None,
    force_selected_source: bool = False,
) -> dict[str, Any]:
    current_text = actual_text if actual_text in current_question else current_question
    context_frames = tuple(
        _items_from_prompt_section(
            prompt,
            marker="Available context frames:\n",
            key="available_context_frames",
        )
    )
    value_frame = requested_value_frame or _value_frame_for_prompt_context(
        context_frames=context_frames,
        selected_sources=selected_sources,
        current_value_text=current_text,
        integrated_question=integrated_question,
    )
    dependencies = dependencies or _dependencies_for_resolution(
        selected_sources=selected_sources,
        current_question=current_question,
        current_text=current_text,
        integrated_question=integrated_question,
        excluded_phrases=_value_frame_excluded_phrases(value_frame),
    )
    if force_selected_source and not dependencies and selected_sources:
        source = selected_sources[0]
        components = _meaning_components_for_source(source, resolved_text="")
        source_text = _source_text_for_source(source)
        if components:
            dependencies = [
                {
                    "anchor_text": current_text,
                    "occurrence": 1,
                    "kind": "reference",
                    "meaning_components": [
                        {**component, "resolved_text": source_text}
                        for component in components
                    ],
                    "resolved_text": "prior referenced context",
                    "must_preserve_terms": [],
                }
            ]
    return {
        "kind": "conversation_resolution",
        "status": "resolved",
        "current_question_text": current_question,
        "clause_resolutions": [
            {
                "current_clause_text": current_text,
                "occurrence": 1,
                "requested_value_frame": value_frame,
                "dependencies": dependencies,
                "resolved_clause_text": integrated_question,
            }
        ],
        "unresolved": _resolved_unresolved(),
    }


def _dependencies_for_resolution(
    *,
    selected_sources: tuple[dict[str, Any], ...],
    current_question: str,
    current_text: str,
    integrated_question: str,
    excluded_phrases: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in selected_sources:
        source_text = _source_text_for_source(source)
        phrase = _shared_phrase(
            source_text,
            integrated_question=integrated_question,
        )
        if phrase and phrase not in excluded_phrases:
            components = _meaning_components_for_source(source, resolved_text=phrase)
            if not components:
                continue
            items.append(
                {
                    "anchor_text": current_text,
                    "occurrence": 1,
                    "kind": "reference",
                    "meaning_components": components,
                    "resolved_text": phrase,
                    "must_preserve_terms": [phrase],
                }
            )
    if items:
        return items
    phrase = _shared_phrase(
        current_text,
        integrated_question=integrated_question,
    )
    source_text = current_text if phrase else current_question
    phrase = phrase or _shared_phrase(
        source_text,
        integrated_question=integrated_question,
    )
    if not phrase:
        return []
    return []


def _meaning_components_for_source(
    source: dict[str, Any],
    *,
    resolved_text: str,
) -> list[dict[str, Any]]:
    anchors = source.get("meaning_anchors") or ()
    if not isinstance(anchors, list) or not anchors:
        return []
    output: list[dict[str, Any]] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        output.append(
            {
                "kind": _meaning_component_kind(str(anchor.get("kind") or "")),
                "source_id": str(source.get("source_id") or ""),
                "source_text": str(anchor.get("text") or ""),
                "memory_id": str(anchor.get("memory_id") or ""),
                "resolved_text": resolved_text or str(anchor.get("text") or ""),
            }
        )
    return output


def _meaning_component_kind(anchor_kind: str) -> str:
    if anchor_kind == "entity_identity":
        return "entity"
    if anchor_kind == "time_scope":
        return "scope"
    if anchor_kind == "row_set":
        return "row_set"
    if anchor_kind == "scalar_value":
        return "value"
    return "other"


def _literal_requested_value_frame(
    *,
    current_value_text: str,
    context_frames: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    return {
        "current_value_surface": {
            "text": current_value_text,
            "kind": "self_sufficient_current_value",
        },
        "context_frame_choices": [
            {
                "frame_id": str(frame.get("frame_id") or ""),
                "choice": "not_for_this_clause",
                "current_conflict_quotes": [],
            }
            for frame in context_frames
            if str(frame.get("frame_id") or "").strip()
        ],
    }


def _value_frame_for_prompt_context(
    *,
    context_frames: tuple[dict[str, Any], ...],
    selected_sources: tuple[dict[str, Any], ...],
    current_value_text: str,
    integrated_question: str,
) -> dict[str, Any]:
    selected_source_ids = {
        str(source.get("source_id") or "") for source in selected_sources
    }
    for frame in context_frames:
        requested_frame = str(frame.get("requested_frame") or "")
        source_ids = {str(source_id) for source_id in frame.get("source_ids") or ()}
        if requested_frame in integrated_question and selected_source_ids & source_ids:
            selected_frame_id = str(frame.get("frame_id") or "")
            return {
                "current_value_surface": {
                    "text": current_value_text,
                    "kind": "broad_current_value",
                },
                "context_frame_choices": [
                    {
                        "frame_id": str(item.get("frame_id") or ""),
                        "choice": (
                            "use_frame"
                            if str(item.get("frame_id") or "") == selected_frame_id
                            else "not_for_this_clause"
                        ),
                        "current_conflict_quotes": [],
                    }
                    for item in context_frames
                    if str(item.get("frame_id") or "").strip()
                ],
            }
    return _literal_requested_value_frame(
        current_value_text=current_value_text,
        context_frames=context_frames,
    )


def _value_frame_excluded_phrases(value_frame: dict[str, Any]) -> tuple[str, ...]:
    choices = value_frame.get("context_frame_choices") or ()
    if not any(item.get("choice") == "use_frame" for item in choices):
        return ()
    return ("selected context frame",)


def _resolved_unresolved() -> dict[str, Any]:
    return {
        "unresolved_kind": "none",
        "why_unresolved": "",
        "candidate_interpretations": [],
    }


def _shared_phrase(
    source_text: str,
    *,
    integrated_question: str,
) -> str:
    words = re.findall(r"[A-Za-z0-9]+", source_text)
    words = [word for word in words if word]
    for length in range(len(words), 0, -1):
        for start in range(0, len(words) - length + 1):
            phrase = " ".join(words[start : start + length])
            if phrase in source_text and phrase in integrated_question:
                return phrase
    return ""


def _source_text_for_source(source: dict[str, Any]) -> str:
    text = str(source.get("text") or "").strip()
    if not text:
        return ""
    return text


def _answer_output_support_role(answer_output: dict[str, Any]) -> str:
    return str(answer_output.get("role") or "ROW_POPULATION")


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
    support_role: str = "ROW_POPULATION",
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

    if isinstance(plan.outcome, AnswerPlan):
        return _canonicalized_plan_payload(
            _pattern_payload_from_answer_plan(plan.outcome),
            question_contract=question_contract,
        )
    payload = _without_none(asdict(plan))
    outcome = payload.get("outcome")
    if isinstance(outcome, dict):
        outcome.pop("values", None)
        if "kind" not in outcome:
            outcome["kind"] = "fact_plan"
    return _canonicalized_plan_payload(payload, question_contract=question_contract)


def _pattern_payload_from_answer_plan(plan: AnswerPlan) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "fact_plan",
            "answers": [_pattern_answer_from_answer_plan(plan)],
        }
    }


def _pattern_answer_from_answer_plan(plan: AnswerPlan) -> dict[str, Any]:
    relation_by_id = {relation.id: relation for relation in plan.relations}
    if len(plan.operations) == 1 and isinstance(plan.operations[0].spec, AggregateSpec):
        aggregate = plan.operations[0]
        metric = aggregate.spec.aggregations[0]
        pattern = (
            "aggregate_by_group" if aggregate.spec.group_by else "aggregate_scalar"
        )
        source_relation = relation_by_id[aggregate.spec.input_relation]
        metric_payload = (
            {
                "kind": "count_records",
                "record_id_field_id": metric.input_field
                or next(iter(field.field_id for field in source_relation.fields), ""),
                "label": metric.output_field,
            }
            if metric.function == AggregationFunction.COUNT
            else {
                "kind": "aggregate_field",
                "function": metric.function.value,
                "field_id": metric.input_field,
                "label": metric.output_field,
            }
        )
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
        rendered_output_ids = {
            fulfillment.render_output_id for fulfillment in plan.fulfillment
        }
        output_fields = [
            {"field_id": field.source}
            for field in project.spec.fields
            if (field.output or field.source) in rendered_output_ids
        ]
        return {
            **_pattern_answer_base(plan, relation_by_id[project.spec.input_relation]),
            "pattern": "list_rows",
            "output_fields": output_fields,
        }
    raise AssertionError("test plan shape is not expressible as a model fact pattern")


def _pattern_answer_base(plan: AnswerPlan, relation: Relation) -> dict[str, Any]:
    read_id = relation.source.read_id
    if not read_id:
        raise AssertionError("model fact pattern requires an API read")
    source: dict[str, Any] = {"kind": "read", "read_id": read_id}
    if relation.source.param_bindings:
        source["param_bindings"] = [
            {"param_id": item.param_id, "value": item.value}
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
