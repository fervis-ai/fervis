"""Build typed source-candidate registries from canonical candidate cards."""

from ._shared import Any, RelationSource, RelationSourceAppliedFilter, SourceKind
from .bindings import _bound_param_bindings
from .model import SourceCandidate


def _source_candidates_from_cards(
    payload: dict[str, Any],
    *,
    model_visible: bool = True,
) -> dict[str, SourceCandidate]:
    output: dict[str, SourceCandidate] = {}
    for fact_sources in payload.get("requested_fact_sources") or ():
        for candidate in _source_options_for_fact_sources(
            fact_sources,
            model_visible=model_visible,
        ):
            output[candidate.id] = candidate
    for candidate in payload.get("utility_source_candidates") or ():
        if not isinstance(candidate, dict):
            continue
        built = _utility_candidate(candidate)
        output[built.id] = built
    for candidate in payload.get("value_source_candidates") or ():
        if not isinstance(candidate, dict):
            continue
        built = _value_candidate(candidate, requested_fact_id="")
        output[built.id] = built
    return output


def _source_options_for_fact_sources(
    fact_sources: dict[str, Any],
    *,
    model_visible: bool,
) -> tuple[SourceCandidate, ...]:
    requested_fact_id = str(fact_sources.get("requested_fact_id") or "")
    return tuple(
        _fact_source_candidate(
            candidate,
            requested_fact_id=requested_fact_id,
            model_visible=model_visible,
        )
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
    )


def _fact_source_candidate(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
    model_visible: bool,
) -> SourceCandidate:
    candidate_kind = str(candidate.get("kind") or "")
    if candidate_kind == "same_scope_api_read":
        return _same_scope_candidate(
            candidate,
            requested_fact_id=requested_fact_id,
            model_visible=model_visible,
        )
    if candidate_kind == "prior_answer_rows":
        return _prior_rows_candidate(candidate, requested_fact_id=requested_fact_id)
    if candidate_kind == "value":
        return _value_candidate(candidate, requested_fact_id=requested_fact_id)
    if candidate_kind == "generated_calendar":
        return _calendar_candidate(candidate, requested_fact_id=requested_fact_id)
    if candidate_kind != "new_api_read":
        raise ValueError(f"unsupported source candidate kind: {candidate_kind}")
    return _new_api_candidate(
        candidate,
        requested_fact_id=requested_fact_id,
        model_visible=model_visible,
    )


def _new_api_candidate(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
    model_visible: bool = True,
) -> SourceCandidate:
    payload = (
        _model_visible_candidate_payload(candidate) if model_visible else candidate
    )
    return SourceCandidate(
        id=str(payload.get("source_candidate_id") or ""),
        requested_fact_id=requested_fact_id,
        kind="new_api_read",
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id=str(payload.get("read_id") or ""),
            row_source_id=str(payload.get("row_source_id") or ""),
        ),
        params=tuple(payload.get("params") or ()),
        applied_param_bindings=(*_bound_param_bindings(payload.get("bound_params")),),
        applied_filters=_applied_filters(payload),
        fields=tuple(payload.get("fields") or ()),
        population_bindings=tuple(payload.get("population_bindings") or ()),
        payload=payload,
    )


def _same_scope_candidate(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
    model_visible: bool = True,
) -> SourceCandidate:
    payload = (
        _model_visible_candidate_payload(candidate) if model_visible else candidate
    )
    applied_sets = _candidate_applied_param_binding_sets(payload)
    return SourceCandidate(
        id=str(payload.get("source_candidate_id") or ""),
        requested_fact_id=requested_fact_id,
        kind="same_scope_api_read",
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id=str(payload.get("read_id") or ""),
            row_source_id=str(payload.get("row_source_id") or ""),
        ),
        applied_param_bindings=applied_sets[0] if applied_sets else (),
        applied_param_binding_sets=applied_sets,
        applied_filters=_applied_filters(payload),
        fields=tuple(payload.get("fields") or ()),
        applies_to_requested_fact_ids=_candidate_applies_to_requested_facts(payload),
        population_bindings=tuple(payload.get("population_bindings") or ()),
        payload=payload,
    )


def _model_visible_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(candidate)
    binding_surface = (
        candidate.get("binding_surface")
        if isinstance(candidate.get("binding_surface"), dict)
        else {}
    )
    for key in (
        "applied_filters",
        "bound_params",
        "source_invocations",
        "population_bindings",
        "params",
        "row_predicates",
        "fulfillment_support_sets",
    ):
        if key in binding_surface:
            payload[key] = binding_surface[key]
    if "fulfillment_choices" in candidate:
        payload["fulfillment_support_sets"] = candidate["fulfillment_choices"]
    evidence_items = _fulfillment_support_set_evidence_items(
        payload.get("fulfillment_support_sets")
    )
    if evidence_items:
        payload["evidence_items"] = evidence_items
    fields = _model_visible_fields(candidate, evidence_items=evidence_items)
    if fields:
        payload["fields"] = fields
    return payload


def _model_visible_fields(
    candidate: dict[str, Any],
    *,
    evidence_items: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in candidate.get("fields") or ():
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("field_id") or field.get("id") or "")
        if not field_id or field_id in seen:
            continue
        seen.add(field_id)
        output.append(
            {
                key: field[key]
                for key in (
                    "field_id",
                    "id",
                    "type",
                    "label",
                    "roles",
                    "row_cardinality",
                    "identity",
                )
                if key in field and field[key] not in (None, "", [], ())
            }
        )
    for field in _fields_from_evidence_items(evidence_items):
        field_id = str(field.get("field_id") or field.get("id") or "")
        if not field_id or field_id in seen:
            continue
        seen.add(field_id)
        output.append(field)
    return tuple(output)


def _fulfillment_support_set_evidence_items(
    raw_support_sets: Any,
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for support_set in raw_support_sets or ():
        if not isinstance(support_set, dict):
            continue
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for key in (
                "metric_measure_evidence",
                "row_count_basis_evidence",
                "scope_evidence",
                "group_key_evidence",
            ):
                for item in slot.get(key) or ():
                    if not isinstance(item, dict):
                        continue
                    evidence_id = str(item.get("evidence_id") or "")
                    if not evidence_id or evidence_id in seen:
                        continue
                    seen.add(evidence_id)
                    output.append(dict(item))
    return tuple(output)


def _fields_from_evidence_items(
    evidence_items: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence_items:
        field_id = str(item.get("field_id") or "")
        if not field_id or field_id in seen:
            continue
        seen.add(field_id)
        output.append(
            {
                key: item[key]
                for key in (
                    "field_id",
                    "type",
                    "label",
                    "roles",
                    "row_cardinality",
                    "identity",
                )
                if key in item and item[key] not in (None, "")
            }
        )
    return tuple(output)


def _prior_rows_candidate(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
) -> SourceCandidate:
    return SourceCandidate(
        id=str(candidate.get("source_candidate_id") or ""),
        requested_fact_id=requested_fact_id,
        kind="prior_answer_rows",
        source=RelationSource(
            kind=SourceKind.MEMORY_READ,
            memory_relation_id=str(candidate.get("memory_relation_id") or ""),
        ),
        fields=tuple(candidate.get("fields") or ()),
        applies_to_requested_fact_ids=_candidate_applies_to_requested_facts(candidate),
        population_bindings=tuple(candidate.get("population_bindings") or ()),
        payload=candidate,
    )


def _utility_candidate(candidate: dict[str, Any]) -> SourceCandidate:
    return _calendar_candidate(candidate, requested_fact_id="")


def _calendar_candidate(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
) -> SourceCandidate:
    return SourceCandidate(
        id=str(candidate.get("source_candidate_id") or ""),
        requested_fact_id=requested_fact_id,
        kind="calendar",
        source=RelationSource(
            kind=SourceKind.GENERATED_CALENDAR,
            calendar_id=str(candidate.get("calendar_id") or ""),
        ),
        params=tuple(candidate.get("params") or ()),
        fields=tuple(candidate.get("fields") or ()),
        population_bindings=tuple(candidate.get("population_bindings") or ()),
        payload=candidate,
    )


def _value_candidate(
    candidate: dict[str, Any],
    *,
    requested_fact_id: str,
) -> SourceCandidate:
    return SourceCandidate(
        id=str(candidate.get("source_candidate_id") or ""),
        requested_fact_id=requested_fact_id,
        kind="value",
        value_id=str(candidate.get("value_id") or ""),
        applies_to_requested_fact_ids=tuple(
            str(item)
            for item in candidate.get("applies_to_requested_facts") or ()
            if str(item)
        ),
        population_bindings=tuple(candidate.get("population_bindings") or ()),
        payload=candidate,
    )


def _candidate_applied_param_binding_sets(
    candidate: dict[str, Any],
) -> tuple[tuple[Any, ...], ...]:
    invocations = tuple(
        item
        for item in candidate.get("source_invocations") or ()
        if isinstance(item, dict)
    )
    if invocations:
        return tuple(
            _bound_param_bindings(invocation.get("bound_params"))
            for invocation in invocations
        )
    bindings = _bound_param_bindings(candidate.get("bound_params"))
    return (bindings,) if bindings else ()


def _applied_filters(
    candidate: dict[str, Any],
) -> tuple[RelationSourceAppliedFilter, ...]:
    return RelationSourceAppliedFilter.from_payloads(
        item
        for item in candidate.get("applied_filters") or ()
        if isinstance(item, dict)
    )


def _candidate_applies_to_requested_facts(
    candidate: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        str(item)
        for item in candidate.get("applies_to_requested_facts") or ()
        if str(item)
    )
