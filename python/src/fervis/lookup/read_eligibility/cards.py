"""Compact API read cards for retention review."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from fervis.lookup.relation_catalog import (
    CatalogFactAvailability,
    CatalogFact,
    EndpointRead,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.turn_prompts.projections import ApiReadResponseShapeProjector
from fervis.lookup.fact_plan.row_sources import RowSource
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.grounding.model import (
    CompatibleInputBinding,
    InputBindingOption,
    KnownInputBindingTask,
)
from fervis.lookup.grounding.surface import resolver_option_surface_from_catalog
from fervis.lookup.question_contract import (
    RequestedFact,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
)
from fervis.lookup.read_eligibility.source_groups import (
    read_card_source_groups_by_read,
)
from fervis.lookup.read_eligibility.input_bindings import (
    canonical_input_options,
    interpretation_question,
)
from fervis.lookup.read_eligibility.model import CanonicalInputOption


@dataclass(frozen=True)
class ReadEligibilityCards:
    payload: dict[str, object]
    canonical_options: tuple[CanonicalInputOption, ...]


def read_eligibility_cards_payload(
    *,
    requested_facts: tuple[RequestedFact, ...],
    catalog_selection: CatalogSelectionResult,
    resolver_catalog: RelationCatalog | None,
    binding_tasks: tuple[KnownInputBindingTask, ...] = (),
    compatible_reference_bindings: tuple[CompatibleInputBinding, ...] = (),
    canonical_values: tuple[FactValue, ...] = (),
) -> dict[str, object]:
    return build_read_eligibility_cards(
        requested_facts=requested_facts,
        catalog_selection=catalog_selection,
        resolver_catalog=resolver_catalog,
        binding_tasks=binding_tasks,
        compatible_reference_bindings=compatible_reference_bindings,
        canonical_values=canonical_values,
    ).payload


def build_read_eligibility_cards(
    *,
    requested_facts: tuple[RequestedFact, ...],
    catalog_selection: CatalogSelectionResult,
    resolver_catalog: RelationCatalog | None,
    binding_tasks: tuple[KnownInputBindingTask, ...] = (),
    compatible_reference_bindings: tuple[CompatibleInputBinding, ...] = (),
    canonical_values: tuple[FactValue, ...] = (),
) -> ReadEligibilityCards:
    known_input_tokens_by_id = _known_input_tokens_by_id(requested_facts)
    requested_facts_by_id = {fact.id: fact for fact in requested_facts}
    facts_by_read = _facts_by_read(catalog_selection.relation_catalog)
    reads_by_id = {read.id: read for read in catalog_selection.relation_catalog.reads}
    resolver_reads = resolver_catalog.reads if resolver_catalog is not None else ()
    resolver_reads_by_id = {read.id: read for read in resolver_reads}
    source_groups_by_read = read_card_source_groups_by_read(
        catalog_selection.relation_catalog
    )
    next_source_index = 1
    requested_fact_read_candidates: list[dict[str, object]] = []
    all_canonical_options: list[CanonicalInputOption] = []
    row_source_counts_by_read = {
        read_id: len(source_groups)
        for read_id, source_groups in source_groups_by_read.items()
    }
    resolver_options_by_id = {
        option.id: option for task in binding_tasks for option in task.options
    }
    for selection in catalog_selection.requested_fact_selections:
        requested_fact = requested_facts_by_id.get(selection.requested_fact_id)
        if requested_fact is None:
            continue
        fact_canonical_options = canonical_input_options(
            requested_fact_id=selection.requested_fact_id,
            binding_tasks=binding_tasks,
            compatible_reference_bindings=compatible_reference_bindings,
            known_input_tokens_by_id=known_input_tokens_by_id,
            canonical_values=canonical_values,
        )
        all_canonical_options.extend(fact_canonical_options)
        read_candidates = []
        for read_id in selection.selected_read_ids:
            if read_id not in reads_by_id:
                continue
            for source_group in source_groups_by_read.get(read_id, ()):
                card = _read_card(
                    reads_by_id[read_id],
                    source_candidate_id=f"source_{next_source_index}",
                    facts_by_read=facts_by_read,
                    source_group=source_group,
                    read_row_source_count=row_source_counts_by_read.get(read_id, 1),
                )
                read_candidates.append(card)
                next_source_index += 1
        requested_fact_read_candidates.append(
            {
                "requested_fact_id": selection.requested_fact_id,
                "answer_request": requested_fact.answer_request_model_dict(),
                "known_inputs": [
                    _known_input_payload(
                        known_input,
                        token=known_input_tokens_by_id[known_input.id],
                        answer_fact=requested_fact.description,
                        canonical_options=tuple(
                            option
                            for option in fact_canonical_options
                            if option.known_input_id == known_input.id
                        ),
                        resolver_options_by_id=resolver_options_by_id,
                        resolver_reads_by_id=resolver_reads_by_id,
                    )
                    for known_input in requested_fact.known_inputs
                ],
                "read_candidates": read_candidates,
            }
        )
    return ReadEligibilityCards(
        payload={
            "requested_fact_read_candidates": requested_fact_read_candidates,
        },
        canonical_options=tuple(all_canonical_options),
    )


def _read_card(
    read: EndpointRead,
    *,
    source_candidate_id: str,
    facts_by_read: dict[str, tuple[CatalogFact, ...]],
    source_group: tuple[RowSource, ...],
    read_row_source_count: int,
) -> dict[str, object]:
    read_shape = ApiReadResponseShapeProjector(read)
    read_facts = (*read.facts, *facts_by_read.get(read.id, ()))
    payload: dict[str, object] = {
        "source_candidate_id": source_candidate_id,
        "row_path_id": source_group[0].row_path_id if source_group else "",
        "read_row_source_count": read_row_source_count,
        **read_shape.prompt_payload(
            row_path_ids=tuple(source.row_path_id or "root" for source in source_group),
            source_candidate_id=source_candidate_id,
            include_evidence_tokens=True,
        ),
        "catalog_facts": [_catalog_fact_payload(fact) for fact in read_facts],
    }
    docstring = _read_docstring(read, facts=read_facts)
    if docstring:
        payload["docstring"] = docstring
    if len(source_group) == 1:
        payload["row_source_id"] = source_group[0].id
    bound_params = _source_group_bound_params(source_group)
    if bound_params:
        payload["bound_params"] = bound_params
    return payload


def _known_input_payload(
    known_input: RequestedFactKnownInput,
    *,
    token: str,
    answer_fact: str,
    canonical_options: tuple[CanonicalInputOption, ...],
    resolver_options_by_id: dict[str, InputBindingOption],
    resolver_reads_by_id: dict[str, EndpointRead],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": token,
        "source_text": known_input.text,
    }
    if isinstance(known_input, RequestedFactLiteralInput):
        payload.update(
            {
                "role": known_input.role.value,
                "resolved_text": known_input.resolved_value_text,
            }
        )
        if known_input.field_label_text:
            payload["field_label_text"] = known_input.field_label_text
        if known_input.value_meaning_hint:
            payload["value_meaning_hint"] = known_input.value_meaning_hint
    if canonical_options:
        payload["interpretation_question"] = interpretation_question(
            known_input_text=known_input.text,
            answer_fact=answer_fact,
        )
        payload["canonical_options"] = list(
            {
                option.id: _canonical_option_payload(
                    option,
                    resolver_options_by_id=resolver_options_by_id,
                    resolver_reads_by_id=resolver_reads_by_id,
                )
                for option in canonical_options
            }.values()
        )
    return payload


def _canonical_option_payload(
    option: CanonicalInputOption,
    *,
    resolver_options_by_id: dict[str, InputBindingOption],
    resolver_reads_by_id: dict[str, EndpointRead],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": option.id,
        "result": f"{option.entity_kind}:{option.key_id}",
    }
    if option.resolver_binding is None:
        payload["canonical_value"] = option.canonical_value_id
        return payload

    resolver_binding = option.resolver_binding
    resolver_option = resolver_options_by_id[resolver_binding.option_id]
    resolver_read = resolver_reads_by_id[resolver_option.candidate.resolver_read_id]
    resolver_catalog = RelationCatalog(reads=(resolver_read,))
    resolver_surface = resolver_option_surface_from_catalog(
        resolver_catalog,
        resolver_option,
    )
    payload["resolver"] = {
        **resolver_surface.prompt_payload(),
        "request_values": [
            {"param_ref": item.param_ref, "value": item.value}
            for item in resolver_binding.request_values
        ],
        "response_match_alternatives": list(
            resolver_binding.response_match_field_paths
        ),
    }
    return payload


def _known_input_tokens_by_id(
    requested_facts: tuple[RequestedFact, ...],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for fact in requested_facts:
        for known_input in fact.known_inputs:
            if known_input.id in output:
                continue
            ordinal = len(output) + 1
            output[known_input.id] = (
                f"{_input_token_stem(known_input.text)}_qi_{ordinal}"
            )
    return output


def _input_token_stem(value: str) -> str:
    words = re.findall(r"\w+", value.casefold(), flags=re.UNICODE)
    stem = "_".join(words).strip("_")[:32].rstrip("_")
    return stem or "input"


def _catalog_fact_payload(fact: CatalogFact) -> dict[str, str]:
    output = {
        "fact_ref": fact.ref,
        "availability": fact.availability.value,
    }
    if fact.availability == CatalogFactAvailability.AVAILABLE and fact.field_ref:
        output["field_ref"] = fact.field_ref
    return output


def _read_docstring(read: EndpointRead, *, facts: tuple[CatalogFact, ...]) -> str:
    if any(fact.availability != CatalogFactAvailability.AVAILABLE for fact in facts):
        return ""
    metadata = read.source_metadata if isinstance(read.source_metadata, dict) else {}
    return str(metadata.get("description") or "").strip()


def _source_group_bound_params(sources: tuple[RowSource, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for param in source.params:
            if param.default_source != "source_variant":
                continue
            key = (param.param_ref, str(param.default))
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "param_id": param.id,
                    "param_ref": param.param_ref,
                    "name": param.name,
                    "value": param.default,
                    "semantics": param.semantics.value,
                }
            )
    return output


def _facts_by_read(catalog: RelationCatalog) -> dict[str, tuple[CatalogFact, ...]]:
    output: dict[str, list[CatalogFact]] = {}
    for fact in catalog.facts:
        if not fact.read_id:
            continue
        output.setdefault(fact.read_id, []).append(fact)
    return {key: tuple(value) for key, value in output.items()}
