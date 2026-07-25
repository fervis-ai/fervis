#!/usr/bin/env python3
"""Build a reusable semantic Read Eligibility stability boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python" / "src"
EXPERIMENTS = Path(__file__).parent
for path in (PYTHON_SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_semantic_grounding_boundary import build_request as grounding_request  # noqa: E402
from fervis.lookup.relation_catalog.row_sources import (  # noqa: E402
    RowSourceCatalog,
    build_row_source_catalog,
)
from fervis.lookup.grounding import (  # noqa: E402
    CompatibleIdentityRoute,
    GroundingPartition,
    IdentifierKind,
    identity_resolution_tasks,
    reference_grounding_tasks,
)
from fervis.lookup.question_contract.parser import (  # noqa: E402
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)
from fervis.lookup.read_eligibility.semantic import (  # noqa: E402
    SemanticReadEligibilityRequest,
)
from fervis.lookup.read_eligibility.semantic_prompt import (  # noqa: E402
    SemanticReadEligibilityTurnPrompt,
)
from fervis.lookup.relation_catalog import (  # noqa: E402
    CatalogField,
    CandidateKey,
    CandidateKeyComponent,
    EndpointRead,
    EntityReference,
    EntityReferenceComponent,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402
from semantic_contract_fixtures import count_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model-key", default="openai:gpt-5.4-mini")
    args = parser.parse_args()
    request_payload = json.loads(args.request_file.read_text())
    request = build_request(request_payload)
    invocation = SemanticReadEligibilityTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=str(request_payload["question"]),
            conversation_context={},
        )
    )
    boundary = {
        "source_run_id": "semantic-read-eligibility-experiment",
        "sequence": 1,
        "purpose": "read_eligibility",
        "provider": args.provider,
        "model_key": args.model_key,
        "system_prompt": invocation.system_prompt,
        "prompt": invocation.prompt_text,
        "tool_specs": [asdict(item) for item in invocation.tool_specs],
        "assertion_context": request_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(boundary, indent=2) + "\n")
    return 0


def build_request(payload: dict[str, Any]) -> SemanticReadEligibilityRequest:
    if payload.get("kind") == "boolean_coverage":
        return _boolean_coverage_request(payload)
    if payload.get("kind") == "returned_identity":
        return _returned_identity_request(payload)
    grounding = grounding_request(payload)
    full_catalog = RelationCatalog(
        reads=(
            *grounding.resolver_catalog.reads,
            _observation_read(
                entity_kind=str(payload["resource_type"]),
                key_component=str(payload["key_component"]),
            ),
        )
    )
    all_sources = build_row_source_catalog(full_catalog)
    answer_sources = RowSourceCatalog(
        sources=tuple(
            source
            for source in all_sources.sources
            if source.read_id == "list_observation"
        )
    )
    parsed = _semantic_contract(payload)
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    identity_uses = tuple(
        item for item in index.input_use_sites if item.identity_set_ref is not None
    )
    [semantic_use] = identity_uses
    [grounding_task] = reference_grounding_tasks(
        (
            GroundingPartition(
                input_ref=input_term.id,
                use_refs=(semantic_use.use_ref,),
                expected_value_type=input_term.value_type,
                expected_set_ref=semantic_use.identity_set_ref,
                operand_meaning=semantic_use.operand_meaning,
            ),
        ),
        resolver_options_by_use_ref={
            semantic_use.use_ref: grounding.tasks[0].options,
        },
        denoted_instance_kinds_by_input_ref={
            input_term.id: str(
                payload.get("denoted_instance_kind")
                or payload["input_meaning"]
            )
        },
    )
    resolver_route = next(
        option
        for option in grounding_task.options
        if option.candidate.resolver_read_id == str(payload["expected_read"])
    )
    primary_key = str(payload["expected_read"]).startswith("get_")
    binding = CompatibleIdentityRoute(
        option_id=resolver_route.id,
        identifier_kind=(
            IdentifierKind.PRIMARY_KEY if primary_key else IdentifierKind.DESCRIPTIVE
        ),
        lookup_request_param_refs=(
            (
                f"get_{payload['resource_type']}.path.{payload['key_component']}"
                if primary_key
                else f"list_{payload['resource_type']}.query.name"
            ),
        ),
        returned_identity_verification_field_paths=(
            (f"data.{payload['key_component']}" if primary_key else "data.name"),
        ),
    )
    [identity_task] = identity_resolution_tasks(
        (grounding_task,),
        compatible_bindings_by_task_ref={grounding_task.task_ref: (binding,)},
    )
    return SemanticReadEligibilityRequest(
        indexes=parsed.semantic_indexes,
        source_catalog=answer_sources,
        answer_catalog=full_catalog,
        identity_tasks=(identity_task,),
        resolver_catalog=full_catalog,
    )


def _boolean_coverage_request(
    payload: dict[str, Any],
) -> SemanticReadEligibilityRequest:
    parsed = count_contract(
        question=str(payload["question"]),
        operand=str(payload["operand"]),
        candidate_meaning="measurements",
        fact_meaning="measurement amount",
        fact_value_type={"kind": "integer"},
        input_value_type={"kind": "integer"},
        operator="gt",
    )
    catalog = RelationCatalog(reads=(_measurement_read(),))
    sources = build_row_source_catalog(catalog)
    return SemanticReadEligibilityRequest(
        indexes=parsed.semantic_indexes,
        source_catalog=sources,
        answer_catalog=catalog,
        identity_tasks=(),
        resolver_catalog=catalog,
    )


def _returned_identity_request(
    payload: dict[str, Any],
) -> SemanticReadEligibilityRequest:
    parsed = _returned_identity_contract(str(payload["question"]))
    catalog = RelationCatalog(reads=(_event_read(),))
    return SemanticReadEligibilityRequest(
        indexes=parsed.semantic_indexes,
        source_catalog=build_row_source_catalog(catalog),
        answer_catalog=catalog,
        identity_tasks=(),
        resolver_catalog=catalog,
    )


def _returned_identity_contract(question: str) -> ParsedSemanticQuestionContract:
    def origin(meaning: str) -> dict[str, object]:
        return {
            "source": "question_context",
            "meaning": meaning,
            "resolved_input_ref": None,
        }

    meaning = parse_semantic_question_frame(
        {
            "decision_basis": "The request asks for the highest grouped total.",
            "outcome": {
                "kind": "question_meaning",
                "answer_requests": [
                    {
                        "result_kind": "grouped_results",
                        "qualifying_row_kind": {
                            "meaning": "amount occurrence",
                            "origin": {"kind": "question"},
                        },
                        "grouping_meanings": [
                            {
                                "meaning": "member",
                                "origin": {"kind": "question"},
                                "grouping_kind": "related_entity_identity",
                            }
                        ],
                        "return_request_basis": (
                            "The answer returns the winning member identity."
                        ),
                        "returned_result": {"kind": "identities"},
                        "answer_values": [
                            {
                                "value_ref": "v1",
                                "meaning": "total amount",
                                "origin": {"kind": "question"},
                            }
                        ],
                        "returned_value_refs": [],
                        "ordering_value_refs": ["v1"],
                        "selection": {
                            "kind": "first_rank_with_ties",
                        },
                        "universal_shape": "none",
                    }
                ],
                "supplied_values": [],
            },
        },
        question_context_texts=(question,),
    )
    if not isinstance(meaning, ParsedSemanticQuestionMeaning):
        raise ValueError("experiment requires complete requested-result meaning")
    member_identifier = {
        "kind": "fact",
        "observed_for_ref": "a1",
        "value_type": {"kind": "identifier", "set_ref": "s2"},
        "origin": origin("member"),
    }
    amount = {
        "kind": "fact",
        "observed_for_ref": "s1",
        "value_type": {"kind": "integer"},
        "origin": origin("amount"),
    }
    total = {
        "kind": "aggregate",
        "function": "sum",
        "argument": amount,
        "distinct_argument": False,
    }
    parsed = parse_semantic_question_contract(
        {
            "decision_basis": "Group occurrences by member and rank their totals.",
            "outcome": {
                "kind": "question_contract",
                "answer_requests": [
                    {
                        "requested_fact_ref": "fact_1",
                        "origin": origin("member with the highest total amount"),
                        "other_sets": [{"id": "s2", "origin": origin("member")}],
                        "other_associations": [
                            {
                                "id": "a1",
                                "from_set_ref": "s1",
                                "to_set_ref": "s2",
                                "origin": origin("member associated with occurrence"),
                            }
                        ],
                        "candidate_set": {
                            "instance_interpretation": "normal_business_instance"
                        },
                        "qualification": None,
                        "grouping": [{"id": "g1", "expression": member_identifier}],
                        "ordering": [
                            {
                                "expression": total,
                                "direction": "descending",
                            }
                        ],
                        "selection": {"kind": "first_rank_with_ties"},
                        "distinct_by": [],
                        "outputs": [{"expression": {"kind": "group_ref", "ref": "g1"}}],
                    }
                ],
            },
        },
        meaning=meaning,
        question_context_texts=(question,),
    )
    if not isinstance(parsed, ParsedSemanticQuestionContract):
        raise ValueError("experiment requires a complete semantic contract")
    return parsed


def _semantic_contract(payload: dict[str, Any]) -> ParsedSemanticQuestionContract:
    raw_operand = payload["operand"]
    collection = isinstance(raw_operand, list)
    operand = [str(item) for item in raw_operand] if collection else str(raw_operand)
    return count_contract(
        question=str(payload["question"]),
        operand=operand,
        candidate_meaning="observations",
        fact_meaning="associated member identity",
        fact_value_type={"kind": "text"},
        input_value_type=(
            {"kind": "collection", "element_type": {"kind": "text"}}
            if collection
            else {"kind": "text"}
        ),
        operator="in" if collection else "equals",
        identity_set_meaning="members",
    )


def _observation_read(
    *,
    entity_kind: str,
    key_component: str,
) -> EndpointRead:
    return EndpointRead(
        id="list_observation",
        endpoint_name="list_observation",
        resource_names=("observation",),
        row_paths=(RowPath("data", "data", RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="observation.member_id",
                path="data.member_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="observation.observation_id",
                path="data.observation_id",
                row_path_id="data",
                type="string",
            ),
        ),
        entity_references=(
            EntityReference(
                id=f"{entity_kind}_reference",
                target_entity_kind=entity_kind,
                target_key_id="primary_key",
                components=(
                    EntityReferenceComponent(
                        local_field_ref="observation.member_id",
                        target_component_id=key_component,
                    ),
                ),
            ),
        ),
    )


def _measurement_read() -> EndpointRead:
    return EndpointRead(
        id="list_measurement",
        endpoint_name="list_measurement",
        resource_names=("measurement",),
        row_paths=(RowPath("data", "data", RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="measurement.measurement_id",
                path="data.measurement_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="measurement.amount",
                path="data.amount",
                row_path_id="data",
                type="integer",
            ),
        ),
    )


def _event_read() -> EndpointRead:
    return EndpointRead(
        id="list_event_list",
        endpoint_name="list_event_list",
        resource_names=("event",),
        row_paths=(RowPath("data", "data", RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="event.event_id",
                path="data.event_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="event.member_id",
                path="data.member_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="event.amount",
                path="data.amount",
                row_path_id="data",
                type="integer",
            ),
            CatalogField(
                ref="event.status",
                path="data.status",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="event.occurred_at",
                path="data.occurred_at",
                row_path_id="data",
                type="datetime",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="primary_key",
                entity_kind="event",
                components=(
                    CandidateKeyComponent(id="event_id", field_ref="event.event_id"),
                ),
                primary=True,
            ),
        ),
        entity_references=(
            EntityReference(
                id="member_id_reference",
                target_entity_kind="member",
                target_key_id="primary_key",
                components=(
                    EntityReferenceComponent(
                        local_field_ref="event.member_id",
                        target_component_id="member_id",
                    ),
                ),
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
