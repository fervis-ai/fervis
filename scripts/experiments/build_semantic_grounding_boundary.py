#!/usr/bin/env python3
"""Build a reusable semantic Grounding stability boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from fervis.lookup.relation_catalog.row_sources import (  # noqa: E402
    build_row_source_catalog,
)
from fervis.lookup.grounding import (  # noqa: E402
    GroundingPartition,
    SemanticGroundingRequest,
    reference_binding_options,
    reference_grounding_tasks,
    time_grounding_tasks,
)
from fervis.lookup.grounding.semantic_prompt import (  # noqa: E402
    SemanticGroundingTurnPrompt,
)
from fervis.lookup.question_contract.model import (  # noqa: E402
    FactLocalKind,
    FactLocalRef,
    InputTerm,
)
from fervis.lookup.relation_catalog import (  # noqa: E402
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityKeyComponentTarget,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.semantic_types import (  # noqa: E402
    CollectionType,
    IdentifierType,
    SourceOrigin,
    SourceOriginKind,
    TemporalScopeType,
    TextType,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model-key", default="openai:gpt-5.4-mini")
    args = parser.parse_args()
    request_payload = json.loads(args.request_file.read_text())
    request = build_request(request_payload)
    invocation = SemanticGroundingTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
        )
    )
    boundary = {
        "source_run_id": "semantic-grounding-experiment",
        "sequence": 1,
        "purpose": "grounding",
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


def build_request(payload: dict[str, Any]) -> SemanticGroundingRequest:
    if payload.get("kind") == "time":
        return _time_request(payload)
    resource_type = str(payload["resource_type"])
    key_component = str(payload["key_component"])
    catalog = identity_catalog(
        resource_type=resource_type,
        key_component=key_component,
    )
    raw_operand = payload["operand"]
    operand = (
        tuple(str(item) for item in raw_operand)
        if isinstance(raw_operand, list)
        else str(raw_operand)
    )
    input_value_type = (
        CollectionType(TextType())
        if isinstance(operand, tuple)
        else TextType()
    )
    expected_value_type = (
        CollectionType(IdentifierType("s1"))
        if isinstance(operand, tuple)
        else IdentifierType("s1")
    )
    input_term = InputTerm(
        id="input_1",
        origin=_origin(str(payload["input_meaning"])),
        operand=operand,
        value_type=input_value_type,
    )
    set_ref = FactLocalRef("fact_1", FactLocalKind.SET, "s1")
    options = reference_binding_options(
        input_id=input_term.id,
        resolver_catalog=catalog,
        resolver_row_sources=build_row_source_catalog(catalog),
        expected_identity=None,
    )
    use_ref = "fact_1:input_use:e1:1"
    [task] = reference_grounding_tasks(
        (
            GroundingPartition(
                input_ref=input_term.id,
                use_refs=(use_ref,),
                expected_value_type=expected_value_type,
                expected_set_ref=set_ref,
                operand_meaning=str(
                    payload.get("operand_meaning") or payload["input_meaning"]
                ),
            ),
        ),
        resolver_options_by_use_ref={use_ref: options},
        denoted_instance_kinds_by_input_ref={
            input_term.id: str(
                payload.get("denoted_instance_kind")
                or payload.get("expected_set_meaning")
                or payload["input_meaning"]
            )
        },
    )
    return SemanticGroundingRequest(
        question=str(payload["question"]),
        inputs=(input_term,),
        tasks=(task,),
        set_origins={set_ref: _origin(str(payload["expected_set_meaning"]))},
        resolver_catalog=catalog,
    )


def _time_request(payload: dict[str, Any]) -> SemanticGroundingRequest:
    input_term = InputTerm(
        id="input_1",
        origin=_origin(str(payload["input_meaning"])),
        operand=str(payload["operand"]),
        value_type=TemporalScopeType(),
    )
    partition = GroundingPartition(
        input_ref=input_term.id,
        use_refs=("fact_1:input_use:e1:1",),
        expected_value_type=input_term.value_type,
        expected_set_ref=None,
        operand_meaning=str(
            payload.get("operand_meaning") or payload["input_meaning"]
        ),
    )
    return SemanticGroundingRequest(
        question=str(payload["question"]),
        inputs=(input_term,),
        tasks=(),
        time_tasks=time_grounding_tasks(
            (partition,), inputs={input_term.id: input_term}
        ),
        set_origins={},
        resolver_catalog=RelationCatalog(),
        runtime_date=str(payload["runtime_date"]),
        timezone=str(payload["timezone"]),
    )


def identity_catalog(*, resource_type: str, key_component: str) -> RelationCatalog:
    key = CandidateKey(
        id="primary_key",
        entity_kind=resource_type,
        components=(CandidateKeyComponent(key_component, f"entity.{key_component}"),),
        primary=True,
        context_field_refs=("entity.name",),
    )
    fields = (
        CatalogField(
            ref=f"entity.{key_component}",
            path=f"data.{key_component}",
            row_path_id="data",
            type="string",
        ),
        CatalogField(
            ref="entity.name",
            path="data.name",
            row_path_id="data",
            type="string",
        ),
    )
    return RelationCatalog(
        reads=(
            EndpointRead(
                id=f"list_{resource_type}",
                endpoint_name=f"list_{resource_type}",
                resource_names=(resource_type,),
                params=(
                    CatalogParam(
                        ref=f"list_{resource_type}.query.name",
                        name="name",
                        source=ParamSource.QUERY,
                        type="string",
                    ),
                ),
                row_paths=(RowPath("data", "data", RowCardinality.MANY),),
                fields=fields,
                candidate_keys=(key,),
            ),
            EndpointRead(
                id=f"get_{resource_type}",
                endpoint_name=f"get_{resource_type}",
                resource_names=(resource_type,),
                params=(
                    CatalogParam(
                        ref=f"get_{resource_type}.path.{key_component}",
                        name=key_component,
                        source=ParamSource.PATH,
                        type="string",
                        required=True,
                        entity_target=EntityKeyComponentTarget(
                            resource_type,
                            "primary_key",
                            key_component,
                        ),
                    ),
                ),
                row_paths=(RowPath("data", "data", RowCardinality.ONE),),
                fields=fields,
                candidate_keys=(key,),
            ),
        )
    )


def _origin(meaning: str) -> SourceOrigin:
    return SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, meaning)


if __name__ == "__main__":
    raise SystemExit(main())
