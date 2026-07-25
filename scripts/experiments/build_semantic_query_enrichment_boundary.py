#!/usr/bin/env python3
"""Build a reusable semantic Query Enrichment stability boundary."""

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

from fervis.lookup.question_contract.model import FactLocalRef  # noqa: E402
from fervis.lookup.query_enrichment.semantic import (  # noqa: E402
    ReferenceInputRecallTask,
    SemanticQueryEnrichmentRequest,
    SemanticRecallBucket,
    SemanticRecallBucketKind,
)
from fervis.lookup.query_enrichment.semantic_prompt import (  # noqa: E402
    SemanticQueryEnrichmentTurnPrompt,
)
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind  # noqa: E402
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--catalog-index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model-key", default="openai:gpt-5.4-mini")
    args = parser.parse_args()
    request_payload = json.loads(args.request_file.read_text())
    if args.catalog_index is not None:
        vocabulary = _catalog_vocabulary(json.loads(args.catalog_index.read_text()))
        request_payload["resource_names"] = vocabulary["resource_names"]
    request = request_from_payload(request_payload)
    invocation = SemanticQueryEnrichmentTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question="Match the shown semantic requirements for recall.",
            conversation_context={},
        )
    )
    boundary = {
        "source_run_id": "semantic-query-enrichment-experiment",
        "sequence": 1,
        "purpose": "query_enrichment",
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


def _catalog_vocabulary(index: dict[str, Any]) -> dict[str, list[str]]:
    prompts = [
        turn["prompt"]
        for run in index["runs"]
        for turn in run["model_turns"]
        if turn["purpose"] == "query_enrichment"
    ]
    if len(prompts) != 1:
        raise ValueError("catalog index must contain one Query Enrichment turn")
    marker = "Catalog vocabulary:\n"
    start = prompts[0].index(marker) + len(marker)
    decoder = json.JSONDecoder()
    vocabulary, _ = decoder.raw_decode(prompts[0][start:])
    groups = vocabulary.get("resource_name_groups")
    if groups is not None:
        return {
            "resource_names": [
                name
                for group in groups
                for name in group["resource_names"]
            ],
        }
    return {"resource_names": list(vocabulary["resource_names"])}


def request_from_payload(payload: dict[str, Any]) -> SemanticQueryEnrichmentRequest:
    return SemanticQueryEnrichmentRequest(
        recall_buckets=tuple(_bucket(item) for item in payload["recall_buckets"]),
        reference_tasks=tuple(
            _reference_task(item) for item in payload.get("reference_tasks") or ()
        ),
        resource_names=tuple(payload["resource_names"]),
    )


def _origin(meaning: str) -> SourceOrigin:
    return SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, meaning)


def _bucket(item: dict[str, Any]) -> SemanticRecallBucket:
    return SemanticRecallBucket(
        bucket_ref=item["bucket_ref"],
        kind=SemanticRecallBucketKind(item["kind"]),
        origin=_origin(item["meaning"]),
        requirement_refs=tuple(
            FactLocalRef.from_token(ref) for ref in item["requirement_refs"]
        ),
    )


def _reference_task(item: dict[str, Any]) -> ReferenceInputRecallTask:
    return ReferenceInputRecallTask(
        input_use_ref=item["input_use_ref"],
        input_ref=item["input_ref"],
        input_origin=_origin(item["input_text"]),
        operand_meaning=item["operand_meaning"],
        reference_fact_ref=FactLocalRef.from_token(item["reference_fact_ref"]),
        expected_set_ref=(
            FactLocalRef.from_token(item["expected_set_ref"])
            if item.get("expected_set_ref")
            else None
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
