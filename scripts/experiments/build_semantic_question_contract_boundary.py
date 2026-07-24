#!/usr/bin/env python3
"""Build a standalone stability boundary from the production-target contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from fervis.lookup.question_contract import (  # noqa: E402
    ParsedSemanticQuestionMeaning,
    QuestionContractRequest,
    parse_semantic_question_frame,
    SemanticQuestionContractTurnPrompt,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--frame-payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model-key", default="openai:gpt-5.4-mini")
    parser.add_argument("--expected-request-count", type=int)
    parser.add_argument(
        "--expected-result-grain",
        choices=("subject_rows", "groups", "singleton"),
    )
    parser.add_argument("--expected-input-text", action="append", default=[])
    parser.add_argument("--expected-input-kind", action="append", default=[])
    parser.add_argument("--required-input-term", action="append", default=[])
    parser.add_argument("--required-reference-input-term", action="append", default=[])
    parser.add_argument("--expected-collection-operand", action="append", default=[])
    parser.add_argument("--required-expression-kind", action="append", default=[])
    parser.add_argument("--required-boolean-use-site", action="append", default=[])
    parser.add_argument(
        "--expected-selection",
        choices=("all_results", "first_rank_with_ties", "take_with_boundary_ties"),
    )
    parser.add_argument(
        "--grouping-identifier-set",
        choices=("candidate", "distinct_from_candidate"),
    )
    args = parser.parse_args()
    expected_input_kinds = dict(item.split("=", 1) for item in args.expected_input_kind)
    payload = build_boundary_payload(
        question=args.question,
        frame_payload=json.loads(args.frame_payload.read_text()),
        provider=args.provider,
        model_key=args.model_key,
        expected_request_count=args.expected_request_count,
        expected_result_grain=args.expected_result_grain,
        expected_input_texts=tuple(args.expected_input_text),
        expected_input_kinds=expected_input_kinds,
        required_input_terms=tuple(args.required_input_term),
        required_reference_input_terms=tuple(args.required_reference_input_term),
        expected_collection_operands=tuple(args.expected_collection_operand),
        required_expression_kinds=tuple(args.required_expression_kind),
        required_boolean_use_sites=tuple(args.required_boolean_use_site),
        expected_selection=args.expected_selection,
        grouping_identifier_set=args.grouping_identifier_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def build_boundary_payload(
    *,
    question: str,
    frame_payload: dict[str, object],
    provider: str = "openai",
    model_key: str = "openai:gpt-5.4-mini",
    expected_request_count: int | None = None,
    expected_result_grain: str | None = None,
    expected_input_texts: tuple[str, ...] = (),
    expected_input_kinds: dict[str, str] | None = None,
    accepted_input_inventories: tuple[tuple[str, ...], ...] = (),
    required_input_terms: tuple[str, ...] = (),
    required_reference_input_terms: tuple[str, ...] = (),
    expected_collection_operands: tuple[str, ...] = (),
    required_expression_kinds: tuple[str, ...] = (),
    required_boolean_use_sites: tuple[str, ...] = (),
    expected_selection: str | None = None,
    grouping_identifier_set: str | None = None,
    expected_instance_interpretation: str | None = None,
) -> dict[str, object]:
    request = QuestionContractRequest(
        current_question=question,
        conversation_context={},
    )
    meaning = parse_semantic_question_frame(
        frame_payload,
        question_context_texts=(question,),
    )
    if not isinstance(meaning, ParsedSemanticQuestionMeaning):
        raise ValueError(
            "semantic Question Contract boundary requires complete meaning"
        )
    invocation = SemanticQuestionContractTurnPrompt(
        request,
        meaning=meaning,
    ).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context={},
            host=request.host,
        )
    )
    payload = {
        "source_run_id": "semantic-question-contract-experiment",
        "sequence": 1,
        "purpose": "question_contract",
        "provider": provider,
        "model_key": model_key,
        "system_prompt": invocation.system_prompt,
        "prompt": invocation.prompt_text,
        "tool_specs": [asdict(item) for item in invocation.tool_specs],
        "assertion_context": {
            "question_context_texts": [question],
            "frame_payload": frame_payload,
            "expected_request_count": expected_request_count,
            "expected_result_grain": expected_result_grain,
            "expected_input_texts": list(expected_input_texts),
            "expected_input_kinds": dict(expected_input_kinds or {}),
            "accepted_input_inventories": [
                list(inventory) for inventory in accepted_input_inventories
            ],
            "required_input_terms": list(required_input_terms),
            "required_reference_input_terms": list(required_reference_input_terms),
            "expected_collection_operands": list(expected_collection_operands),
            "required_expression_kinds": list(required_expression_kinds),
            "required_boolean_use_sites": list(required_boolean_use_sites),
            "expected_selection": expected_selection,
            "grouping_identifier_set": grouping_identifier_set,
            "expected_instance_interpretation": expected_instance_interpretation,
        },
    }
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
