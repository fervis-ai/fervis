#!/usr/bin/env python3
"""Capture one Question Contract invocation from its production prompt class."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from fervis.lookup.question_contract import (  # noqa: E402
    ParsedSemanticQuestionMeaning,
    QuestionContractRequest,
    parse_semantic_question_frame,
    SemanticQuestionContractTurnPrompt,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402
from scripts.experiments.shared.boundary import captured_boundary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--frame-payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boundary-variant", type=Path)
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
    parser.add_argument("--expected-aggregate-function", action="append", default=[])
    parser.add_argument("--required-boolean-use-site", action="append", default=[])
    parser.add_argument("--required-coverage-observation-input-ref")
    parser.add_argument(
        "--expected-selection",
        choices=("all_results", "first_rank_with_ties", "take_with_boundary_ties", "position_with_ties"),
    )
    parser.add_argument(
        "--grouping-identifier-set",
        choices=("candidate", "distinct_from_candidate"),
    )
    parser.add_argument(
        "--reference-identifier-set",
        choices=("candidate", "distinct_from_candidate", "grouping"),
    )
    parser.add_argument(
        "--output-identifier-set",
        choices=(
            "candidate",
            "distinct_from_candidate",
            "distinct_from_candidate_and_reference",
        ),
    )
    parser.add_argument(
        "--expected-requested-output-kind",
        action="append",
        choices=("related_entity", "value"),
        default=[],
    )
    parser.add_argument("--forbid-identifier-ordering", action="store_true")
    args = parser.parse_args()
    expected_input_kinds = dict(item.split("=", 1) for item in args.expected_input_kind)
    frame_document = json.loads(args.frame_payload.read_text())
    frame_payload = frame_document.get("frame_payload", frame_document)
    if not isinstance(frame_payload, dict):
        raise ValueError("frame payload must be an object")
    conversation_text_by_resolved_input_ref = frame_document.get(
        "conversation_text_by_resolved_input_ref", {}
    )
    if not isinstance(conversation_text_by_resolved_input_ref, dict):
        raise ValueError("conversation text map must be an object")
    payload = build_boundary_payload(
        question=args.question,
        frame_payload=frame_payload,
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
        expected_aggregate_functions=tuple(args.expected_aggregate_function),
        required_boolean_use_sites=tuple(args.required_boolean_use_site),
        required_coverage_observation_input_ref=(
            args.required_coverage_observation_input_ref
        ),
        expected_selection=args.expected_selection,
        grouping_identifier_set=args.grouping_identifier_set,
        reference_identifier_set=args.reference_identifier_set,
        output_identifier_set=args.output_identifier_set,
        expected_requested_output_kinds=tuple(
            args.expected_requested_output_kind
        ),
        forbid_identifier_ordering=args.forbid_identifier_ordering,
        conversation_text_by_resolved_input_ref={
            str(key): str(value)
            for key, value in conversation_text_by_resolved_input_ref.items()
        },
    )
    if args.boundary_variant is not None:
        payload = _load_boundary_variant(args.boundary_variant)(payload)
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
    expected_aggregate_functions: tuple[str, ...] = (),
    required_boolean_use_sites: tuple[str, ...] = (),
    required_coverage_observation_input_ref: str | None = None,
    expected_selection: str | None = None,
    grouping_identifier_set: str | None = None,
    reference_identifier_set: str | None = None,
    output_identifier_set: str | None = None,
    expected_requested_output_kinds: tuple[str, ...] = (),
    forbid_identifier_ordering: bool = False,
    expected_instance_interpretation: str | None = None,
    prompt_type: type[SemanticQuestionContractTurnPrompt] = (
        SemanticQuestionContractTurnPrompt
    ),
    conversation_text_by_resolved_input_ref: dict[str, str] | None = None,
) -> dict[str, object]:
    request = QuestionContractRequest(
        current_question=question,
        conversation_context={},
    )
    meaning = parse_semantic_question_frame(
        frame_payload,
        question_context_texts=(question,),
        conversation_text_by_resolved_input_ref=(
            conversation_text_by_resolved_input_ref
        ),
    )
    if not isinstance(meaning, ParsedSemanticQuestionMeaning):
        raise ValueError(
            "semantic Question Contract boundary requires complete meaning"
        )
    invocation = prompt_type(
        request,
        meaning=meaning,
    ).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context={},
            host=request.host,
        )
    )
    payload = captured_boundary(
        invocation,
        purpose="question_contract",
        provider=provider,
        model_key=model_key,
        assertion_context={
            "question_context_texts": [question],
            "frame_payload": frame_payload,
            "conversation_text_by_resolved_input_ref": dict(
                conversation_text_by_resolved_input_ref or {}
            ),
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
            "expected_aggregate_functions": list(expected_aggregate_functions),
            "required_boolean_use_sites": list(required_boolean_use_sites),
            "required_coverage_observation_input_ref": (
                required_coverage_observation_input_ref
            ),
            "expected_selection": expected_selection,
            "grouping_identifier_set": grouping_identifier_set,
            "reference_identifier_set": reference_identifier_set,
            "output_identifier_set": output_identifier_set,
            "expected_requested_output_kinds": list(
                expected_requested_output_kinds
            ),
            "forbid_identifier_ordering": forbid_identifier_ordering,
            "expected_instance_interpretation": expected_instance_interpretation,
        },
    )
    return payload


def _load_boundary_variant(path: Path):
    spec = importlib.util.spec_from_file_location(
        "question_contract_boundary_variant", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load boundary variant: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = getattr(module, "transform", None)
    if not callable(transform):
        raise ValueError("boundary variant must define transform(boundary)")
    return transform


if __name__ == "__main__":
    raise SystemExit(main())
