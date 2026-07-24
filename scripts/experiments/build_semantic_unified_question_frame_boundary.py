#!/usr/bin/env python3
"""Build the production unified requested-meaning and supplied-value boundary."""

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
    QuestionContractRequest,
    SemanticQuestionFrameTurnPrompt,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--expected-request-count", type=int, required=True)
    parser.add_argument("--assertion-context", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assertion_context = (
        json.loads(args.assertion_context.read_text())
        if args.assertion_context is not None
        else {}
    )
    payload = build_boundary_payload(
        question=args.question,
        expected_request_count=args.expected_request_count,
        assertion_context=assertion_context,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def build_boundary_payload(
    *,
    question: str,
    expected_request_count: int,
    assertion_context: dict[str, object] | None = None,
    provider: str = "openai",
    model_key: str = "openai:gpt-5.4-mini",
) -> dict[str, object]:
    request = QuestionContractRequest(
        current_question=question,
        conversation_context={},
    )
    invocation = SemanticQuestionFrameTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context={},
            host=request.host,
        )
    )
    return {
        "source_run_id": "semantic-question-frame-production-gate",
        "sequence": 1,
        "purpose": "question_frame",
        "provider": provider,
        "model_key": model_key,
        "system_prompt": invocation.system_prompt,
        "prompt": invocation.prompt_text,
        "tool_specs": [asdict(item) for item in invocation.tool_specs],
        "assertion_context": {
            "question": question,
            "expected_request_count": expected_request_count,
            **dict(assertion_context or {}),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
