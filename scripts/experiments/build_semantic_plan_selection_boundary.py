#!/usr/bin/env python3
"""Build a reusable semantic Plan Selection stability boundary."""

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

from build_semantic_read_eligibility_boundary import (  # noqa: E402
    build_request as read_eligibility_request,
)
from fervis.lookup.available_sources import (  # noqa: E402
    build_available_source_catalog,
)
from fervis.lookup.answer_program.values import FactValue  # noqa: E402
from fervis.lookup.canonical_data import (  # noqa: E402
    EntityKeyComponentValue,
    EntityKeyValue,
)
from fervis.lookup.grounding import CanonicalInputValue  # noqa: E402
from fervis.lookup.plan_selection.semantic import (  # noqa: E402
    SemanticPlanSelectionRequest,
)
from fervis.lookup.plan_selection.semantic_prompt import (  # noqa: E402
    SemanticPlanSelectionTurnPrompt,
)
from fervis.lookup.read_eligibility.semantic import (  # noqa: E402
    ReadRequirementAssessment,
    SemanticReadDecision,
    SemanticReadEligibilityResult,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model-key", default="openai:gpt-5.4-mini")
    args = parser.parse_args()
    payload = json.loads(args.request_file.read_text())
    request = build_request(payload)
    invocation = SemanticPlanSelectionTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=str(payload["question"]),
            conversation_context={},
        )
    )
    boundary = {
        "source_run_id": "semantic-plan-selection-experiment",
        "sequence": 1,
        "purpose": "plan_selection",
        "provider": args.provider,
        "model_key": args.model_key,
        "system_prompt": invocation.system_prompt,
        "prompt": invocation.prompt_text,
        "tool_specs": [asdict(item) for item in invocation.tool_specs],
        "assertion_context": payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(boundary, indent=2) + "\n")
    return 0


def build_request(payload: dict[str, Any]) -> SemanticPlanSelectionRequest:
    read_request = read_eligibility_request(payload)
    read_result = SemanticReadEligibilityResult(
        read_assessments=tuple(
            ReadRequirementAssessment(
                requested_fact_id=index.requested_fact_id,
                candidate_ref=candidate.candidate_ref,
                source_refs=candidate.source_refs,
                read_id=candidate.read_id,
                relevant_field_refs=tuple(
                    field.field_ref for field in candidate.fields
                ),
                assessment_basis="The source exposes every shown requirement.",
                decision=SemanticReadDecision.RETAIN,
            )
            for index in read_request.indexes
            for candidate in read_request.read_candidates
        ),
        identity_outcomes=(),
    )
    available = build_available_source_catalog(
        read_request.source_catalog,
        read_eligibility=read_result,
    )
    canonical_values = tuple(
        CanonicalInputValue(
            canonical_value_id=f"canonical_{task.input_ref}",
            input_ref=task.input_ref,
            use_refs=task.use_refs,
            typed_value=FactValue.identity(
                id=f"canonical_{task.input_ref}",
                known_input_id=task.input_ref,
                key=EntityKeyValue(
                    entity_kind=str(payload["resource_type"]),
                    key_id="primary_key",
                    components=(
                        EntityKeyComponentValue(
                            str(payload["key_component"]),
                            f"{payload['resource_type']}-canonical-1",
                        ),
                    ),
                ),
                display_value=str(payload["operand"]),
                proof_refs=(f"resolver:{payload['expected_read']}",),
            ),
            certification_refs=(f"resolver:{payload['expected_read']}",),
        )
        for task in read_request.identity_tasks
    )
    return SemanticPlanSelectionRequest(
        indexes=read_request.indexes,
        source_catalog=available,
        canonical_values=canonical_values,
    )


if __name__ == "__main__":
    raise SystemExit(main())
