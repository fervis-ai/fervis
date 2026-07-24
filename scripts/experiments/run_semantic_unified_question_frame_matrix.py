#!/usr/bin/env python3
"""Run unified Question Contract frame cases through the shared model harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PYTHON = REPO_ROOT / "python" / ".venv" / "bin" / "python"
PYTHON_SRC = REPO_ROOT / "python" / "src"
if REPO_PYTHON.exists() and Path(sys.executable).resolve() != REPO_PYTHON.resolve():
    os.execv(str(REPO_PYTHON), (str(REPO_PYTHON), *sys.argv))
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_semantic_unified_question_frame_boundary import (  # noqa: E402
    build_boundary_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-failures", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.max_failures < 1:
        raise SystemExit("--runs and --max-failures must be positive")

    cases = _selected_cases(tuple(args.case_id))
    output_dir = args.output_dir or Path(
        tempfile.mkdtemp(prefix="fervis-unified-question-frame-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case in cases:
        case_id = str(case["id"])
        boundary_path = output_dir / f"{case_id}.boundary.json"
        results_path = output_dir / f"{case_id}.results.jsonl"
        boundary = build_boundary_payload(
            question=str(case["question"]),
            expected_request_count=int(case["expected_request_count"]),
            assertion_context={
                key: value
                for key, value in case.items()
                if key not in {"id", "question", "expected_request_count"}
            },
        )
        boundary_path.write_text(json.dumps(boundary, indent=2) + "\n")
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run-model-step-stability.py"),
                "--boundary-file",
                str(boundary_path),
                "--runs",
                str(args.runs),
                "--workers",
                "1",
                "--assertion-file",
                str(
                    Path(__file__).with_name(
                        "semantic_unified_question_frame_assertion.py"
                    )
                ),
                "--output-jsonl",
                str(results_path),
                "--label",
                f"unified-question-frame:{case_id}",
            ],
            check=False,
        )
        if result.returncode != 0:
            failures += 1
            if failures >= args.max_failures:
                break
    print(f"MATRIX: {len(cases) - failures}/{len(cases)} cases passed")
    print(f"ARTIFACTS: {output_dir}")
    return 1 if failures else 0


def _selected_cases(case_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    cases = json.loads(
        Path(__file__)
        .with_name("semantic_unified_question_frame_cases.json")
        .read_text()
    )
    if not isinstance(cases, list) or any(not isinstance(item, dict) for item in cases):
        raise ValueError("unified question frame cases must be an object array")
    by_id = {str(item["id"]): item for item in cases}
    if len(by_id) != len(cases):
        raise ValueError("unified question frame case IDs must be unique")
    if not case_ids:
        return cases
    unknown = set(case_ids) - set(by_id)
    if unknown:
        raise ValueError(f"unknown case ID: {sorted(unknown)[0]}")
    return [by_id[case_id] for case_id in case_ids]


if __name__ == "__main__":
    raise SystemExit(main())
