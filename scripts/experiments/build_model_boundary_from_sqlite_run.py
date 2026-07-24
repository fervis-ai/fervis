#!/usr/bin/env python3
"""Build a one-model-call boundary from artifacts in a local Fervis SQLite run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


_REQUIRED_ARTIFACTS = ("system_prompt", "prompt", "tool_spec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with sqlite3.connect(args.database) as connection:
        connection.row_factory = sqlite3.Row
        call = connection.execute(
            """
            SELECT
                call.model_call_id,
                call.provider,
                call.model_key,
                step.sequence
            FROM fervis_model_call AS call
            JOIN fervis_run_step AS step ON step.step_id = call.step_id
            WHERE call.run_id = ? AND step.step_key = ?
            ORDER BY call.call_index
            """,
            (args.run_id, args.step),
        ).fetchall()
        if len(call) != 1:
            raise ValueError("run step must contain exactly one model call")
        artifacts = {
            str(row["artifact_kind"]): str(row["content"])
            for row in connection.execute(
                """
                SELECT artifact_kind, content
                FROM fervis_run_artifact
                WHERE run_id = ? AND model_call_id = ?
                """,
                (args.run_id, call[0]["model_call_id"]),
            )
            if row["artifact_kind"] in _REQUIRED_ARTIFACTS
        }
    missing = set(_REQUIRED_ARTIFACTS) - artifacts.keys()
    if missing:
        raise ValueError(f"model call lacks required artifacts: {sorted(missing)}")

    boundary = {
        "source_run_id": args.run_id,
        "sequence": int(call[0]["sequence"]),
        "purpose": args.step,
        "provider": str(call[0]["provider"]),
        "model_key": str(call[0]["model_key"]),
        "system_prompt": artifacts["system_prompt"],
        "prompt": artifacts["prompt"],
        "tool_specs": json.loads(artifacts["tool_spec"]),
        "assertion_context": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(boundary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
