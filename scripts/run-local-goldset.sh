#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run-local-goldset.sh [options]

Options:
  --case-ids CASES        Comma-separated goldset case ids. Defaults to
                          FERVIS_GOLDSET_CASE_IDS. Required.
  --project-root PATH     Host API project root. Defaults to FERVIS_HOST_PROJECT_ROOT
                          or the current directory when config/fervis.json exists.
  --suite REF             Goldset suite path or import entrypoint. Defaults to
                          FERVIS_GOLDSET_SUITE. Required.
  --suite-path REF        Alias for --suite.
  --tenant-id ID          Tenant id. Defaults to FERVIS_GOLDSET_TENANT_ID.
                          Required.
  --principal-id ID       Principal id. Defaults to FERVIS_GOLDSET_PRINCIPAL_ID.
                          Required.
  --database-url URL      Database URL override.
  --ledger-dir PATH       Output directory for per-case ledgers and command output.
  --python PATH           Python executable. Defaults to <project-root>/.venv/bin/python.
  --wait-seconds SECONDS  Per-case wait timeout. Defaults to 300.
  -h, --help              Show this help.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case_ids="${FERVIS_GOLDSET_CASE_IDS:-}"
project_root="${FERVIS_HOST_PROJECT_ROOT:-}"
suite_ref="${FERVIS_GOLDSET_SUITE:-}"
tenant_id="${FERVIS_GOLDSET_TENANT_ID:-}"
principal_id="${FERVIS_GOLDSET_PRINCIPAL_ID:-}"
database_url="${FERVIS_LOCAL_DATABASE_URL:-}"
ledger_dir=""
python_bin=""
wait_seconds="300"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-ids)
      case_ids="${2:-}"
      shift 2
      ;;
    --project-root)
      project_root="${2:-}"
      shift 2
      ;;
    --suite)
      suite_ref="${2:-}"
      shift 2
      ;;
    --suite-path)
      suite_ref="${2:-}"
      shift 2
      ;;
    --tenant-id)
      tenant_id="${2:-}"
      shift 2
      ;;
    --principal-id)
      principal_id="${2:-}"
      shift 2
      ;;
    --database-url)
      database_url="${2:-}"
      shift 2
      ;;
    --ledger-dir)
      ledger_dir="${2:-}"
      shift 2
      ;;
    --python)
      python_bin="${2:-}"
      shift 2
      ;;
    --wait-seconds)
      wait_seconds="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$project_root" && -f "config/fervis.json" ]]; then
  project_root="$(pwd)"
fi

if [[ -z "$project_root" || ! -d "$project_root" ]]; then
  echo "Host project root not found. Pass --project-root or set FERVIS_HOST_PROJECT_ROOT." >&2
  exit 2
fi

if [[ -f "$project_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$project_root/.env"
  set +a
fi

case_ids="${case_ids:-${FERVIS_GOLDSET_CASE_IDS:-}}"
suite_ref="${suite_ref:-${FERVIS_GOLDSET_SUITE:-}}"
tenant_id="${tenant_id:-${FERVIS_GOLDSET_TENANT_ID:-}}"
principal_id="${principal_id:-${FERVIS_GOLDSET_PRINCIPAL_ID:-}}"
database_url="${database_url:-${FERVIS_LOCAL_DATABASE_URL:-}}"

if [[ -z "$case_ids" ]]; then
  echo "Goldset case ids not found. Pass --case-ids or set FERVIS_GOLDSET_CASE_IDS." >&2
  usage >&2
  exit 2
fi
if [[ -z "$suite_ref" ]]; then
  echo "Goldset suite not found. Pass --suite or set FERVIS_GOLDSET_SUITE." >&2
  exit 2
fi
if [[ -z "$tenant_id" ]]; then
  echo "Tenant id not found. Pass --tenant-id or set FERVIS_GOLDSET_TENANT_ID." >&2
  exit 2
fi
if [[ -z "$principal_id" ]]; then
  echo "Principal id not found. Pass --principal-id or set FERVIS_GOLDSET_PRINCIPAL_ID." >&2
  exit 2
fi

if [[ -n "$database_url" ]]; then
  export DATABASE_URL="$database_url"
  export FERVIS_DATABASE_URL="$database_url"
fi

if [[ -z "$python_bin" ]]; then
  python_bin="$project_root/.venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  exit 2
fi

if [[ -z "$ledger_dir" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  ledger_dir="$project_root/.goldset-runs/local-$timestamp"
fi
mkdir -p "$ledger_dir"

goldset_src_path=""
if [[ "$suite_ref" != *:* ]]; then
  suite_path="$suite_ref"
  if [[ "$suite_path" != /* ]]; then
    suite_path="$project_root/$suite_path"
  fi
  suite_ref="$suite_path"
  if [[ -d "$suite_path/src" ]]; then
    goldset_src_path="$suite_path/src"
  fi
fi

export PYTHONSAFEPATH=1
export PYTHONPATH="$repo_root/python/src${goldset_src_path:+:$goldset_src_path}:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export FERVIS_GOLDSET_SUITE="$suite_ref"
export FERVIS_GOLDSET_TENANT_ID="$tenant_id"
export FERVIS_GOLDSET_PRINCIPAL_ID="$principal_id"
export FERVIS_GOLDSET_ADMIN_USER_ID="${FERVIS_GOLDSET_ADMIN_USER_ID:-$principal_id}"

"$python_bin" -P - "$repo_root" <<'PY'
from __future__ import annotations

import inspect
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
expected = (repo_root / "python" / "src" / "fervis").resolve()

import fervis
import fervis.lookup.question_contract.prompt as question_contract_prompt

actual = pathlib.Path(fervis.__file__).resolve()
prompt_path = pathlib.Path(inspect.getsourcefile(question_contract_prompt) or "").resolve()
if expected not in (actual, *actual.parents):
    raise SystemExit(f"wrong fervis import: {actual} does not come from {expected}")
if expected not in (prompt_path, *prompt_path.parents):
    raise SystemExit(
        f"wrong question_contract prompt import: {prompt_path} does not come from {expected}"
    )
PY

IFS=',' read -r -a cases <<< "$case_ids"
failed=0
for raw_case in "${cases[@]}"; do
  case_id="$(printf '%s' "$raw_case" | xargs)"
  [[ -n "$case_id" ]] || continue

  stdout_file="$ledger_dir/$case_id.stdout.json"
  stderr_file="$ledger_dir/$case_id.stderr.txt"
  ledger_file="$ledger_dir/$case_id.ledger.jsonl"

  "$python_bin" -P - "$case_id" <<'PY'
from __future__ import annotations

import os
import sys

from fervis.evaluation.goldsets.loader import load_goldset_suite

case_id = sys.argv[1]
suite = load_goldset_suite(os.environ["FERVIS_GOLDSET_SUITE"])
case = next((case for case in suite.cases if case.case_id == case_id), None)
if case is None:
    raise SystemExit(f"goldset case not found: {case_id}")
print(f"RUNNING: {case.case_id}, {case.question}")
PY

  set +e
  (
    cd "$project_root"
    FERVIS_GOLDSET_CASE_IDS="$case_id" "$python_bin" -P - "$ledger_file" "$wait_seconds" <<'PY'
from __future__ import annotations

import sys
from fervis.interfaces.cli.main import main

ledger_file, wait_seconds = sys.argv[1:3]
raise SystemExit(
    main(
        (
            "goldset",
            "run",
            "--ledger-file",
            ledger_file,
            "--wait-seconds",
            wait_seconds,
        )
    )
)
PY
  ) >"$stdout_file" 2>"$stderr_file"
  exit_code=$?
  set -e

  "$python_bin" -P - "$case_id" "$exit_code" "$stdout_file" "$stderr_file" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys

case_id, exit_code, stdout_file, stderr_file = sys.argv[1:5]
stdout_path = pathlib.Path(stdout_file)
stderr_path = pathlib.Path(stderr_file)
try:
    payload = json.loads(stdout_path.read_text())
    result = payload["payload"]["cases"][0]
    answer = result.get("answer")
    if str(result.get("status")) == "passed" and str(exit_code) == "0":
        print(f"PASS: {answer}")
    else:
        message = result.get("message") or answer or f"exit={exit_code}"
        print(f"FAIL: {message}")
except Exception as exc:
    print(f"FAIL: parse_error={exc}; exit={exit_code}")
    if stdout_path.exists():
        print(stdout_path.read_text()[-1200:])

stderr = stderr_path.read_text() if stderr_path.exists() else ""
if stderr.strip():
    print(f"STDERR {case_id}: {stderr.strip()[-1200:]}")
PY

  if [[ "$exit_code" -ne 0 ]]; then
    failed=1
  fi
done

echo "Artifacts: $ledger_dir"
exit "$failed"
