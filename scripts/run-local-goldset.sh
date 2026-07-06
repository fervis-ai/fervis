#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run-local-goldset.sh --case-ids case_a,case_b [options]

Options:
  --case-ids CASES        Comma-separated goldset case ids. Required.
  --project-root PATH     Host API project root. Defaults to FERVIS_HOST_PROJECT_ROOT
                          or ../../omnishades-code/unified-commerce/api when present.
  --suite-path PATH       Goldset suite directory. Defaults to FERVIS_GOLDSET_SUITE_PATH
                          or ../internal/evals/ozana-goldsets when present.
  --tenant-id ID          Tenant id. Defaults to local.
  --principal-id ID       Principal id. Defaults to FERVIS_GOLDSET_PRINCIPAL_ID from env/.env.
  --database-url URL      Database URL override.
  --ledger-dir PATH       Output directory for per-case ledgers and command output.
  --python PATH           Python executable. Defaults to <project-root>/.venv/bin/python.
  -h, --help              Show this help.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_project_root="$(cd "$repo_root/../.." && pwd)/omnishades-code/unified-commerce/api"
default_suite_path="$(cd "$repo_root/.." && pwd)/internal/evals/ozana-goldsets"

case_ids=""
project_root="${FERVIS_HOST_PROJECT_ROOT:-}"
suite_path="${FERVIS_GOLDSET_SUITE_PATH:-}"
tenant_id="local"
principal_id="${FERVIS_GOLDSET_PRINCIPAL_ID:-}"
database_url="${FERVIS_LOCAL_DATABASE_URL:-}"
ledger_dir=""
python_bin=""

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
    --suite-path)
      suite_path="${2:-}"
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

if [[ -z "$case_ids" ]]; then
  echo "--case-ids is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$project_root" && -d "$default_project_root" ]]; then
  project_root="$default_project_root"
fi
if [[ -z "$suite_path" && -d "$default_suite_path" ]]; then
  suite_path="$default_suite_path"
fi
if [[ -z "$project_root" || ! -d "$project_root" ]]; then
  echo "Host project root not found. Pass --project-root or set FERVIS_HOST_PROJECT_ROOT." >&2
  exit 2
fi
if [[ -z "$suite_path" || ! -d "$suite_path" ]]; then
  echo "Goldset suite path not found. Pass --suite-path or set FERVIS_GOLDSET_SUITE_PATH." >&2
  exit 2
fi

if [[ -f "$project_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$project_root/.env"
  set +a
fi

if [[ -z "$principal_id" ]]; then
  principal_id="${FERVIS_GOLDSET_PRINCIPAL_ID:-${FERVIS_GOLDSET_ADMIN_USER_ID:-}}"
fi
if [[ -z "$principal_id" ]]; then
  echo "Principal id not found. Pass --principal-id or set FERVIS_GOLDSET_PRINCIPAL_ID." >&2
  exit 2
fi

if [[ -n "$database_url" ]]; then
  export DATABASE_URL="$database_url"
elif [[ "${DATABASE_URL:-}" == *"@db:"* ]]; then
  export DATABASE_URL="postgres://postgres:pw@localhost:5433/brands_api_db"
  echo "Using local DATABASE_URL override because project .env points at Docker host 'db'."
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
if [[ -d "$suite_path/src" ]]; then
  goldset_src_path="$suite_path/src"
fi
if [[ ! -f "$suite_path/fervis_goldset.py" ]]; then
  if [[ -f "$suite_path/src/fervis_goldsets/ozana/suite.py" ]]; then
    suite_wrapper_dir="$ledger_dir/suite-entrypoint"
    mkdir -p "$suite_wrapper_dir"
    cat >"$suite_wrapper_dir/fervis_goldset.py" <<'PY'
from fervis_goldsets.ozana.suite import load_suite
PY
    suite_path="$suite_wrapper_dir"
  else
    echo "Goldset suite entrypoint not found: $suite_path/fervis_goldset.py" >&2
    exit 2
  fi
fi

export PYTHONSAFEPATH=1
export PYTHONPATH="$repo_root/python/src${goldset_src_path:+:$goldset_src_path}:$project_root${PYTHONPATH:+:$PYTHONPATH}"

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
print(f"fervis_import={actual}")
print(f"question_contract_prompt={prompt_path}")
PY

IFS=',' read -r -a cases <<< "$case_ids"
failed=0
for raw_case in "${cases[@]}"; do
  case_id="$(printf '%s' "$raw_case" | xargs)"
  [[ -n "$case_id" ]] || continue

  stdout_file="$ledger_dir/$case_id.stdout.json"
  stderr_file="$ledger_dir/$case_id.stderr.txt"
  ledger_file="$ledger_dir/$case_id.ledger.jsonl"

  echo "RUNNING $case_id"
  set +e
  (
    cd "$project_root"
    "$python_bin" -P - "$case_id" "$suite_path" "$tenant_id" "$principal_id" "$ledger_file" <<'PY'
from __future__ import annotations

import sys
from fervis.interfaces.cli.main import main

case_id, suite_path, tenant_id, principal_id, ledger_file = sys.argv[1:6]
raise SystemExit(
    main(
        (
            "goldset",
            "run",
            "--suite-path",
            suite_path,
            "--case-ids",
            case_id,
            "--tenant-id",
            tenant_id,
            "--principal-id",
            principal_id,
            "--ledger-file",
            ledger_file,
            "--wait-seconds",
            "300",
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
    print(
        "RESULT "
        f"{case_id} exit={exit_code} status={result.get('status')} "
        f"message={result.get('message')} run_id={result.get('run_id')}"
    )
    if answer not in ("", None):
        print(f"ANSWER {case_id}: {answer}")
except Exception as exc:
    print(f"RESULT {case_id} exit={exit_code} parse_error={exc}")
    if stdout_path.exists():
        print(stdout_path.read_text()[-1200:])

stderr = stderr_path.read_text() if stderr_path.exists() else ""
if stderr.strip():
    print(f"STDERR {case_id}: {stderr.strip()[-1200:]}")
PY

  if [[ "$exit_code" -ne 0 ]]; then
    failed=1
  fi
  echo "DONE $case_id"
done

echo "Artifacts: $ledger_dir"
exit "$failed"
