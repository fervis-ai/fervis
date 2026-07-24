#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/local-runtime.sh
. "$repo_root/scripts/lib/local-runtime.sh"

local_profile="${FERVIS_LOCAL_GOLDSET_PROFILE:-$repo_root/.fervis/local-goldset.env}"
fervis_load_env_file "$local_profile"
if [[ -n "${FERVIS_HOST_PROJECT_ROOT:-}" ]]; then
  fervis_load_env_file "$FERVIS_HOST_PROJECT_ROOT/.env"
fi
fervis_load_env_file "$repo_root/.env"

request_file="$repo_root/scripts/experiments/semantic_query_enrichment_nairobi_identity.json"
catalog_index=""
patch_file=""
assertion_file="$repo_root/scripts/experiments/semantic_query_enrichment_assertion.py"
runs=10
while (($#)); do
  case "$1" in
    --request-file)
      request_file="$2"
      shift 2
      ;;
    --runs)
      runs="$2"
      shift 2
      ;;
    --catalog-index)
      catalog_index="$2"
      shift 2
      ;;
    --patch-file)
      patch_file="$2"
      shift 2
      ;;
    --assertion-file)
      assertion_file="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

output_dir="$(mktemp -d "${TMPDIR:-/tmp}/fervis-semantic-qe.XXXXXX")"
boundary_file="$output_dir/boundary.json"
builder_command=(
  "$repo_root/python/.venv/bin/python"
  "$repo_root/scripts/experiments/build_semantic_query_enrichment_boundary.py" \
  --request-file "$request_file" \
  --output "$boundary_file"
)
if [[ -n "$catalog_index" ]]; then
  builder_command+=(--catalog-index "$catalog_index")
fi
"${builder_command[@]}"

command=(
  "$repo_root/python/.venv/bin/python"
  "$repo_root/scripts/run-model-step-stability.py"
  --boundary-file "$boundary_file"
  --runs "$runs"
  --workers 1
  --assertion-file "$assertion_file"
  --output-jsonl "$output_dir/results.jsonl"
  --label semantic-query-enrichment
)
if [[ -n "$patch_file" ]]; then
  command+=(--patch-file "$patch_file")
fi
exec "${command[@]}"
