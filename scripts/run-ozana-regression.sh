#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

. "$repo_root/scripts/lib/ozana-cases.sh"
cases=("${ozana_regression_cases[@]}")

case_ids="$(IFS=,; printf '%s' "${cases[*]}")"

export FERVIS_GOLDSET_CASE_IDS="$case_ids"
export FERVIS_GOLDSET_SUITE="fervis_goldsets.ozana.suite:load_suite"
export FERVIS_GOLDSET_TENANT_ID="${FERVIS_GOLDSET_TENANT_ID:-local-goldset}"

exec "$repo_root/scripts/run-local-goldset.sh" "$@"
