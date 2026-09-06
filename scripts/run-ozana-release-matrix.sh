#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$repo_root/scripts/lib/ozana-cases.sh"
case_ids="$(IFS=,; printf '%s' "${ozana_release_cases[*]}")"
exec "$repo_root/scripts/run-ozana-regression.sh" --case-ids "$case_ids" --stable-runs 3 "$@"
