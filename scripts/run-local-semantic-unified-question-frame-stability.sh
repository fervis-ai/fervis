#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/local-runtime.sh
. "$repo_root/scripts/lib/local-runtime.sh"

local_profile="${FERVIS_LOCAL_GOLDSET_PROFILE:-$repo_root/.fervis/local-goldset.env}"
fervis_load_env_file "$local_profile"

fervis_load_env_file "$repo_root/.env"
if [[ -n "${FERVIS_HOST_PROJECT_ROOT:-}" ]]; then
  fervis_load_env_file "$FERVIS_HOST_PROJECT_ROOT/.env"
fi

cd "$repo_root"
exec "$repo_root/python/.venv/bin/python" \
  -m scripts.experiments.question_frame.matrix \
  "$@"
