#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/local-runtime.sh
. "$repo_root/scripts/lib/local-runtime.sh"

local_profile="${FERVIS_LOCAL_GOLDSET_PROFILE:-$repo_root/.fervis/local-goldset.env}"
fervis_load_env_file "$local_profile"

project_root="${FERVIS_HOST_PROJECT_ROOT:-}"
if [[ -z "$project_root" || ! -d "$project_root" ]]; then
  echo "Host project root not found. Set FERVIS_HOST_PROJECT_ROOT in $local_profile." >&2
  exit 2
fi

fervis_load_env_file "$project_root/.env"
fervis_load_env_file "$repo_root/.env"

database_url="${FERVIS_LOCAL_DATABASE_URL:-${DATABASE_URL:-}}"
if [[ -n "$database_url" ]]; then
  database_url="$(fervis_resolve_compose_database_url "$project_root" "$database_url")"
  export DATABASE_URL="$database_url"
  export FERVIS_DATABASE_URL="$database_url"
fi

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  exit 2
fi

export PYTHONSAFEPATH=1
export PYTHONPATH="$repo_root/python/src:$project_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec "$python_bin" -P -m fervis.interfaces.cli.main "$@"
