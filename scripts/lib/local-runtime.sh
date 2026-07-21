#!/usr/bin/env bash

fervis_load_env_file() {
  local env_file="$1"
  if [[ ! -f "$env_file" ]]; then
    return 0
  fi
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
}

fervis_resolve_compose_database_url() {
  local project_root="$1"
  local database_url="$2"
  DATABASE_URL_TO_RESOLVE="$database_url" python3 -I - "$project_root" <<'PY'
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from urllib.parse import urlparse, urlunparse

project_root = pathlib.Path(sys.argv[1])
database_url = os.environ["DATABASE_URL_TO_RESOLVE"]
parsed = urlparse(database_url)
hostname = parsed.hostname
port = parsed.port
if not hostname or port is None:
    print(database_url)
    raise SystemExit(0)

try:
    services = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(project_root),
            "config",
            "--services",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
except (FileNotFoundError, subprocess.CalledProcessError):
    print(database_url)
    raise SystemExit(0)

if hostname not in services:
    print(database_url)
    raise SystemExit(0)

try:
    published = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(project_root),
            "port",
            hostname,
            str(port),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1]
except (IndexError, subprocess.CalledProcessError):
    raise SystemExit(
        f"database service {hostname!r} has no running published port for {port}"
    )

published_url = urlparse(f"//{published}")
published_host = published_url.hostname
published_port = published_url.port
if published_host in {"0.0.0.0", "::"}:
    published_host = "127.0.0.1"
if not published_host or published_port is None:
    raise SystemExit(f"invalid published database address: {published!r}")

credentials = ""
if parsed.username is not None:
    credentials = parsed.username
    if parsed.password is not None:
        credentials += f":{parsed.password}"
    credentials += "@"
host = f"[{published_host}]" if ":" in published_host else published_host
print(urlunparse(parsed._replace(netloc=f"{credentials}{host}:{published_port}")))
PY
}
