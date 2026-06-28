#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_APP_DIR="$ROOT_DIR/desktop-app"

if [ ! -d "$DESKTOP_APP_DIR/node_modules" ]; then
  npm --prefix "$DESKTOP_APP_DIR" ci
fi

npm --prefix "$DESKTOP_APP_DIR" run package:mac -- "$@"
