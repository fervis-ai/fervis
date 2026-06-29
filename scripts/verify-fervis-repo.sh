#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_DIR="$REPO_DIR/python"
DESKTOP_DIR="$REPO_DIR/desktop-app"

if ! command -v uv >/dev/null 2>&1; then
  echo "verify: uv is required." >&2
  exit 127
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "verify: npm is required." >&2
  exit 127
fi

echo "==> Python dependencies"
uv --directory "$PYTHON_DIR" sync --extra dev

echo "==> Ruff"
uv --directory "$PYTHON_DIR" run ruff check src

echo "==> Fervis tests"
uv --directory "$PYTHON_DIR" run pytest

if [ ! -d "$DESKTOP_DIR/node_modules" ]; then
  echo "==> Desktop dependencies"
  npm --prefix "$DESKTOP_DIR" ci
fi

echo "==> Desktop app tests"
npm --prefix "$DESKTOP_DIR" test
