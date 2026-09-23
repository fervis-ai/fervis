#!/bin/sh
# Build through the sdist and smoke-test the wheel outside the editable checkout.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_DIR="$REPO_DIR/python"
DIST_TMP=$(mktemp -d "${TMPDIR:-/tmp}/fervis-distribution.XXXXXX")
trap 'rm -rf "$DIST_TMP"' EXIT HUP INT TERM

uv build --project "$PYTHON_DIR" --out-dir "$DIST_TMP/dist"
uv venv --python "$PYTHON_DIR/.venv/bin/python" "$DIST_TMP/venv"
set -- "$DIST_TMP"/dist/*.whl
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "verify: expected exactly one built wheel." >&2
    exit 1
fi
uv pip install --python "$DIST_TMP/venv/bin/python" "$1"
uv pip check --python "$DIST_TMP/venv/bin/python"
(
    cd "$DIST_TMP"
    env -u PYTHONPATH "$DIST_TMP/venv/bin/python" -I - <<'PY'
from pathlib import Path
import sys
import fervis
from fervis.evaluation.goldsets import GoldsetCase, GoldsetSuite
from fervis.lookup.answer_program.values import IdentityMatchEvidence
from fervis.lookup.source_binding.association_choices import association_choices

assert Path(fervis.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert GoldsetCase and GoldsetSuite and IdentityMatchEvidence and association_choices
print("Installed distribution imports passed.")
PY
    env -u PYTHONPATH "$DIST_TMP/venv/bin/fervis" --help >/dev/null
)
echo "Installed distribution CLI passed."
