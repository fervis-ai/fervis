#!/usr/bin/env bash
set -euo pipefail

readonly SUMMARY_PATH="docs/architecture/summary.md"
readonly SMALL_CORE_PATH="docs/architecture/small-core.md"

forbidden_paths=()
while IFS= read -r -d '' path; do
  case "$path" in
    "$SUMMARY_PATH" | "$SMALL_CORE_PATH") ;;
    docs/*) forbidden_paths+=("$path") ;;
  esac
done < <(git ls-files -z -- docs/)

if ((${#forbidden_paths[@]} == 0)); then
  exit 0
fi

echo "pre-commit: only these docs files may be tracked:" >&2
echo "  $SUMMARY_PATH" >&2
echo "  $SMALL_CORE_PATH" >&2
echo "pre-commit: remove these paths from the index:" >&2
printf '  %s\n' "${forbidden_paths[@]}" >&2
exit 1
