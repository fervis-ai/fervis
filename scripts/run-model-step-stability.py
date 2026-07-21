#!/usr/bin/env python3
"""Replay one persisted model-step boundary with controlled contract changes.

The input is the ``index.json`` produced by ``fervis debug prompts``. Only the
selected model step is called. Upstream prompts and outputs are not regenerated.

Patch-file shape::

    {
      "prompt_replacements": [
        {"old": "old instruction", "new": "new instruction", "expected_count": 1}
      ],
      "system_prompt_replacements": [],
      "tool_spec_patches": [
        {"tool_name": "submit_step", "op": "replace", "path": "/description", "value": "..."}
      ],
      "schema_patches": [
        {"tool_name": "submit_step", "op": "add", "path": "/properties/new_field", "value": {"type": "string"}}
      ]
    }

``tool_spec_patches`` paths are relative to the serialized ToolSpec.
``schema_patches`` paths are relative to that tool's ``input_schema``.

An optional assertion file may define::

    def validate(arguments: dict, context: dict) -> list[str]:
        ...

Return an empty list for a semantic pass. Provider and strict-schema failures
are always failures before the assertion runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python" / "src"
REPO_PYTHON = REPO_ROOT / "python" / ".venv" / "bin" / "python"
if REPO_PYTHON.exists() and Path(sys.executable).resolve() != REPO_PYTHON.resolve():
    os.execv(str(REPO_PYTHON), (str(REPO_PYTHON), *sys.argv))
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from fervis.model_io.backbone.dto import ToolSpec  # noqa: E402
from fervis.model_io.backbone.factory import build_provider_backbone  # noqa: E402
from fervis.model_io.structured_output.generation import (  # noqa: E402
    generate_one_of_tool_output,
)


Assertion = Callable[[dict[str, Any], dict[str, Any]], list[str]]


@dataclass(frozen=True)
class ExperimentBoundary:
    source_run_id: str
    sequence: int
    purpose: str
    provider: str
    model_key: str
    system_prompt: str
    prompt: str
    tool_specs: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class StabilityResult:
    run_number: int
    arguments: dict[str, Any] | None
    errors: tuple[str, ...]
    arguments_hash: str = ""

    @property
    def passed(self) -> bool:
        return not self.errors


class _ModelPort:
    def __init__(self, *, provider: str, model_key: str) -> None:
        self._router = build_provider_backbone(provider).model_router
        self._model_key = model_key

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        return self._router.generate(model_key=self._model_key, **kwargs)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--step", required=True, help="Persisted model-turn purpose")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--patch-file",
        type=Path,
        action="append",
        help="Patch file to apply; repeat to layer one controlled change",
    )
    parser.add_argument("--assertion-file", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--model-key")
    parser.add_argument("--max-thinking-tokens", type=int, default=0)
    parser.add_argument("--label")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--enforce-structured-stability", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.runs < 1 or args.workers < 1:
        raise SystemExit("--runs and --workers must be positive")
    patch = _combined_patches(args.patch_file or [])
    boundary = _load_boundary(
        args.index,
        purpose=args.step,
        sequence=args.sequence,
        patch=patch,
        provider_override=args.provider,
        model_key_override=args.model_key,
    )
    assertion = _load_assertion(args.assertion_file)
    label = args.label or boundary.purpose
    model_port = _ModelPort(
        provider=boundary.provider,
        model_key=boundary.model_key,
    )
    print_lock = Lock()

    def run(run_number: int) -> StabilityResult:
        try:
            output = generate_one_of_tool_output(
                model_port=model_port,
                provider=boundary.provider,
                system_prompt=boundary.system_prompt,
                prompt=boundary.prompt,
                max_thinking_tokens=args.max_thinking_tokens,
                tool_specs=boundary.tool_specs,
            )
            context = {
                "label": label,
                "run_number": run_number,
                "source_run_id": boundary.source_run_id,
                "sequence": boundary.sequence,
                "purpose": boundary.purpose,
            }
            errors = tuple(assertion(output.arguments, context))
            canonical = json.dumps(
                output.arguments,
                sort_keys=True,
                separators=(",", ":"),
            )
            return StabilityResult(
                run_number=run_number,
                arguments=output.arguments,
                errors=errors,
                arguments_hash=sha256(canonical.encode()).hexdigest(),
            )
        except Exception as exc:  # provider/schema boundary is reported per run
            cause = f"; cause={exc.__cause__!r}" if exc.__cause__ is not None else ""
            error_code = str(getattr(exc, "error_code", "") or "")
            error_context = getattr(exc, "error_context", None)
            provider_details = (
                f"; error_code={error_code}; "
                f"error_context={json.dumps(error_context, sort_keys=True, default=str)}"
                if error_code or error_context
                else ""
            )
            return StabilityResult(
                run_number=run_number,
                arguments=None,
                errors=(
                    f"{type(exc).__name__}: {exc!r}{cause}{provider_details}",
                ),
            )

    results: list[StabilityResult] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, args.runs)) as executor:
        futures = {executor.submit(run, number): number for number in range(1, args.runs + 1)}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            message = "PASS" if result.passed else "FAIL: " + "; ".join(result.errors)
            with print_lock:
                print(
                    f"STABILITY RUN {result.run_number}/{args.runs} {message}",
                    flush=True,
                )

    ordered = sorted(results, key=lambda item: item.run_number)
    if args.output_jsonl:
        _write_results(args.output_jsonl, ordered)
    failures = sum(not item.passed for item in ordered)
    hashes = {item.arguments_hash for item in ordered if item.arguments_hash}
    stability_failure = args.enforce_structured_stability and len(hashes) > 1
    print(
        f"SUMMARY {label}: {args.runs - failures}/{args.runs} passed; "
        f"structured_variants={len(hashes)}",
        flush=True,
    )
    if stability_failure:
        print("FAIL: structured outputs were not identical", flush=True)
    return 1 if failures or stability_failure else 0


def _combined_patches(paths: list[Path]) -> dict[str, list[object]]:
    combined: dict[str, list[object]] = {}
    for path in paths:
        for key, value in _json_object(path).items():
            if not isinstance(value, list):
                raise ValueError(f"patch section {key!r} must be an array")
            combined.setdefault(key, []).extend(value)
    return combined


def _load_boundary(
    index_path: Path,
    *,
    purpose: str,
    sequence: int | None,
    patch: dict[str, Any],
    provider_override: str | None,
    model_key_override: str | None,
) -> ExperimentBoundary:
    index = _json_object(index_path)
    turns = [
        turn
        for run in index.get("runs", [])
        for turn in run.get("model_turns", [])
        if turn.get("purpose") == purpose
        and (sequence is None or turn.get("sequence") == sequence)
    ]
    if len(turns) != 1:
        raise ValueError(
            f"expected one {purpose!r} turn, found {len(turns)}; use --sequence"
        )
    turn = turns[0]
    prompt = _replace_text(
        str(turn.get("prompt") or ""),
        patch.get("prompt_replacements", []),
        surface="prompt",
    )
    system_prompt = _replace_text(
        str(turn.get("system_prompt") or ""),
        patch.get("system_prompt_replacements", []),
        surface="system prompt",
    )
    raw_specs = [dict(item) for item in turn.get("tool_specs", [])]
    _apply_named_tool_patches(raw_specs, patch.get("tool_spec_patches", []), schema=False)
    _apply_named_tool_patches(raw_specs, patch.get("schema_patches", []), schema=True)
    _remove_named_schema_properties(
        raw_specs,
        patch.get("schema_property_removals", []),
    )
    tool_specs = tuple(_tool_spec(item) for item in raw_specs)
    if not tool_specs:
        raise ValueError("selected turn has no tool specs")
    provider = str(provider_override or turn.get("provider") or "").strip()
    model_key = str(model_key_override or turn.get("model_key") or "").strip()
    if not provider or not model_key:
        raise ValueError("selected turn requires provider and model_key")
    return ExperimentBoundary(
        source_run_id=str(turn.get("run_id") or ""),
        sequence=int(turn.get("sequence") or 0),
        purpose=purpose,
        provider=provider,
        model_key=model_key,
        system_prompt=system_prompt,
        prompt=prompt,
        tool_specs=tool_specs,
    )


def _replace_text(text: str, replacements: object, *, surface: str) -> str:
    items = replacements if isinstance(replacements, list) else []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{surface} replacement must be an object")
        old = str(item.get("old") or "")
        new = str(item.get("new") or "")
        expected_count = int(item.get("expected_count", 1))
        actual_count = text.count(old)
        if not old or actual_count != expected_count:
            raise ValueError(
                f"{surface} replacement expected {expected_count} matches, found {actual_count}"
            )
        text = text.replace(old, new)
    return text


def _apply_named_tool_patches(
    specs: list[dict[str, Any]],
    patches: object,
    *,
    schema: bool,
) -> None:
    items = patches if isinstance(patches, list) else []
    specs_by_name = {str(spec.get("name") or ""): spec for spec in specs}
    for patch in items:
        if not isinstance(patch, dict):
            raise ValueError("tool patch must be an object")
        tool_name = str(patch.get("tool_name") or "")
        spec = specs_by_name.get(tool_name)
        if spec is None:
            raise ValueError(f"tool patch references unknown tool {tool_name!r}")
        target = spec.get("input_schema") if schema else spec
        if not isinstance(target, (dict, list)):
            raise ValueError("tool patch target is not structured JSON")
        _apply_json_patch(target, patch)


def _remove_named_schema_properties(
    specs: list[dict[str, Any]],
    removals: object,
) -> None:
    items = removals if isinstance(removals, list) else []
    specs_by_name = {str(spec.get("name") or ""): spec for spec in specs}
    for removal in items:
        if not isinstance(removal, dict):
            raise ValueError("schema property removal must be an object")
        tool_name = str(removal.get("tool_name") or "")
        spec = specs_by_name.get(tool_name)
        if spec is None:
            raise ValueError(
                f"schema property removal references unknown tool {tool_name!r}"
            )
        names = {
            str(name)
            for name in removal.get("property_names", [])
            if str(name)
        }
        if not names:
            raise ValueError("schema property removal requires property names")
        schema = spec.get("input_schema")
        if not isinstance(schema, dict):
            raise ValueError("tool schema is not an object")
        removed = _remove_schema_properties(schema, names=names)
        if removed != names:
            missing = ", ".join(sorted(names - removed))
            raise ValueError(f"schema properties were not found: {missing}")


def _remove_schema_properties(
    value: object,
    *,
    names: set[str],
) -> set[str]:
    removed: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            for name in names & properties.keys():
                del properties[name]
                removed.add(name)
            required = value.get("required")
            if isinstance(required, list):
                value["required"] = [name for name in required if name not in names]
        for child in value.values():
            removed.update(_remove_schema_properties(child, names=names))
    elif isinstance(value, list):
        for child in value:
            removed.update(_remove_schema_properties(child, names=names))
    return removed


def _apply_json_patch(target: dict[str, Any] | list[Any], patch: dict[str, Any]) -> None:
    operation = str(patch.get("op") or "")
    path = str(patch.get("path") or "")
    if operation not in {"add", "replace", "remove", "move", "move_before"}:
        raise ValueError(f"unsupported patch operation {operation!r}")
    if operation == "move_before":
        source_path = str(patch.get("from") or "")
        _move_json_object_member_before(target, source_path, path)
        return
    if operation == "move":
        source_path = str(patch.get("from") or "")
        value = _remove_json_pointer(target, source_path)
        _add_json_pointer(target, path, value)
        return
    parent, token = _json_pointer_parent(target, path)
    if isinstance(parent, list):
        index = len(parent) if token == "-" else int(token)
        if operation == "add":
            parent.insert(index, patch.get("value"))
        elif operation == "replace":
            parent[index] = patch.get("value")
        else:
            del parent[index]
        return
    if operation == "add":
        parent[token] = patch.get("value")
    elif operation == "replace":
        if token not in parent:
            raise ValueError(f"replace path does not exist: {path}")
        parent[token] = patch.get("value")
    else:
        if token not in parent:
            raise ValueError(f"remove path does not exist: {path}")
        del parent[token]


def _move_json_object_member_before(
    target: dict[str, Any] | list[Any],
    source_path: str,
    before_path: str,
) -> None:
    source_parent, source_token = _json_pointer_parent(target, source_path)
    before_parent, before_token = _json_pointer_parent(target, before_path)
    if not isinstance(source_parent, dict) or source_parent is not before_parent:
        raise ValueError("move_before requires members of the same JSON object")
    if source_token not in source_parent or before_token not in source_parent:
        raise ValueError("move_before path does not exist")
    value = source_parent[source_token]
    reordered: dict[str, Any] = {}
    for key, item in source_parent.items():
        if key == source_token:
            continue
        if key == before_token:
            reordered[source_token] = value
        reordered[key] = item
    source_parent.clear()
    source_parent.update(reordered)


def _remove_json_pointer(
    target: dict[str, Any] | list[Any],
    path: str,
) -> Any:
    parent, token = _json_pointer_parent(target, path)
    if isinstance(parent, list):
        return parent.pop(int(token))
    if token not in parent:
        raise ValueError(f"move source path does not exist: {path}")
    return parent.pop(token)


def _add_json_pointer(
    target: dict[str, Any] | list[Any],
    path: str,
    value: Any,
) -> None:
    parent, token = _json_pointer_parent(target, path)
    if isinstance(parent, list):
        index = len(parent) if token == "-" else int(token)
        parent.insert(index, value)
        return
    parent[token] = value


def _json_pointer_parent(
    target: dict[str, Any] | list[Any],
    path: str,
) -> tuple[Any, str]:
    tokens = (
        [_pointer_token(token) for token in path.split("/")[1:]]
        if path.startswith("/")
        else []
    )
    if not tokens:
        raise ValueError("patch path must address a child value")
    parent: Any = target
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    return parent, tokens[-1]


def _pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _tool_spec(payload: dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        name=str(payload.get("name") or ""),
        description=str(payload.get("description") or ""),
        input_schema=dict(payload.get("input_schema") or {}),
        input_examples=tuple(payload.get("input_examples") or ()),
        json_object_arguments=tuple(payload.get("json_object_arguments") or ()),
        strict=bool(payload.get("strict", True)),
    )


def _load_assertion(path: Path | None) -> Assertion:
    if path is None:
        return lambda _arguments, _context: []
    spec = importlib.util.spec_from_file_location("fervis_stability_assertion", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load assertion file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate = getattr(module, "validate", None)
    if not callable(validate):
        raise ValueError("assertion file must define validate(arguments, context)")
    return validate


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_results(path: Path, results: list[StabilityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "run_number": result.run_number,
                        "passed": result.passed,
                        "errors": list(result.errors),
                        "arguments_hash": result.arguments_hash,
                        "arguments": result.arguments,
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


if __name__ == "__main__":
    raise SystemExit(main())
