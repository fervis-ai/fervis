"""Shared framework-mount contracts and file transactions."""

from __future__ import annotations

import ast
import io
import os
import tempfile
import tokenize
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fervis.interfaces.agent.actions import run_init_action

from .bindings import AssignedNode, top_level_assignments


@dataclass(frozen=True)
class FileChange:
    path: str
    changed: bool


@dataclass(frozen=True)
class BlockedPatch:
    path: str
    reason: str


@dataclass(frozen=True)
class PlannedFilePatch:
    path: str
    absolute_path: Path
    original: str
    updated: str
    encoding: str

    @property
    def changed(self) -> bool:
        return self.updated != self.original

    def apply(self) -> FileChange:
        if self.changed:
            _atomic_write_text(self.absolute_path, self.updated, encoding=self.encoding)
        return FileChange(self.path, changed=self.changed)


@dataclass(frozen=True)
class FrameworkPatchResult:
    changed_files: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    blocked: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked)


@dataclass(frozen=True)
class FrameworkCheck:
    id: str
    passed: bool
    message: str
    fix: dict[str, object] | None = None


@dataclass(frozen=True)
class PythonFile:
    relative_path: str
    absolute_path: Path
    text: str
    tree: ast.Module
    encoding: str

    @classmethod
    def load(cls, root: Path, relative_path: str) -> PythonFile | BlockedPatch:
        absolute_path = root / relative_path
        if not absolute_path.is_file():
            return BlockedPatch(relative_path, f"{relative_path} was not found.")
        try:
            raw = absolute_path.read_bytes()
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            text = raw.decode(encoding)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            return BlockedPatch(
                relative_path,
                f"{relative_path} could not be decoded as Python source: {exc}.",
            )
        try:
            tree = parse_python_source(text)
        except SyntaxError as exc:
            return BlockedPatch(
                relative_path,
                f"{relative_path} is not valid Python: {exc.msg}.",
            )
        return cls(relative_path, absolute_path, text, tree, encoding)

    def plan_validated(
        self,
        updated: str,
        *,
        validate: Callable[[ast.Module], bool],
    ) -> PlannedFilePatch | BlockedPatch:
        try:
            tree = parse_python_source(updated)
        except SyntaxError as exc:
            return BlockedPatch(
                self.relative_path,
                f"{self.relative_path} would not remain valid Python: {exc.msg}.",
            )
        if not validate(tree):
            return BlockedPatch(
                self.relative_path,
                f"{self.relative_path} would not contain the expected Fervis hook after patching.",
            )
        return PlannedFilePatch(
            path=self.relative_path,
            absolute_path=self.absolute_path,
            original=self.text,
            updated=updated,
            encoding=self.encoding,
        )


def plan_then_apply(
    plans_or_blocks: list[PlannedFilePatch | BlockedPatch],
) -> FrameworkPatchResult:
    blocked = [
        (item.path, item.reason)
        for item in plans_or_blocks
        if isinstance(item, BlockedPatch)
    ]
    if blocked:
        return FrameworkPatchResult(blocked=blocked)

    changed: list[str] = []
    skipped: list[str] = []
    applied: list[PlannedFilePatch] = []
    try:
        for item in plans_or_blocks:
            if isinstance(item, BlockedPatch):
                continue
            result = item.apply()
            applied.append(item)
            if result.changed:
                changed.append(result.path)
            else:
                skipped.append(result.path)
    except OSError as exc:
        for item in reversed(applied):
            if item.changed:
                _atomic_write_text(
                    item.absolute_path,
                    item.original,
                    encoding=item.encoding,
                )
        return FrameworkPatchResult(
            blocked=[
                (
                    "framework hook files",
                    f"Failed to write framework hook files; reverted changes: {exc}",
                )
            ]
        )
    return FrameworkPatchResult(changed_files=changed, skipped_existing=skipped)


def single_top_level_assignment(
    tree: ast.Module,
    name: str,
    *,
    path: str,
) -> AssignedNode | BlockedPatch:
    matches = top_level_assignments(tree, name)
    if not matches:
        return BlockedPatch(path, f"Could not find a top-level {name} assignment.")
    if len(matches) > 1:
        return BlockedPatch(path, f"{name} is assigned more than once.")
    return matches[0]


def init_fix(framework: str) -> dict[str, object]:
    return run_init_action(framework)


def parse_python_source(text: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text)


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    mode = path.stat().st_mode if path.exists() else None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding=encoding,
        newline="",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(text)
    if mode is not None:
        os.chmod(temporary_path, mode)
    os.replace(temporary_path, path)
