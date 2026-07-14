"""FastAPI framework mounting."""

from __future__ import annotations

import ast
from pathlib import Path

from ..discovery import ProjectInspection
from fervis.interfaces.agent.actions import edit_config_action
from ..importing import project_python_import_paths, project_python_source_roots
from ..integration import FastAPIAppSource, FervisConfig
from ..source_paths import normalize_source_path_prefixes
from .bindings import (
    AssignedNode,
    directly_assigns_name,
    has_import,
    import_before,
    is_configured_fervis_call,
    module_import_before,
    top_level_assignment_after_line,
    top_level_name_shadowed_before_line,
)
from .common import (
    BlockedPatch,
    FrameworkCheck,
    FrameworkPatchResult,
    PlannedFilePatch,
    PythonFile,
    init_fix,
    parse_python_source,
    plan_then_apply,
)
from .fastapi_sources import fastapi_source_path_prefixes
from .source import (
    insert_before_node,
    insert_import_line,
    newline_for,
    rstrip_line_endings,
)


FASTAPI_IMPORT = "from fervis import configured_fervis"
FASTAPI_MOUNT = "configured_fervis().mount(app)"


def patch_fastapi(
    project: ProjectInspection,
    *,
    app_factory: str | None = None,
) -> FrameworkPatchResult:
    if app_factory:
        target = fastapi_factory_target(project.root_path, app_factory)
        if isinstance(target, BlockedPatch):
            return FrameworkPatchResult(blocked=[(target.path, target.reason)])
        return plan_then_apply(
            [_plan_fastapi_factory(project.root_path, target[0], target[1])]
        )
    entrypoint = fastapi_entrypoint(project.root_path)
    if isinstance(entrypoint, BlockedPatch):
        return FrameworkPatchResult(blocked=[(entrypoint.path, entrypoint.reason)])
    return plan_then_apply([_plan_fastapi_entrypoint(project.root_path, entrypoint)])


def fastapi_checks(
    project: ProjectInspection,
    config: FervisConfig,
) -> list[FrameworkCheck]:
    mount_valid = _configured_source_mount_valid(project, config)
    return [
        FrameworkCheck(
            id="framework.fastapi.mount",
            passed=mount_valid,
            message=(
                "Configured FastAPI source imports Fervis and calls "
                "configured_fervis().mount(app)."
            ),
            fix=init_fix(project.framework),
        ),
        FrameworkCheck(
            id="source.fastapi.entrypoint",
            passed=mount_valid,
            message="FastAPI source import path matches the mounted app entrypoint.",
            fix=edit_config_action(),
        ),
    ]


def _configured_source_mount_valid(
    project: ProjectInspection,
    config: FervisConfig,
) -> bool:
    for source in config.sources:
        if not isinstance(source, FastAPIAppSource):
            continue
        for import_path in source.import_paths:
            if _configured_import_mount_valid(project.root_path, import_path):
                return True
    return False


def _configured_import_mount_valid(root: Path, import_path: str) -> bool:
    if ":" not in import_path:
        return False
    module_path, object_name = import_path.split(":", 1)
    source_path = _source_path_for_module(root, module_path)
    if isinstance(source_path, BlockedPatch):
        return False
    loaded = PythonFile.load(root, source_path)
    if isinstance(loaded, BlockedPatch):
        return False
    if object_name == "app":
        return fastapi_entrypoint_contains_hooks(loaded.tree)
    if "." not in object_name and object_name:
        return _fastapi_factory_contains_hooks(loaded.tree, object_name)
    return False


def fastapi_source_schema(
    root: Path,
    *,
    path_prefixes: tuple[str, ...] | None = None,
) -> dict[str, object] | BlockedPatch:
    entrypoint = fastapi_entrypoint(root)
    if isinstance(entrypoint, BlockedPatch):
        return entrypoint
    preflight = _plan_fastapi_entrypoint(root, entrypoint)
    if isinstance(preflight, BlockedPatch):
        return preflight
    module = _module_path_for_entrypoint(root, entrypoint)
    if isinstance(module, BlockedPatch):
        return module
    if path_prefixes is None:
        source_prefixes = fastapi_source_path_prefixes(root, f"{module}:app")
        if isinstance(source_prefixes, BlockedPatch):
            return source_prefixes
    else:
        try:
            source_prefixes = normalize_source_path_prefixes(path_prefixes)
        except ValueError as exc:
            return BlockedPatch(
                "config/fervis.json",
                f"FastAPI source path prefixes are invalid: {exc}",
            )
    return {
        "kind": "fastapi_app",
        "name": "default",
        "import_paths": [f"{module}:app"],
        "path_prefixes": list(source_prefixes),
    }


def fastapi_factory_source_schema(
    root: Path,
    app_factory: str,
    *,
    path_prefixes: tuple[str, ...] | None = None,
) -> dict[str, object] | BlockedPatch:
    target = fastapi_factory_target(root, app_factory)
    if isinstance(target, BlockedPatch):
        return target
    entrypoint, factory_name = target
    preflight = _plan_fastapi_factory(root, entrypoint, factory_name)
    if isinstance(preflight, BlockedPatch):
        return preflight
    if path_prefixes is None:
        del entrypoint, factory_name
        inferred = fastapi_source_path_prefixes(root, app_factory)
        if isinstance(inferred, BlockedPatch):
            return inferred
        source_prefixes = inferred
    else:
        try:
            source_prefixes = normalize_source_path_prefixes(path_prefixes)
        except ValueError as exc:
            return BlockedPatch(
                "config/fervis.json",
                f"FastAPI source path prefixes are invalid: {exc}",
            )
    return {
        "kind": "fastapi_app",
        "name": "default",
        "import_paths": [app_factory],
        "path_prefixes": list(source_prefixes),
    }


def fastapi_factory_target(
    root: Path,
    app_factory: str,
) -> tuple[str, str] | BlockedPatch:
    if ":" not in app_factory:
        return BlockedPatch(
            "config/fervis.json",
            "FastAPI app factory must use module:function import path syntax.",
        )
    module_path, factory_name = app_factory.split(":", 1)
    if "." in factory_name or not factory_name:
        return BlockedPatch(
            "config/fervis.json",
            "FastAPI app factory must reference one top-level function.",
        )
    relative_path = _source_path_for_module(root, module_path)
    if isinstance(relative_path, BlockedPatch):
        return relative_path
    return relative_path, factory_name


def fastapi_entrypoint(root: Path) -> str | BlockedPatch:
    try:
        source_roots = project_python_source_roots(root)
    except ValueError as exc:
        return BlockedPatch("pyproject.toml", str(exc))
    matches: list[str] = []
    for relative_path in _fastapi_entrypoint_candidates(root, source_roots):
        loaded = PythonFile.load(root, relative_path)
        if isinstance(loaded, BlockedPatch) and (root / relative_path).is_file():
            return loaded
        if isinstance(loaded, PythonFile) and _fastapi_app_assignment(loaded.tree):
            matches.append(relative_path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return BlockedPatch(
            "application entrypoint",
            "Multiple FastAPI app entrypoints were found; mount manually.",
        )
    return BlockedPatch(
        "application entrypoint",
        "Could not find a single top-level FastAPI app entrypoint.",
    )


def _fastapi_entrypoint_candidates(
    root: Path, source_roots: tuple[str, ...]
) -> list[str]:
    candidates: list[str] = []
    for relative_path in ("app/main.py", "main.py"):
        if (root / relative_path).is_file():
            candidates.append(relative_path)

    for source_root in source_roots:
        base = root / source_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if _ignored_path(root, path):
                continue
            if not _looks_like_fastapi_entrypoint_source(path):
                continue
            try:
                candidates.append(path.relative_to(root).as_posix())
            except ValueError:
                continue
    return list(dict.fromkeys(candidates))


def _ignored_path(root: Path, path: Path) -> bool:
    ignored_parts = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        "tests",
    }
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in ignored_parts for part in parts)


def _looks_like_fastapi_entrypoint_source(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return "FastAPI" in text and "app" in text


def _module_path_for_entrypoint(root: Path, relative_path: str) -> str | BlockedPatch:
    entrypoint = Path(relative_path)
    try:
        source_roots = project_python_source_roots(root)
    except ValueError as exc:
        return BlockedPatch("pyproject.toml", str(exc))
    for source_root in source_roots:
        source = Path(source_root)
        try:
            module_path = entrypoint.relative_to(source)
        except ValueError:
            continue
        return module_path.with_suffix("").as_posix().replace("/", ".")
    return entrypoint.with_suffix("").as_posix().replace("/", ".")


def _source_path_for_module(root: Path, module_path: str) -> str | BlockedPatch:
    if not module_path or any(
        not part.isidentifier() for part in module_path.split(".")
    ):
        return BlockedPatch(
            "config/fervis.json",
            "FastAPI app factory module must be a dotted Python module path.",
        )
    relative = Path(module_path.replace(".", "/"))
    candidates: list[str] = []
    for base in project_python_import_paths(root):
        for path in (
            base / relative.with_suffix(".py"),
            base / relative / "__init__.py",
        ):
            if path.is_file():
                candidates.append(path.relative_to(root).as_posix())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return BlockedPatch(
            "config/fervis.json",
            f"{module_path} resolves to multiple Python source files.",
        )
    return BlockedPatch(
        "config/fervis.json",
        f"{module_path} could not be resolved to a Python source file.",
    )


def fastapi_entrypoint_contains_hooks(tree: ast.Module) -> bool:
    app_assignment = _fastapi_app_assignment(tree)
    if app_assignment is None:
        return False
    mount_call = _fastapi_mount_shape_valid(tree, app_assignment)
    return (
        mount_call is not None
        and import_before(tree, "fervis", "configured_fervis", mount_call.lineno)
        and not top_level_name_shadowed_before_line(
            tree,
            "configured_fervis",
            mount_call.lineno,
            allowed_import=("fervis", "configured_fervis"),
        )
        and not top_level_assignment_after_line(tree, "app", app_assignment.lineno)
    )


def _fastapi_factory_contains_hooks(tree: ast.Module, factory_name: str) -> bool:
    factory = _fastapi_factory_function(tree, factory_name)
    if factory is None:
        return False
    app_assignment = _fastapi_app_assignment_in_statements(tree, factory.body)
    returned_app = _single_returned_app(factory)
    if app_assignment is None or returned_app is None:
        return False
    mount_call = _factory_mount_shape_valid(
        tree,
        factory,
        app_assignment,
        returned_app,
    )
    return mount_call is not None and import_before(
        tree,
        "fervis",
        "configured_fervis",
        mount_call.lineno,
    )


def _plan_fastapi_entrypoint(
    root: Path,
    relative_path: str,
) -> PlannedFilePatch | BlockedPatch:
    loaded = PythonFile.load(root, relative_path)
    if isinstance(loaded, BlockedPatch):
        return loaded
    app_assignment = _fastapi_app_assignment(loaded.tree)
    if app_assignment is None:
        return BlockedPatch(
            relative_path,
            "Could not find `app = FastAPI(...)` as a top-level assignment.",
        )
    import_problem = _fervis_import_problem(loaded.tree, path=relative_path)
    if import_problem is not None:
        return import_problem
    if top_level_assignment_after_line(loaded.tree, "app", app_assignment.lineno):
        return BlockedPatch(
            relative_path,
            "`app` is reassigned after `app = FastAPI(...)`; mount manually.",
        )
    if _contains_any_fervis_mount_call(loaded.tree) and not _fastapi_mount_shape_valid(
        loaded.tree,
        app_assignment,
    ):
        return BlockedPatch(
            relative_path,
            "Existing Fervis mount must be exactly one top-level "
            "`configured_fervis().mount(app)` after `app = FastAPI(...)`.",
        )
    if top_level_name_shadowed_before_line(
        loaded.tree,
        "configured_fervis",
        _fastapi_hook_line(loaded.tree, app_assignment),
        allowed_import=("fervis", "configured_fervis"),
    ):
        return BlockedPatch(
            relative_path,
            "Top-level `configured_fervis` is rebound before the Fervis mount.",
        )
    updated = loaded.text
    if not _contains_fervis_mount_app_call(loaded.tree):
        updated = _insert_entrypoint_mount(updated, loaded.tree)
    imported = _ensure_config_fervis_import(updated, path=relative_path)
    if isinstance(imported, BlockedPatch):
        return imported
    updated = imported
    return loaded.plan_validated(updated, validate=fastapi_entrypoint_contains_hooks)


def _plan_fastapi_factory(
    root: Path,
    relative_path: str,
    factory_name: str,
) -> PlannedFilePatch | BlockedPatch:
    loaded = PythonFile.load(root, relative_path)
    if isinstance(loaded, BlockedPatch):
        return loaded
    factory = _fastapi_factory_function(loaded.tree, factory_name)
    if factory is None:
        return BlockedPatch(
            relative_path,
            f"Could not find FastAPI app factory {factory_name!r}.",
        )
    app_assignment = _fastapi_app_assignment_in_statements(
        loaded.tree,
        factory.body,
    )
    if app_assignment is None:
        return BlockedPatch(
            relative_path,
            f"{factory_name} must assign exactly one local `app = FastAPI(...)`.",
        )
    return_app = _single_returned_app(factory)
    if return_app is None:
        return BlockedPatch(
            relative_path,
            f"{factory_name} must return the local FastAPI app exactly once.",
        )
    import_problem = _fervis_import_problem(loaded.tree, path=relative_path)
    if import_problem is not None:
        return import_problem
    if _contains_any_nested_fervis_mount_call(
        factory
    ) and not _factory_mount_shape_valid(
        loaded.tree,
        factory,
        app_assignment,
        return_app,
    ):
        return BlockedPatch(
            relative_path,
            "Existing factory Fervis mount must be `configured_fervis().mount(app)` "
            "before `return app`.",
        )

    updated = loaded.text
    if not _factory_mount_shape_valid(
        loaded.tree,
        factory,
        app_assignment,
        return_app,
    ):
        indent = " " * return_app.col_offset
        updated = insert_before_node(updated, return_app, f"{indent}{FASTAPI_MOUNT}")
    imported = _ensure_config_fervis_import(updated, path=relative_path)
    if isinstance(imported, BlockedPatch):
        return imported
    updated = imported
    return loaded.plan_validated(
        updated,
        validate=lambda tree: _fastapi_factory_contains_hooks(tree, factory_name),
    )


def _fastapi_app_assignment(tree: ast.Module) -> AssignedNode | None:
    return _fastapi_app_assignment_in_statements(tree, tree.body)


def _fastapi_app_assignment_in_statements(
    tree: ast.Module,
    statements: list[ast.stmt],
) -> AssignedNode | None:
    matches: list[AssignedNode] = []
    for node in statements:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        if not directly_assigns_name(node, "app"):
            continue
        if node.value is not None and _is_fastapi_call(
            tree, node.value, line=node.lineno
        ):
            matches.append(node)
    if len(matches) != 1:
        return None
    return matches[0]


def _fastapi_factory_function(
    tree: ast.Module,
    factory_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == factory_name
        ):
            return node
    return None


def _single_returned_app(
    factory: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Return | None:
    returns = [
        node
        for node in factory.body
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "app"
    ]
    return returns[0] if len(returns) == 1 else None


def _is_fastapi_call(tree: ast.Module, node: ast.AST, *, line: int) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return (
            node.func.id == "FastAPI"
            and import_before(tree, "fastapi", "FastAPI", line)
            and not top_level_name_shadowed_before_line(
                tree,
                "FastAPI",
                line,
                allowed_import=("fastapi", "FastAPI"),
            )
        )
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "FastAPI"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "fastapi"
        and module_import_before(tree, "fastapi", "fastapi", line)
        and not top_level_name_shadowed_before_line(
            tree,
            "fastapi",
            line,
            allowed_module_import=("fastapi", "fastapi"),
        )
    )


def _fervis_import_problem(
    tree: ast.Module,
    *,
    path: str,
) -> BlockedPatch | None:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != "fervis":
            continue
        imported_names = {alias.name for alias in node.names}
        if "configured_fervis" not in imported_names:
            continue
        if node.level or any(
            alias.name == "*" or alias.asname is not None for alias in node.names
        ):
            return BlockedPatch(
                path,
                "Aliased or wildcard imports of configured_fervis must be edited manually.",
            )
    return None


def _ensure_config_fervis_import(text: str, *, path: str) -> str | BlockedPatch:
    tree = parse_python_source(text)
    import_problem = _fervis_import_problem(tree, path=path)
    if import_problem is not None:
        return import_problem
    if has_import(tree, "fervis", "configured_fervis"):
        return text
    return insert_import_line(text, FASTAPI_IMPORT)


def _contains_fervis_mount_app_call(tree: ast.Module) -> bool:
    return _fervis_mount_app_call(tree) is not None


def _fervis_mount_app_call(tree: ast.Module) -> ast.Expr | None:
    for node in tree.body:
        call = _as_fervis_mount_app_expr(node)
        if call is not None:
            return call
    return None


def _fastapi_hook_line(
    tree: ast.Module,
    app_assignment: AssignedNode,
) -> int:
    mount_call = _fervis_mount_after_app_assignment(tree, app_assignment)
    if mount_call is not None:
        return mount_call.lineno
    return app_assignment.lineno


def _fervis_mount_after_app_assignment(
    tree: ast.Module,
    app_assignment: AssignedNode,
) -> ast.Expr | None:
    app_line = app_assignment.end_lineno or app_assignment.lineno
    for node in tree.body:
        call = _as_fervis_mount_app_expr(node)
        if node.lineno > app_line and call is not None:
            return call
    return None


def _fastapi_mount_shape_valid(
    tree: ast.Module,
    app_assignment: AssignedNode,
) -> ast.Expr | None:
    module_flow_calls = _fervis_mount_calls_in_module_flow(tree)
    top_level_calls = _fervis_mount_calls(tree)
    if len(module_flow_calls) != 1 or len(top_level_calls) != 1:
        return None
    call = top_level_calls[0]
    if module_flow_calls[0] is not call:
        return None
    if not _is_fervis_mount_app_expr(call):
        return None
    app_line = app_assignment.end_lineno or app_assignment.lineno
    if call.lineno <= app_line:
        return None
    if not _entrypoint_mount_is_final(tree, call):
        return None
    return call


def _insert_entrypoint_mount(text: str, tree: ast.Module) -> str:
    main_guard = _main_guard(tree)
    if main_guard is not None:
        return insert_before_node(text, main_guard, FASTAPI_MOUNT)
    newline = newline_for(text)
    source = rstrip_line_endings(text)
    return f"{source}{newline}{newline}{FASTAPI_MOUNT}{newline}"


def _entrypoint_mount_is_final(tree: ast.Module, call: ast.Expr) -> bool:
    main_guard = _main_guard(tree)
    if main_guard is not None:
        guard_index = tree.body.index(main_guard)
        return guard_index > 0 and tree.body[guard_index - 1] is call
    return bool(tree.body) and tree.body[-1] is call


def _main_guard(tree: ast.Module) -> ast.If | None:
    guards = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.If) and _is_main_guard_test(node.test)
    )
    return guards[0] if len(guards) == 1 else None


def _is_main_guard_test(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _fervis_mount_calls(tree: ast.Module) -> list[ast.Expr]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "mount"
        and is_configured_fervis_call(node.value.func.value)
    ]


def _is_fervis_mount_app_expr(node: ast.AST) -> bool:
    return _as_fervis_mount_app_expr(node) is not None


def _as_fervis_mount_app_expr(node: ast.AST) -> ast.Expr | None:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    call = node.value
    func = call.func
    valid = (
        isinstance(func, ast.Attribute)
        and func.attr == "mount"
        and is_configured_fervis_call(func.value)
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "app"
    )
    return node if valid else None


def _contains_any_nested_fervis_mount_call(
    factory: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "mount"
        and is_configured_fervis_call(node.value.func.value)
        for node in ast.walk(factory)
    )


def _factory_mount_shape_valid(
    tree: ast.Module,
    factory: ast.FunctionDef | ast.AsyncFunctionDef,
    app_assignment: AssignedNode,
    returned_app: ast.Return,
) -> ast.Expr | None:
    del tree
    calls = tuple(
        call
        for node in factory.body
        if (call := _as_fervis_mount_app_expr(node)) is not None
    )
    if len(calls) != 1:
        return None
    call = calls[0]
    app_line = app_assignment.end_lineno or app_assignment.lineno
    return_line = returned_app.lineno
    if not (app_line < call.lineno < return_line):
        return None
    return call


def _contains_any_fervis_mount_call(tree: ast.Module) -> bool:
    return bool(_fervis_mount_calls_in_module_flow(tree))


def _fervis_mount_calls_in_module_flow(tree: ast.Module) -> list[ast.Expr]:
    calls: list[ast.Expr] = []
    for node in tree.body:
        calls.extend(_fervis_mount_calls_in_statement_flow(node))
    return calls


def _fervis_mount_calls_in_statement_flow(node: ast.stmt) -> list[ast.Expr]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return []
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "mount"
        and is_configured_fervis_call(node.value.func.value)
    ):
        return [node]
    calls: list[ast.Expr] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            calls.extend(_fervis_mount_calls_in_statement_flow(child))
    return calls
