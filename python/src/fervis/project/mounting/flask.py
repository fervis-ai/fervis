"""Flask framework mounting."""

from __future__ import annotations

import ast
from pathlib import Path

from ..discovery import ProjectInspection
from ..importing import project_python_import_paths
from ..integration import FervisConfig, FlaskAppSource
from ..source_paths import normalize_source_path_prefixes
from .bindings import (
    AssignedNode,
    directly_assigns_name,
    has_import,
    import_before,
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
from .source import insert_after_node, insert_before_node, insert_import_line


FLASK_IMPORT = "from fervis.flask import configured_fervis"
FLASK_MOUNT = "configured_fervis().init_app(app)"


def patch_flask(
    project: ProjectInspection, *, app_target: str | None
) -> FrameworkPatchResult:
    if not app_target:
        return FrameworkPatchResult(
            blocked=[
                ("config/fervis.json", "Flask app target must be provided with --app.")
            ]
        )
    target = flask_app_target(project.root_path, app_target)
    if isinstance(target, BlockedPatch):
        return FrameworkPatchResult(blocked=[(target.path, target.reason)])
    relative_path, object_name = target
    if object_name == "app":
        plan = _plan_flask_app_object(project.root_path, relative_path)
    else:
        plan = _plan_flask_app_factory(
            project.root_path,
            relative_path,
            factory_name=object_name,
        )
    return plan_then_apply([plan])


def flask_checks(
    project: ProjectInspection, config: FervisConfig
) -> list[FrameworkCheck]:
    mount_valid = _configured_source_mount_valid(project, config)
    return [
        FrameworkCheck(
            id="framework.flask.mount",
            passed=mount_valid,
            message=(
                "Configured Flask source imports Fervis and calls "
                "configured_fervis().init_app(app)."
            ),
            fix=init_fix(project.framework),
        )
    ]


def flask_source_schema(
    app_target: str | None,
    *,
    path_prefixes: tuple[str, ...] | None,
    blueprints: tuple[str, ...] = (),
) -> dict[str, object] | BlockedPatch:
    if not app_target:
        return BlockedPatch(
            "config/fervis.json",
            "Flask app target must be provided with --app.",
        )
    if ":" not in app_target:
        return BlockedPatch(
            "config/fervis.json",
            "Flask app target must use module:object import path syntax.",
        )
    if not path_prefixes:
        return BlockedPatch(
            "config/fervis.json",
            "At least one Flask source prefix must be provided with --source-prefix.",
        )
    try:
        source_prefixes = normalize_source_path_prefixes(path_prefixes)
    except ValueError as exc:
        return BlockedPatch(
            "config/fervis.json",
            f"Flask source path prefixes are invalid: {exc}",
        )
    return {
        "kind": "flask_app",
        "name": "default",
        "app": app_target,
        "app_args": [],
        "app_kwargs": {},
        "path_prefixes": list(source_prefixes),
        "blueprints": list(blueprints),
    }


def flask_app_target(root: Path, app_target: str) -> tuple[str, str] | BlockedPatch:
    if ":" not in app_target:
        return BlockedPatch(
            "config/fervis.json",
            "Flask app target must use module:object import path syntax.",
        )
    module_path, object_name = app_target.split(":", 1)
    if "." in object_name or not object_name:
        return BlockedPatch(
            "config/fervis.json",
            "Flask app target must reference one top-level object.",
        )
    relative_path = _source_path_for_module(root, module_path)
    if isinstance(relative_path, BlockedPatch):
        return relative_path
    return relative_path, object_name


def _configured_source_mount_valid(
    project: ProjectInspection, config: FervisConfig
) -> bool:
    for source in config.sources:
        if not isinstance(source, FlaskAppSource):
            continue
        if _configured_import_mount_valid(project.root_path, source.app):
            return True
    return False


def _configured_import_mount_valid(root: Path, import_path: str) -> bool:
    try:
        from fervis.host_api.adapters.flask.loading import import_flask_app

        app = import_flask_app(import_path, project_root=root)
    except Exception:
        return False
    return _runtime_app_has_fervis_routes(app)


def _runtime_app_has_fervis_routes(app: object) -> bool:
    url_map = getattr(app, "url_map", None)
    iter_rules = getattr(url_map, "iter_rules", None)
    if not callable(iter_rules):
        return False
    return any(
        str(getattr(rule, "endpoint", "")).startswith("fervis.")
        for rule in iter_rules()
    )


def flask_entrypoint_contains_hooks(tree: ast.Module) -> bool:
    app_assignment = _top_level_app_assignment(tree)
    if app_assignment is None:
        return False
    mount_call = _flask_mount_after_app_assignment(tree, app_assignment)
    return (
        mount_call is not None
        and import_before(tree, "fervis.flask", "configured_fervis", mount_call.lineno)
        and not top_level_name_shadowed_before_line(
            tree,
            "configured_fervis",
            mount_call.lineno,
            allowed_import=("fervis.flask", "configured_fervis"),
        )
        and not top_level_assignment_after_line(tree, "app", app_assignment.lineno)
    )


def _plan_flask_app_object(
    root: Path,
    relative_path: str,
) -> PlannedFilePatch | BlockedPatch:
    loaded = PythonFile.load(root, relative_path)
    if isinstance(loaded, BlockedPatch):
        return loaded
    app_assignment = _top_level_app_assignment(loaded.tree)
    if app_assignment is None:
        return BlockedPatch(
            relative_path,
            "Could not find `app = ...` as a single top-level assignment.",
        )
    import_problem = _fervis_import_problem(loaded.tree, path=relative_path)
    if import_problem is not None:
        return import_problem
    if top_level_name_shadowed_before_line(
        loaded.tree,
        "configured_fervis",
        app_assignment.lineno,
        allowed_import=("fervis.flask", "configured_fervis"),
    ):
        return BlockedPatch(
            relative_path,
            "Top-level `configured_fervis` is rebound before the Fervis mount.",
        )
    if top_level_assignment_after_line(loaded.tree, "app", app_assignment.lineno):
        return BlockedPatch(
            relative_path,
            "`app` is reassigned after the configured app assignment; mount manually.",
        )

    updated = loaded.text
    if _flask_mount_after_app_assignment(loaded.tree, app_assignment) is None:
        updated = insert_after_node(updated, app_assignment, FLASK_MOUNT)
    import_update = _ensure_config_fervis_import(updated, path=relative_path)
    if isinstance(import_update, BlockedPatch):
        return import_update
    return loaded.plan_validated(
        import_update,
        validate=flask_entrypoint_contains_hooks,
    )


def _plan_flask_app_factory(
    root: Path,
    relative_path: str,
    *,
    factory_name: str,
) -> PlannedFilePatch | BlockedPatch:
    loaded = PythonFile.load(root, relative_path)
    if isinstance(loaded, BlockedPatch):
        return loaded
    factory = _flask_app_factory(loaded.tree, factory_name=factory_name)
    if factory is None:
        return BlockedPatch(
            relative_path,
            (
                f"Could not prove `{factory_name}` has one `app = Flask(...)` "
                "and one `return app`; mount manually."
            ),
        )
    import_problem = _fervis_import_problem(loaded.tree, path=relative_path)
    if import_problem is not None:
        return import_problem
    updated = loaded.text
    if _flask_mount_before_return(factory) is None:
        updated = insert_before_node(updated, factory.return_node, f"    {FLASK_MOUNT}")
    import_update = _ensure_config_fervis_import(updated, path=relative_path)
    if isinstance(import_update, BlockedPatch):
        return import_update
    return loaded.plan_validated(
        import_update,
        validate=lambda tree: flask_factory_contains_hooks(
            tree,
            factory_name=factory_name,
        ),
    )


def _top_level_app_assignment(tree: ast.Module) -> AssignedNode | None:
    matches: list[AssignedNode] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        if not directly_assigns_name(node, "app"):
            continue
        matches.append(node)
    if len(matches) != 1:
        return None
    return matches[0]


class _FlaskFactory:
    def __init__(
        self,
        *,
        assignment: AssignedNode,
        return_node: ast.Return,
        function: ast.FunctionDef,
    ) -> None:
        self.assignment = assignment
        self.return_node = return_node
        self.function = function


def _flask_app_factory(
    tree: ast.Module,
    *,
    factory_name: str,
) -> _FlaskFactory | None:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == factory_name
    ]
    if len(matches) != 1:
        return None
    function = matches[0]
    assignments = [
        node
        for node in function.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and directly_assigns_name(node, "app")
        and node.value is not None
        and _is_flask_call(tree, node.value, line=node.lineno)
    ]
    all_returns = _function_owned_returns(function)
    returns = [
        node
        for node in all_returns
        if isinstance(node.value, ast.Name) and node.value.id == "app"
    ]
    if len(assignments) != 1 or len(all_returns) != 1 or len(returns) != 1:
        return None
    assignment = assignments[0]
    return_node = returns[0]
    if assignment.lineno >= return_node.lineno:
        return None
    return _FlaskFactory(
        assignment=assignment,
        return_node=return_node,
        function=function,
    )


def _function_owned_returns(function: ast.FunctionDef) -> list[ast.Return]:
    returns: list[ast.Return] = []

    class ReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, node: ast.Return) -> None:
            returns.append(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    visitor = ReturnVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return returns


def flask_factory_contains_hooks(
    tree: ast.Module,
    *,
    factory_name: str,
) -> bool:
    factory = _flask_app_factory(tree, factory_name=factory_name)
    if factory is None:
        return False
    mount_call = _flask_mount_before_return(factory)
    return mount_call is not None and import_before(
        tree, "fervis.flask", "configured_fervis", mount_call.lineno
    )


def _flask_mount_before_return(factory: _FlaskFactory) -> ast.Expr | None:
    for node in factory.function.body:
        if (
            node.lineno > factory.assignment.lineno
            and node.lineno < factory.return_node.lineno
            and isinstance(node, ast.Expr)
            and _is_flask_mount_app_expr(node)
        ):
            return node
    return None


def _is_flask_call(tree: ast.Module, node: ast.AST, *, line: int) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return (
            node.func.id == "Flask"
            and import_before(tree, "flask", "Flask", line)
            and not top_level_name_shadowed_before_line(
                tree,
                "Flask",
                line,
                allowed_import=("flask", "Flask"),
            )
        )
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "Flask"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "flask"
        and module_import_before(tree, "flask", "flask", line)
        and not top_level_name_shadowed_before_line(
            tree,
            "flask",
            line,
            allowed_module_import=("flask", "flask"),
        )
    )


def _flask_mount_after_app_assignment(
    tree: ast.Module,
    app_assignment: AssignedNode,
) -> ast.Expr | None:
    app_line = app_assignment.end_lineno or app_assignment.lineno
    for node in tree.body:
        if (
            node.lineno > app_line
            and isinstance(node, ast.Expr)
            and _is_flask_mount_app_expr(node)
        ):
            return node
    return None


def _is_flask_mount_app_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "init_app"
        and _is_configured_fervis_result(node.value.func.value)
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "app"
        and not node.value.keywords
    )


def _is_configured_fervis_result(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "configured_fervis"
        and not node.args
        and not node.keywords
    )


def _fervis_import_problem(tree: ast.Module, *, path: str) -> BlockedPatch | None:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != "fervis.flask":
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
    if has_import(tree, "fervis.flask", "configured_fervis"):
        return text
    return insert_import_line(text, FLASK_IMPORT)


def _source_path_for_module(root: Path, module_path: str) -> str | BlockedPatch:
    if not module_path or any(
        not part.isidentifier() for part in module_path.split(".")
    ):
        return BlockedPatch(
            "config/fervis.json",
            "Flask app target module must be a dotted Python module path.",
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
