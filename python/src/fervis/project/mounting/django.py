"""Django + DRF framework mounting."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..discovery import ProjectInspection
from ..integration import FervisConfig
from .bindings import (
    AssignedNode,
    directly_assigns_name,
    has_import,
    import_before,
    is_configured_fervis_call,
    module_binding_modified_after_line,
    top_level_assignment,
    top_level_assignments,
    top_level_bound_names,
    statement_mutates_module_namespace,
    target_names,
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
    single_top_level_assignment,
)
from .source import (
    append_literal_sequence_items,
    ensure_multiline_sequence_trailing_comma,
    insert_import_line,
    newline_for,
    replace_node_source,
    rstrip_line_endings,
)


DJANGO_APP = "fervis.django"
FERVIS_DJANGO_RUNTIME_APPS = (
    "fervis.lineage",
    "fervis.run_work.queue.django",
    "fervis.interfaces.django",
    DJANGO_APP,
)
DRF_APP = "rest_framework"
REQUIRED_DJANGO_APPS = (DRF_APP, *FERVIS_DJANGO_RUNTIME_APPS)
FERVIS_CONFIG_PATH_VALUE = "config/fervis.json"
FERVIS_CONFIG_PATH_SETTING = f'FERVIS_CONFIG_PATH = "{FERVIS_CONFIG_PATH_VALUE}"'
DJANGO_URL_IMPORT = "from fervis import configured_fervis"
DJANGO_URL_MOUNT = (
    "path(configured_fervis().routes.django_path, include(configured_fervis().urls)),"
)


@dataclass(frozen=True)
class UrlpatternMountTarget:
    list_assignment: AssignedNode
    wrapper_call: ast.Call | None = None


@dataclass(frozen=True)
class ProtectedListPolicy:
    name: str
    allows_method_call: Callable[[ast.Call], bool]
    allows_reassignment: Callable[[list[ast.stmt], int, ast.stmt], bool] | None = None

    def has_unsafe_change_after(self, tree: ast.Module, line: int) -> bool:
        statements = [node for node in tree.body if node.lineno > line]
        return self.has_unsafe_change_in(statements)

    def has_unsafe_change_in(self, statements: list[ast.stmt]) -> bool:
        for index, node in enumerate(statements):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if statement_mutates_module_namespace(node, self.name):
                return True
            if _statement_aliases_name(node, self.name):
                return True
            if self._allows_reassignment(statements, index, node):
                continue
            if _statement_directly_rebinds_name(node, self.name):
                return True
            if any(
                not self.allows_method_call(call)
                for call in _method_calls_on_name(node, self.name)
            ):
                return True
            if any(
                self.has_unsafe_change_in(child_statements)
                for child_statements in _child_statement_lists(node)
            ):
                return True
        return False

    def _allows_reassignment(
        self,
        statements: list[ast.stmt],
        index: int,
        node: ast.stmt,
    ) -> bool:
        return self.allows_reassignment is not None and self.allows_reassignment(
            statements, index, node
        )


def patch_django(project: ProjectInspection) -> FrameworkPatchResult:
    settings_path = django_settings_path(project.root_path)
    if isinstance(settings_path, BlockedPatch):
        return FrameworkPatchResult(
            blocked=[(settings_path.path, settings_path.reason)]
        )
    urlconf_path = django_root_urlconf_path(project.root_path, settings_path)
    if isinstance(urlconf_path, BlockedPatch):
        return FrameworkPatchResult(blocked=[(urlconf_path.path, urlconf_path.reason)])
    return plan_then_apply(
        [
            _plan_django_settings(project.root_path, settings_path),
            _plan_django_urls(project.root_path, urlconf_path),
        ]
    )


def django_checks(
    project: ProjectInspection,
    config: FervisConfig,
) -> list[FrameworkCheck]:
    del config
    settings_path = django_settings_path(project.root_path)
    settings = (
        PythonFile.load(project.root_path, settings_path)
        if isinstance(settings_path, str)
        else settings_path
    )
    urlconf_path = (
        django_root_urlconf_path(project.root_path, settings_path)
        if isinstance(settings_path, str)
        else settings_path
    )
    urls = (
        PythonFile.load(project.root_path, urlconf_path)
        if isinstance(urlconf_path, str)
        else urlconf_path
    )
    return [
        FrameworkCheck(
            id="framework.django.installed_apps",
            passed=isinstance(settings, PythonFile)
            and _django_settings_contains_hooks(settings.tree),
            message=(
                "Django settings include DRF, Fervis Django runtime apps, and "
                "FERVIS_CONFIG_PATH."
            ),
            fix=init_fix(project.framework),
        ),
        FrameworkCheck(
            id="framework.django.urls",
            passed=isinstance(urls, PythonFile)
            and django_urls_contains_hooks(urls.tree),
            message="Django root URLconf mounts Fervis URLs through configured_fervis().",
            fix=init_fix(project.framework),
        ),
    ]


def django_settings_path(root: Path) -> str | BlockedPatch:
    manage_py = PythonFile.load(root, "manage.py")
    if isinstance(manage_py, BlockedPatch):
        return manage_py
    module_paths = django_settings_modules(manage_py.tree)
    if len(module_paths) != 1:
        return BlockedPatch(
            "manage.py",
            "manage.py must declare exactly one literal DJANGO_SETTINGS_MODULE so Fervis can patch the active settings file.",
        )
    module_path = module_paths[0]
    if not _safe_module_path(module_path):
        return BlockedPatch(
            "manage.py",
            "DJANGO_SETTINGS_MODULE must be a dotted Python module path.",
        )
    return _module_source_path(root, module_path, owner_path="manage.py")


def _module_source_path(
    root: Path,
    module_path: str,
    *,
    owner_path: str,
) -> str | BlockedPatch:
    relative = Path(module_path.replace(".", "/"))
    module_file = relative.with_suffix(".py")
    package_file = relative / "__init__.py"
    candidates = [
        path.as_posix()
        for path in (module_file, package_file)
        if (root / path).is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return BlockedPatch(
            owner_path,
            f"{module_path} resolves to both a module and package; mount manually.",
        )
    return BlockedPatch(
        owner_path,
        f"{module_path} could not be resolved to a Python source file.",
    )


def django_settings_modules(tree: ast.Module) -> list[str]:
    module_paths: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_environ_setdefault_call(node):
            continue
        if len(node.args) < 2:
            continue
        key, value = node.args[0], node.args[1]
        if (
            isinstance(key, ast.Constant)
            and key.value == "DJANGO_SETTINGS_MODULE"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            module_paths.append(value.value)
    return module_paths


def _is_environ_setdefault_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "setdefault"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "environ"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "os"
    )


def django_root_urlconf_path(root: Path, settings_path: str) -> str | BlockedPatch:
    settings = PythonFile.load(root, settings_path)
    if isinstance(settings, BlockedPatch):
        return settings
    assignment = single_top_level_assignment(
        settings.tree,
        "ROOT_URLCONF",
        path=settings_path,
    )
    if isinstance(assignment, BlockedPatch):
        return BlockedPatch(
            settings_path,
            "ROOT_URLCONF must be a literal string so Fervis can patch the active URLconf.",
        )
    if not isinstance(assignment.value, ast.Constant) or not isinstance(
        assignment.value.value,
        str,
    ):
        return BlockedPatch(
            settings_path,
            "ROOT_URLCONF must be a literal string so Fervis can patch the active URLconf.",
        )
    module_path = assignment.value.value
    if not _safe_module_path(module_path):
        return BlockedPatch(
            settings_path,
            "ROOT_URLCONF must be a dotted Python module path.",
        )
    if module_binding_modified_after_line(
        settings.tree,
        "ROOT_URLCONF",
        assignment.end_lineno or assignment.lineno,
    ):
        return BlockedPatch(
            settings_path,
            "ROOT_URLCONF is modified after assignment; mount manually.",
        )
    return module_path.replace(".", "/") + ".py"


def django_settings_contains_apps(tree: ast.Module) -> bool:
    assignment = _installed_apps_assignment(tree, path="settings.py")
    if isinstance(assignment, BlockedPatch):
        return False
    if assignment is None:
        return False
    if _installed_apps_policy().has_unsafe_change_after(
        tree, assignment.end_lineno or assignment.lineno
    ):
        return False
    installed_apps = _static_installed_app_names(assignment.value) or []
    return all(app in installed_apps for app in REQUIRED_DJANGO_APPS)


def django_urls_contains_hooks(tree: ast.Module) -> bool:
    mount = _fervis_urlpattern_mount(tree)
    if mount is None:
        return False
    fervis_call, wrapper_call = mount
    call_lines = [fervis_call.lineno]
    if wrapper_call is not None:
        call_lines.append(wrapper_call.lineno)
    if not all(_django_url_call_imports_are_valid(tree, line) for line in call_lines):
        return False
    return not any(
        _django_url_call_names_are_shadowed(tree, line) for line in call_lines
    )


def _django_url_call_imports_are_valid(tree: ast.Module, line: int) -> bool:
    return (
        _import_before_any(tree, [("fervis", "configured_fervis")], line)
        and _import_before_any(
            tree,
            [("django.urls", "include"), ("django.conf.urls", "include")],
            line,
        )
        and _import_before_any(tree, [("django.urls", "path")], line)
    )


def _django_url_call_names_are_shadowed(tree: ast.Module, line: int) -> bool:
    return any(
        _name_shadowed_before_line_except_imports(
            tree,
            name,
            line,
            allowed_imports=allowed_imports,
        )
        for name, allowed_imports in (
            ("configured_fervis", [("fervis", "configured_fervis")]),
            ("include", [("django.urls", "include"), ("django.conf.urls", "include")]),
            ("path", [("django.urls", "path")]),
        )
    )


def _plan_django_settings(
    root: Path, settings_path: str
) -> PlannedFilePatch | BlockedPatch:
    loaded = PythonFile.load(root, settings_path)
    if isinstance(loaded, BlockedPatch):
        return loaded
    assignment = _installed_apps_assignment(loaded.tree, path=settings_path)
    if isinstance(assignment, BlockedPatch):
        return assignment
    installed_apps = _static_installed_app_names(assignment.value)
    if installed_apps is None:
        return BlockedPatch(
            settings_path,
            "INSTALLED_APPS must be a literal list or tuple before Fervis can patch it.",
        )
    if _installed_apps_policy().has_unsafe_change_after(
        loaded.tree, assignment.end_lineno or assignment.lineno
    ):
        return BlockedPatch(
            settings_path,
            "INSTALLED_APPS is overwritten or may remove Fervis after assignment; mount manually.",
        )
    updated = append_literal_sequence_items(
        loaded.text,
        assignment.value,
        [app for app in REQUIRED_DJANGO_APPS if app not in installed_apps],
        path=settings_path,
    )
    if isinstance(updated, BlockedPatch):
        return updated
    updated = _ensure_fervis_config_setting(updated, path=settings_path)
    if isinstance(updated, BlockedPatch):
        return updated
    return loaded.plan_validated(updated, validate=_django_settings_contains_hooks)


def _plan_django_urls(
    root: Path,
    relative_path: str,
) -> PlannedFilePatch | BlockedPatch:
    loaded = PythonFile.load(root, relative_path)
    if isinstance(loaded, BlockedPatch):
        return loaded
    target = _urlpattern_mount_target(loaded.tree, path=relative_path)
    if isinstance(target, BlockedPatch):
        return target
    updated = _append_urlpattern(
        loaded.text,
        loaded.tree,
        target.list_assignment.value,
        loaded.relative_path,
    )
    if isinstance(updated, BlockedPatch):
        return updated
    updated = _ensure_django_urls_imports(updated, path=relative_path)
    if isinstance(updated, BlockedPatch):
        return updated
    return loaded.plan_validated(updated, validate=django_urls_contains_hooks)


def _django_settings_contains_hooks(tree: ast.Module) -> bool:
    return django_settings_contains_apps(tree) and _has_fervis_config_setting(tree)


def _has_fervis_config_setting(tree: ast.Module) -> bool:
    assignment = top_level_assignment(tree, "FERVIS_CONFIG_PATH")
    return (
        assignment is not None
        and isinstance(assignment.value, ast.Constant)
        and assignment.value.value == FERVIS_CONFIG_PATH_VALUE
        and not module_binding_modified_after_line(
            tree,
            "FERVIS_CONFIG_PATH",
            assignment.end_lineno or assignment.lineno,
        )
    )


def _append_urlpattern(
    text: str,
    tree: ast.Module,
    node: ast.List,
    relative_path: str,
) -> str | BlockedPatch:
    if _contains_urlpattern_call(tree):
        return text
    if ast.get_source_segment(text, node) is None:
        return BlockedPatch(relative_path, "Could not locate urlpatterns source.")
    if node.lineno == node.end_lineno:
        if node.elts:
            return BlockedPatch(
                relative_path,
                "Non-empty single-line urlpatterns must be edited manually.",
            )
        return replace_node_source(
            text,
            node,
            f"[\n    {DJANGO_URL_MOUNT}\n]",
        )
    newline = newline_for(text)
    text = ensure_multiline_sequence_trailing_comma(text, node, path=relative_path)
    if isinstance(text, BlockedPatch):
        return text
    lines = text.splitlines(keepends=True)
    closing_index = (node.end_lineno or node.lineno) - 1
    closing_line = lines[closing_index]
    closing_indent = closing_line[: len(closing_line) - len(closing_line.lstrip())]
    return "".join(
        [
            *lines[:closing_index],
            f"{closing_indent}    {DJANGO_URL_MOUNT}{newline}",
            *lines[closing_index:],
        ]
    )


def _ensure_fervis_config_setting(text: str, *, path: str) -> str | BlockedPatch:
    tree = parse_python_source(text)
    assignments = top_level_assignments(tree, "FERVIS_CONFIG_PATH")
    if not assignments:
        newline = newline_for(text)
        return (
            rstrip_line_endings(text) + newline + FERVIS_CONFIG_PATH_SETTING + newline
        )
    if len(assignments) > 1:
        return BlockedPatch(
            path,
            "FERVIS_CONFIG_PATH is assigned more than once.",
        )
    assignment = assignments[0]
    if (
        isinstance(assignment.value, ast.Constant)
        and assignment.value.value == FERVIS_CONFIG_PATH_VALUE
    ):
        return text
    return BlockedPatch(
        path,
        "FERVIS_CONFIG_PATH already exists with a different value.",
    )


def _ensure_django_urls_imports(text: str, *, path: str) -> str | BlockedPatch:
    tree = parse_python_source(text)
    updated = _ensure_django_url_include_import(text, tree, path=path)
    if isinstance(updated, BlockedPatch):
        return updated
    tree = parse_python_source(updated)
    updated = _ensure_django_url_import_name(updated, tree, "path", path=path)
    if isinstance(updated, BlockedPatch):
        return updated
    tree = parse_python_source(updated)
    if has_import(tree, "fervis", "configured_fervis"):
        return updated
    return insert_import_line(updated, DJANGO_URL_IMPORT)


def _ensure_django_url_import_name(
    text: str,
    tree: ast.Module,
    name: str,
    *,
    path: str,
) -> str | BlockedPatch:
    imports = _valid_django_url_imports(tree, path=path)
    if isinstance(imports, BlockedPatch):
        return imports
    if not imports:
        return BlockedPatch(
            path,
            "Could not find a top-level `from django.urls import ...` statement.",
        )
    if _name_imported_from_unexpected_module(tree, name, {"django.urls"}):
        return BlockedPatch(
            path,
            f"`{name}` is already imported from another module; mount manually.",
        )
    containing_import = next(
        (node for node in imports if any(alias.name == name for alias in node.names)),
        None,
    )
    if containing_import is not None:
        return text
    node = imports[0]
    names = [alias.name for alias in node.names]
    return replace_node_source(
        text,
        node,
        "from django.urls import " + ", ".join([*names, name]),
    )


def _valid_django_url_imports(
    tree: ast.Module,
    *,
    path: str,
) -> list[ast.ImportFrom] | BlockedPatch:
    imports: list[ast.ImportFrom] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "django.urls":
            continue
        if node.level or any(
            alias.asname is not None or alias.name == "*" for alias in node.names
        ):
            return BlockedPatch(
                path,
                "Aliased, relative, or wildcard django.urls imports must be edited manually.",
            )
        imports.append(node)
    return imports


def _ensure_django_url_include_import(
    text: str,
    tree: ast.Module,
    *,
    path: str,
) -> str | BlockedPatch:
    allowed_modules = {"django.urls", "django.conf.urls"}
    if _name_imported_from_unexpected_module(tree, "include", allowed_modules):
        return BlockedPatch(
            path,
            "`include` is already imported from another module; mount manually.",
        )
    if any(has_import(tree, module, "include") for module in allowed_modules):
        return text
    return _ensure_django_url_import_name(text, tree, "include", path=path)


def _name_imported_from_unexpected_module(
    tree: ast.Module,
    name: str,
    allowed_modules: set[str],
) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            bound_name = alias.asname or alias.name
            if bound_name != name:
                continue
            if node.level or node.module not in allowed_modules:
                return True
            if alias.asname is not None or alias.name != name:
                return True
    return False


def _contains_urlpattern_call(tree: ast.Module) -> bool:
    return _fervis_urlpattern_mount(tree) is not None


def _fervis_urlpattern_mount(
    tree: ast.Module,
) -> tuple[ast.Call, ast.Call | None] | None:
    target = _urlpattern_mount_target(tree, path="urls.py")
    if isinstance(target, BlockedPatch):
        return None
    for item in target.list_assignment.value.elts:
        if _is_fervis_path_call(item):
            return item, target.wrapper_call
    return None


def _urlpattern_mount_target(
    tree: ast.Module,
    *,
    path: str,
) -> UrlpatternMountTarget | BlockedPatch:
    assignments = top_level_assignments(tree, "urlpatterns")
    if not assignments:
        return BlockedPatch(path, "Could not find a top-level urlpatterns assignment.")
    if len(assignments) > 1:
        return BlockedPatch(path, "urlpatterns is assigned more than once.")

    assignment = assignments[0]
    if not isinstance(assignment.value, ast.List):
        return BlockedPatch(
            path,
            "urlpatterns must be a literal list before Fervis can patch it.",
        )

    wrapped = _single_wrapped_urlpattern_list(tree, assignment.value)
    if wrapped is None:
        if _urlpattern_list_policy("urlpatterns").has_unsafe_change_after(
            tree,
            assignment.end_lineno or assignment.lineno,
        ):
            return BlockedPatch(
                path,
                "urlpatterns is overwritten or may remove Fervis after assignment; mount manually.",
            )
        return UrlpatternMountTarget(list_assignment=assignment)

    wrapped_name, wrapper_call = wrapped
    wrapped_assignments = top_level_assignments(tree, wrapped_name)
    if len(wrapped_assignments) != 1 or not isinstance(
        wrapped_assignments[0].value,
        ast.List,
    ):
        return BlockedPatch(
            path,
            "urlpatterns wraps a named URL list that Fervis cannot patch safely.",
        )
    wrapped_assignment = wrapped_assignments[0]
    if _urlpattern_list_policy(wrapped_name).has_unsafe_change_after(
        tree,
        wrapped_assignment.end_lineno or wrapped_assignment.lineno,
    ):
        return BlockedPatch(
            path,
            f"{wrapped_name} is overwritten or may remove Fervis after assignment; mount manually.",
        )
    if _urlpattern_list_policy("urlpatterns").has_unsafe_change_after(
        tree,
        assignment.end_lineno or assignment.lineno,
    ):
        return BlockedPatch(
            path,
            "urlpatterns is overwritten or may remove Fervis after assignment; mount manually.",
        )
    return UrlpatternMountTarget(
        list_assignment=wrapped_assignment,
        wrapper_call=wrapper_call,
    )


def _single_wrapped_urlpattern_list(
    tree: ast.Module,
    node: ast.List,
) -> tuple[str, ast.Call] | None:
    if len(node.elts) != 1:
        return None
    wrapper = node.elts[0]
    if not isinstance(wrapper, ast.Call):
        return None
    if not isinstance(wrapper.func, ast.Name) or wrapper.func.id != "path":
        return None
    if len(wrapper.args) != 2 or wrapper.keywords:
        return None
    if not _is_site_wide_wrapper_route(tree, wrapper.args[0], wrapper.lineno):
        return None
    include_call = wrapper.args[1]
    if not isinstance(include_call, ast.Call):
        return None
    if not isinstance(include_call.func, ast.Name) or include_call.func.id != "include":
        return None
    if len(include_call.args) != 1 or include_call.keywords:
        return None
    included = include_call.args[0]
    if not isinstance(included, ast.Name):
        return None
    return included.id, wrapper


def _is_site_wide_wrapper_route(
    tree: ast.Module,
    node: ast.AST,
    line: int,
) -> bool:
    if isinstance(node, ast.Constant) and node.value == "":
        return True
    if not (
        isinstance(node, ast.Attribute)
        and node.attr == "BASE_PATH"
        and isinstance(node.value, ast.Name)
        and node.value.id == "settings"
    ):
        return False
    return _settings_binding_is_django_conf(tree, line)


def _settings_binding_is_django_conf(tree: ast.Module, line: int) -> bool:
    return _import_before_any(tree, [("django.conf", "settings")], line) and not (
        _name_shadowed_before_line_except_imports(
            tree,
            "settings",
            line,
            allowed_imports=[("django.conf", "settings")],
        )
    )


def _is_fervis_path_call(item: ast.AST) -> bool:
    if not isinstance(item, ast.Call):
        return False
    if not isinstance(item.func, ast.Name) or item.func.id != "path":
        return False
    if len(item.args) != 2 or item.keywords:
        return False
    return _is_fervis_route_arg(item.args[0]) and _is_fervis_include_arg(item.args[1])


def _installed_apps_assignment(
    tree: ast.Module,
    *,
    path: str,
) -> AssignedNode | BlockedPatch:
    assignments = top_level_assignments(tree, "INSTALLED_APPS")
    if not assignments:
        return BlockedPatch(
            path, "Could not find a top-level INSTALLED_APPS assignment."
        )
    return assignments[0]


def _static_installed_app_names(node: ast.AST) -> list[str] | None:
    if not isinstance(node, ast.List | ast.Tuple):
        return None
    apps: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Starred):
            continue
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        apps.append(item.value)
    return apps


def _installed_apps_policy() -> ProtectedListPolicy:
    return ProtectedListPolicy(
        name="INSTALLED_APPS",
        allows_method_call=_is_safe_installed_apps_method_call,
        allows_reassignment=_is_safe_installed_apps_reassignment,
    )


def _urlpattern_list_policy(name: str) -> ProtectedListPolicy:
    return ProtectedListPolicy(
        name=name,
        allows_method_call=_is_safe_urlpattern_method_call,
    )


def _is_safe_installed_apps_method_call(call: ast.Call) -> bool:
    method = _method_name(call)
    if method in {"append", "extend"}:
        return True
    if method != "remove":
        return False
    if len(call.args) != 1 or call.keywords:
        return False
    value = call.args[0]
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value not in set(REQUIRED_DJANGO_APPS)
    )


def _is_safe_urlpattern_method_call(call: ast.Call) -> bool:
    return _method_name(call) in {"append", "extend"}


def _is_safe_installed_apps_reassignment(
    statements: list[ast.stmt],
    index: int,
    node: ast.stmt,
) -> bool:
    if not directly_assigns_name(node, "INSTALLED_APPS"):
        return False
    if index == 0:
        return False
    temp_name = _list_call_name_arg(_assigned_value(node))
    if temp_name is None:
        return False
    previous = statements[index - 1]
    return _assigned_names(previous) == {temp_name} and _is_installed_apps_dedupe(
        _assigned_value(previous)
    )


def _statement_directly_rebinds_name(node: ast.stmt, name: str) -> bool:
    if isinstance(node, ast.Assign):
        return name in target_names(node.targets)
    if isinstance(node, ast.AnnAssign):
        return name in target_names([node.target])
    if isinstance(node, ast.AugAssign):
        return name in target_names([node.target])
    if isinstance(node, ast.Delete):
        return name in target_names(node.targets)
    return False


def _statement_aliases_name(node: ast.stmt, name: str) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return False
    if isinstance(node, ast.Assign):
        return bool(
            top_level_bound_names(node)
        ) and _reference_preserving_expr_contains_name(
            node.value,
            name,
        )
    if isinstance(node, ast.AnnAssign):
        return (
            node.value is not None
            and bool(top_level_bound_names(node))
            and _reference_preserving_expr_contains_name(node.value, name)
        )
    return any(
        _statement_aliases_name(child, name)
        for statements in _child_statement_lists(node)
        for child in statements
    )


def _reference_preserving_expr_contains_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.Tuple | ast.List):
        return any(
            _reference_preserving_expr_contains_name(item, name) for item in node.elts
        )
    if isinstance(node, ast.Starred):
        return _reference_preserving_expr_contains_name(node.value, name)
    return False


def _child_statement_lists(node: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(node, ast.If):
        return [node.body, node.orelse]
    if isinstance(node, ast.For | ast.AsyncFor | ast.While):
        return [node.body, node.orelse]
    if isinstance(node, ast.With | ast.AsyncWith):
        return [node.body]
    if isinstance(node, ast.Try):
        return [
            node.body,
            *[handler.body for handler in node.handlers],
            node.orelse,
            node.finalbody,
        ]
    if isinstance(node, ast.Match):
        return [case.body for case in node.cases]
    return []


def _list_call_name_arg(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "list":
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    arg = node.args[0]
    return arg.id if isinstance(arg, ast.Name) else None


def _is_installed_apps_dedupe(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "reversed":
        return False
    if len(node.args) != 1 or node.keywords:
        return False
    list_call = node.args[0]
    if not isinstance(list_call, ast.Call):
        return False
    if not isinstance(list_call.func, ast.Name) or list_call.func.id != "list":
        return False
    if len(list_call.args) != 1 or list_call.keywords:
        return False
    fromkeys_call = list_call.args[0]
    if not isinstance(fromkeys_call, ast.Call):
        return False
    if not (
        isinstance(fromkeys_call.func, ast.Attribute)
        and fromkeys_call.func.attr == "fromkeys"
        and isinstance(fromkeys_call.func.value, ast.Name)
        and fromkeys_call.func.value.id == "dict"
    ):
        return False
    if len(fromkeys_call.args) != 1 or fromkeys_call.keywords:
        return False
    inner = fromkeys_call.args[0]
    return _is_reversed_installed_apps_call(inner)


def _is_reversed_installed_apps_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reversed"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "INSTALLED_APPS"
    )


def _assigned_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    return None


def _assigned_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _method_calls_on_name(node: ast.AST, name: str) -> list[ast.Call]:
    calls: list[ast.Call] = []

    def visit(child: ast.AST) -> None:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return
        if isinstance(child, ast.Call):
            func = child.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == name
            ):
                calls.append(child)
        for grandchild in ast.iter_child_nodes(child):
            visit(grandchild)

    visit(node)
    return calls


def _method_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _import_before_any(
    tree: ast.Module,
    imports: list[tuple[str, str]],
    line: int,
) -> bool:
    return any(import_before(tree, module, name, line) for module, name in imports)


def _name_shadowed_before_line_except_imports(
    tree: ast.Module,
    name: str,
    line: int,
    *,
    allowed_imports: list[tuple[str, str]],
) -> bool:
    for node in tree.body:
        if node.lineno >= line:
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if bound_name != name:
                    continue
                if _is_allowed_import_from(node, alias, allowed_imports):
                    continue
                return True
        elif isinstance(node, ast.Import):
            if name in {
                alias.asname or alias.name.split(".", 1)[0] for alias in node.names
            }:
                return True
        elif name in top_level_bound_names(node):
            return True
        elif statement_mutates_module_namespace(node, name):
            return True
    return False


def _is_allowed_import_from(
    node: ast.ImportFrom,
    alias: ast.alias,
    allowed_imports: list[tuple[str, str]],
) -> bool:
    return (
        not node.level
        and alias.asname is None
        and any(
            node.module == module and alias.name == import_name
            for module, import_name in allowed_imports
        )
    )


def _safe_module_path(value: str) -> bool:
    return bool(value) and all(part.isidentifier() for part in value.split("."))


def _is_fervis_route_arg(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "django_path"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "routes"
        and is_configured_fervis_call(node.value.value)
    )


def _is_fervis_include_arg(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "include":
        return False
    if len(node.args) != 1 or node.keywords:
        return False
    arg = node.args[0]
    return (
        isinstance(arg, ast.Attribute)
        and arg.attr == "urls"
        and is_configured_fervis_call(arg.value)
    )
