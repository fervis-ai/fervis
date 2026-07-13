"""Conservative module-level binding analysis for mount validation."""

from __future__ import annotations

import ast


AssignedNode = ast.Assign | ast.AnnAssign


def top_level_assignment(tree: ast.Module, name: str) -> AssignedNode | None:
    matches = top_level_assignments(tree, name)
    if len(matches) != 1:
        return None
    return matches[0]


def top_level_assignments(tree: ast.Module, name: str) -> list[AssignedNode]:
    matches: list[AssignedNode] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            matches.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            matches.append(node)
    return matches


def directly_assigns_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    return (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    )


def has_import(tree: ast.Module, module: str, name: str) -> bool:
    return import_lineno(tree, module, name) is not None


def is_configured_fervis_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "configured_fervis"
        and not node.args
        and not node.keywords
    )


def import_before(
    tree: ast.Module,
    module: str,
    name: str,
    line: int,
) -> bool:
    import_line = import_lineno(tree, module, name)
    return import_line is not None and import_line < line


def import_lineno(tree: ast.Module, module: str, name: str) -> int | None:
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and not node.level
            and node.module == module
        ):
            if any(alias.name == name and alias.asname is None for alias in node.names):
                return node.lineno
    return None


def module_import_before(
    tree: ast.Module,
    module: str,
    bound_name: str,
    line: int,
) -> bool:
    import_line = module_import_lineno(tree, module, bound_name)
    return import_line is not None and import_line < line


def module_import_lineno(
    tree: ast.Module,
    module: str,
    bound_name: str,
) -> int | None:
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            alias_bound_name = alias.asname or alias.name.split(".", 1)[0]
            if alias.name == module and alias_bound_name == bound_name:
                return node.lineno
    return None


def top_level_name_shadowed_before_line(
    tree: ast.Module,
    name: str,
    line: int,
    *,
    allowed_import: tuple[str, str] | None = None,
    allowed_module_import: tuple[str, str] | None = None,
) -> bool:
    for node in tree.body:
        if node.lineno >= line:
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if bound_name != name:
                    continue
                if (
                    allowed_import is not None
                    and not node.level
                    and node.module == allowed_import[0]
                    and alias.name == allowed_import[1]
                    and alias.asname is None
                ):
                    continue
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                if bound_name != name:
                    continue
                if (
                    allowed_module_import is not None
                    and alias.name == allowed_module_import[0]
                    and bound_name == allowed_module_import[1]
                ):
                    continue
                return True
        elif name in top_level_bound_names(node):
            return True
        elif statement_mutates_module_namespace(node, name):
            return True
    return False


def top_level_assignment_after_line(
    tree: ast.Module,
    name: str,
    line: int,
) -> bool:
    for node in tree.body:
        if node.lineno <= line:
            continue
        if name in top_level_bound_names(node):
            return True
        if statement_mutates_module_namespace(node, name):
            return True
    return False


def module_binding_modified_after_line(
    tree: ast.Module,
    name: str,
    line: int,
) -> bool:
    for node in tree.body:
        if node.lineno <= line:
            continue
        if statement_modifies_module_binding(node, name):
            return True
    return False


def statement_modifies_module_binding(node: ast.stmt, name: str) -> bool:
    if name in top_level_bound_names(node):
        return True
    if statement_mutates_module_namespace(node, name):
        return True
    return _statement_calls_method_on_name(node, name)


def statement_mutates_module_namespace(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return False
    if isinstance(node, ast.Call) and _call_mutates_module_namespace(node, name):
        return True
    return any(
        statement_mutates_module_namespace(child, name)
        for child in ast.iter_child_nodes(node)
        if not isinstance(child, ast.Lambda)
    )


def top_level_bound_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        return target_names(node.targets) | namedexpr_bound_names(node.value)
    if isinstance(node, ast.AnnAssign):
        value_names = namedexpr_bound_names(node.value) if node.value else set()
        return target_names([node.target]) | value_names
    if isinstance(node, ast.AugAssign):
        return target_names([node.target]) | namedexpr_bound_names(node.value)
    if isinstance(node, ast.Delete):
        return target_names(node.targets)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.For | ast.AsyncFor):
        return (
            target_names([node.target])
            | namedexpr_bound_names(node.iter)
            | bound_names_in_statements([*node.body, *node.orelse])
        )
    if isinstance(node, ast.With | ast.AsyncWith):
        targets = [
            item.optional_vars for item in node.items if item.optional_vars is not None
        ]
        context_names = {
            name
            for item in node.items
            for name in namedexpr_bound_names(item.context_expr)
        }
        return (
            target_names(targets) | context_names | bound_names_in_statements(node.body)
        )
    if isinstance(node, ast.If):
        return namedexpr_bound_names(node.test) | bound_names_in_statements(
            [*node.body, *node.orelse]
        )
    if isinstance(node, ast.Try):
        handler_names = {
            handler.name for handler in node.handlers if isinstance(handler.name, str)
        }
        return (
            handler_names
            | try_handler_type_names(node)
            | bound_names_in_statements(
                [
                    *node.body,
                    *[stmt for handler in node.handlers for stmt in handler.body],
                    *node.orelse,
                    *node.finalbody,
                ]
            )
        )
    if isinstance(node, ast.Match):
        return (
            namedexpr_bound_names(node.subject)
            | bound_names_in_statements(
                [stmt for case in node.cases for stmt in case.body]
            )
            | {
                name
                for case in node.cases
                for name in pattern_bound_names(case.pattern)
            }
        )
    return set()


def bound_names_in_statements(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in statements:
        names.update(top_level_bound_names(statement))
    return names


def pattern_bound_names(pattern: ast.pattern) -> set[str]:
    if isinstance(pattern, ast.MatchAs):
        nested = pattern_bound_names(pattern.pattern) if pattern.pattern else set()
        return nested | ({pattern.name} if pattern.name else set())
    if isinstance(pattern, ast.MatchStar):
        return {pattern.name} if pattern.name else set()
    if isinstance(pattern, ast.MatchMapping):
        names: set[str] = set()
        for child_pattern in pattern.patterns:
            names.update(pattern_bound_names(child_pattern))
        if pattern.rest:
            names.add(pattern.rest)
        return names
    if isinstance(pattern, ast.MatchSequence):
        return {
            name for nested in pattern.patterns for name in pattern_bound_names(nested)
        }
    if isinstance(pattern, ast.MatchClass):
        return {
            name
            for nested in [*pattern.patterns, *pattern.kwd_patterns]
            for name in pattern_bound_names(nested)
        }
    if isinstance(pattern, ast.MatchOr):
        return {
            name for nested in pattern.patterns for name in pattern_bound_names(nested)
        }
    return set()


def try_handler_type_names(node: ast.Try) -> set[str]:
    return {
        name
        for handler in node.handlers
        if handler.type is not None
        for name in namedexpr_bound_names(handler.type)
    }


def namedexpr_bound_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.NamedExpr):
        names.update(target_names([node.target]))
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            ast.Lambda | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            continue
        if isinstance(child, ast.NamedExpr):
            names.update(target_names([child.target]))
        names.update(namedexpr_bound_names(child))
    return names


def target_names(targets: list[ast.expr]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name):
                names.add(target.value.id)
            global_name = _global_subscript_target_name(target)
            if global_name is not None:
                names.add(global_name)
        elif isinstance(target, ast.Tuple | ast.List):
            names.update(target_names(list(target.elts)))
    return names


def _statement_calls_method_on_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return False
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == name
    ):
        return True
    return any(
        _statement_calls_method_on_name(child, name)
        for child in ast.iter_child_nodes(node)
        if not isinstance(child, ast.Lambda)
    )


def _global_subscript_target_name(target: ast.Subscript) -> str | None:
    if not (
        isinstance(target.value, ast.Call)
        and isinstance(target.value.func, ast.Name)
        and target.value.func.id == "globals"
        and not target.value.args
        and not target.value.keywords
    ):
        return None
    if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
        return target.slice.value
    return None


def _call_mutates_module_namespace(call: ast.Call, name: str) -> bool:
    if _is_namespace_update_call(call):
        if any(keyword.arg == name for keyword in call.keywords):
            return True
        return bool(call.args) or any(keyword.arg is None for keyword in call.keywords)
    if _is_module_setattr_call(call):
        if len(call.args) < 2:
            return True
        target_name = call.args[1]
        return not isinstance(target_name, ast.Constant) or target_name.value == name
    return False


def _is_namespace_update_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "update"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id in {"globals", "locals", "vars"}
        and not call.func.value.args
        and not call.func.value.keywords
    )


def _is_module_setattr_call(call: ast.Call) -> bool:
    if not (
        isinstance(call.func, ast.Name)
        and call.func.id == "setattr"
        and call.args
        and _is_current_module_expr(call.args[0])
    ):
        return False
    return True


def _is_current_module_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "modules"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sys"
        and isinstance(node.slice, ast.Name)
        and node.slice.id == "__name__"
    )
