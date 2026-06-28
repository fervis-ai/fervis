"""Text-preserving Python source patch helpers."""

from __future__ import annotations

import ast

from .common import BlockedPatch, parse_python_source


def replace_node_source(text: str, node: ast.AST, replacement: str) -> str:
    newline = newline_for(text)
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    leading = lines[start][: node.col_offset]
    trailing = ""
    if node.end_col_offset is not None:
        trailing = strip_line_ending(lines[end - 1][node.end_col_offset :])
    replacement_lines = replacement.splitlines()
    new_lines = [leading + replacement_lines[0] + newline]
    new_lines.extend(line + newline for line in replacement_lines[1:])
    if trailing:
        new_lines[-1] = strip_line_ending(new_lines[-1]) + trailing + newline
    return "".join([*lines[:start], *new_lines, *lines[end:]])


def append_literal_sequence_items(
    text: str,
    node: ast.List | ast.Tuple,
    missing_values: list[str],
    *,
    path: str,
) -> str | BlockedPatch:
    if not missing_values:
        return text
    if ast.get_source_segment(text, node) is None:
        return BlockedPatch(path, "Could not locate literal sequence source.")
    if node.lineno == node.end_lineno:
        if node.elts:
            return BlockedPatch(
                path,
                "Non-empty single-line sequences must be edited manually.",
            )
        opener = "[" if isinstance(node, ast.List) else "("
        closer = "]" if isinstance(node, ast.List) else ")"
        replacement = "\n".join(
            [opener, *[f'    "{value}",' for value in missing_values], closer]
        )
        return replace_node_source(text, node, replacement)

    newline = newline_for(text)
    text = ensure_multiline_sequence_trailing_comma(text, node, path=path)
    if isinstance(text, BlockedPatch):
        return text
    lines = text.splitlines(keepends=True)
    closing_index = (node.end_lineno or node.lineno) - 1
    closing_line = lines[closing_index]
    closing_indent = closing_line[: len(closing_line) - len(closing_line.lstrip())]
    inserted = [f'{closing_indent}    "{value}",{newline}' for value in missing_values]
    return "".join([*lines[:closing_index], *inserted, *lines[closing_index:]])


def insert_import_line(text: str, import_line: str) -> str:
    lines = text.splitlines(keepends=True)
    lines.insert(import_insertion_index(text), import_line + newline_for(text))
    return "".join(lines)


def insert_after_node(text: str, node: ast.AST, line: str) -> str:
    lines = text.splitlines(keepends=True)
    lines.insert(node.end_lineno or node.lineno, line + newline_for(text))
    return "".join(lines)


def insert_before_node(text: str, node: ast.AST, line: str) -> str:
    lines = text.splitlines(keepends=True)
    lines.insert(node.lineno - 1, line + newline_for(text))
    return "".join(lines)


def ensure_multiline_sequence_trailing_comma(
    text: str,
    node: ast.List | ast.Tuple,
    *,
    path: str,
) -> str | BlockedPatch:
    if not node.elts:
        return text
    last_item = node.elts[-1]
    if last_item.end_lineno is None or last_item.end_col_offset is None:
        return BlockedPatch(path, "Could not locate final sequence item source.")
    lines = text.splitlines(keepends=True)
    item_index = last_item.end_lineno - 1
    line = lines[item_index]
    suffix = strip_line_ending(line[last_item.end_col_offset :]).lstrip()
    if suffix.startswith(","):
        return text
    lines[item_index] = (
        line[: last_item.end_col_offset] + "," + line[last_item.end_col_offset :]
    )
    return "".join(lines)


def rstrip_line_endings(text: str) -> str:
    while text.endswith(("\r", "\n")):
        text = text[:-1]
    return text


def newline_for(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def strip_line_ending(text: str) -> str:
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


def import_insertion_index(text: str) -> int:
    tree = parse_python_source(text)
    index = python_header_line_count(text)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        index = tree.body[0].end_lineno or 1
    for node in tree.body:
        if node.lineno <= index:
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            index = node.end_lineno or node.lineno
            continue
        break
    return index


def python_header_line_count(text: str) -> int:
    lines = text.splitlines()
    if not lines:
        return 0
    if lines[0].startswith("#!"):
        return 2 if len(lines) > 1 and is_encoding_comment(lines[1]) else 1
    if is_encoding_comment(lines[0]):
        return 1
    if (
        len(lines) > 1
        and lines[0].lstrip().startswith("#")
        and is_encoding_comment(lines[1])
    ):
        return 2
    return 0


def is_encoding_comment(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("#") and "coding" in stripped
