"""Framework-neutral endpoint resource-name projection."""

from __future__ import annotations

import re


def endpoint_resource_names(
    *,
    tags: tuple[str, ...],
    operation_id: str,
    path_template: str,
) -> tuple[str, ...]:
    names = [
        *(_phrase(tag) for tag in tags),
        _operation_phrase(operation_id),
        _path_phrase(path_template),
    ]
    return tuple(dict.fromkeys(name for name in names if name))


def _operation_phrase(operation_id: str) -> str:
    words = list(_words(operation_id))
    while words and words[0] in {
        "list",
        "get",
        "read",
        "retrieve",
        "search",
        "find",
        "fetch",
    }:
        words = words[1:]
    return _words_to_phrase(tuple(words))


def _path_phrase(path_template: str) -> str:
    segments = [
        segment
        for segment in path_template.strip("/").split("/")
        if segment and not segment.startswith("{")
    ]
    return _phrase(segments[-1] if segments else "")


def _phrase(value: str) -> str:
    return _words_to_phrase(_words(value))


def _words(value: str) -> tuple[str, ...]:
    normalized = value.replace("-", "_").replace(" ", "_")
    words: list[str] = []
    for part in normalized.split("_"):
        words.extend(
            match.group(0).lower()
            for match in re.finditer(
                r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+",
                part,
            )
        )
    return tuple(words)


def _words_to_phrase(words: tuple[str, ...]) -> str:
    values = tuple(word for word in words if word)
    return " ".join(values)
