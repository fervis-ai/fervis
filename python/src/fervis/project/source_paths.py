"""Source path-prefix validation and matching."""

from __future__ import annotations


def normalize_source_path_prefix(value: str) -> str:
    text = str(value).strip()
    if not text.startswith("/"):
        raise ValueError("source path prefixes must start with '/'.")
    if text == "/":
        return text
    return f"{text.rstrip('/')}/"


def normalize_source_path_prefixes(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(
            normalize_source_path_prefix(value)
            for value in values
            if str(value).strip()
        )
    )
    if not normalized:
        raise ValueError("source path prefixes must not be empty.")
    return normalized


def source_path_matches(path: str, prefixes: tuple[str, ...]) -> bool:
    normalized_path = _normalize_source_path(path)
    return any(
        prefix == "/"
        or normalized_path == prefix.rstrip("/")
        or normalized_path.startswith(prefix)
        for prefix in prefixes
    )


def _normalize_source_path(path: str) -> str:
    text = str(path).strip()
    if not text.startswith("/"):
        text = f"/{text}"
    if text == "/":
        return text
    return f"{text.rstrip('/')}/"
