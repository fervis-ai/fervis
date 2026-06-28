"""Host response body normalization shared by read executors."""

from __future__ import annotations

import json
from typing import Any


def response_body(response: object) -> Any:
    try:
        json_body = _response_json(response)
        if json_body is not None:
            return _json_safe(json_body)
    except ValueError:
        pass
    content = getattr(response, "content", b"")
    text = str(getattr(response, "text", "") or "")
    if not content and not text:
        text = _response_text(response)
    if not content and not text:
        return None
    return {
        "contentType": _content_type(response),
        "text": text,
    }


def _response_json(response: object) -> Any:
    json_value = getattr(response, "json", None)
    if callable(json_value):
        return json_value()
    if json_value is not None:
        return json_value
    get_json = getattr(response, "get_json", None)
    if callable(get_json):
        return get_json()
    raise ValueError("response does not expose JSON")


def _content_type(response: object) -> str:
    headers = getattr(response, "headers", {}) or {}
    if hasattr(headers, "get"):
        return str(headers.get("content-type") or headers.get("Content-Type") or "")
    return ""


def _response_text(response: object) -> str:
    get_data = getattr(response, "get_data", None)
    if callable(get_data):
        try:
            value = get_data(as_text=True)
        except TypeError:
            value = get_data()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value or "")
    return ""


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value))
