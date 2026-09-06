"""Host response body normalization shared by read executors."""

from __future__ import annotations

import json
from typing import Any
from fervis.host_api.contracts.response_page import ResponsePage, ResponseFormat


def response_page(response: object) -> ResponsePage:
    status = int(getattr(response, "status_code"))
    media_type = _content_type(response).split(";", 1)[0].strip().lower()
    if (
        not media_type
        or media_type == "application/json"
        or media_type.endswith("+json")
    ):
        try:
            return ResponsePage(
                status, _json_safe(_response_json(response)), ResponseFormat.JSON
            )
        except ValueError:
            pass
    text = str(getattr(response, "text", "") or "") or _response_text(response)
    return ResponsePage(
        status,
        text if text else None,
        ResponseFormat.TEXT if text else ResponseFormat.EMPTY,
    )


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
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode(
            getattr(response, "charset", None) or "utf-8", errors="replace"
        )
    if isinstance(content, str):
        return content
    return ""


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value))
