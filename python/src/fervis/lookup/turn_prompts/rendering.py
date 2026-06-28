"""Prompt rendering primitives for Lookup model turns."""

from __future__ import annotations

import json
from typing import Any


class PromptRenderer:
    def text(self, value: object) -> str:
        return str(value)

    def json(self, value: Any, *, indent: int | None = None) -> str:
        return json.dumps(value, indent=indent, sort_keys=True)
