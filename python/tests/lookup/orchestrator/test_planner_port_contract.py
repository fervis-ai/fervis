from __future__ import annotations

import pytest

from tests.lookup.orchestrator._planner_ports import (
    _tool_name_planner_arguments,
)


def test_tool_name_planner_callable_requires_current_keyword_contract():
    def prompt_only(prompt: str) -> dict[str, object]:
        return {"prompt": prompt}

    with pytest.raises(TypeError):
        _tool_name_planner_arguments(
            prompt_only,
            prompt="Current prompt",
            tool_specs=(),
        )
