import pytest

from fervis.lookup.grounding.model import GroundedInputUse


def test_grounded_input_use_requires_requested_fact_scope() -> None:
    with pytest.raises(
        ValueError,
        match="grounded input use requires requested fact id",
    ):
        GroundedInputUse(
            id="use_1",
            value_id="value_1",
            row_source_id="source_1",
            param_id="param_1",
            requested_fact_id="",
        )
