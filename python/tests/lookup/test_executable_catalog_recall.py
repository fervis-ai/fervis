import pytest

from fervis.lookup.answer_program.values import FactValue, LiteralType




def test_integer_parameter_projection_preserves_large_values_and_rejects_fractions():
    from fervis.lookup.source_binding.param_values import (
        fact_value_parameter_projection,
    )
    from fervis.lookup.answer_program.values import ValueProjectionKind

    def projected(text):
        return fact_value_parameter_projection(
            FactValue.literal(id="n", literal_type=LiteralType.NUMBER, value=text),
            projection=ValueProjectionKind.WHOLE_VALUE,
            component_id=None,
            type_name="integer",
            choices=(),
        )

    assert projected("9007199254740993") == 9007199254740993
    with pytest.raises(ValueError, match="exact integral"):
        projected("3.5")
