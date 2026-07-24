import pytest

from fervis.lookup.relation_catalog.parameter_values import (
    CatalogParameterValueError,
    parse_catalog_parameter_value,
)


@pytest.mark.parametrize(
    ("type_name", "value"),
    (
        ("date", "completed"),
        ("datetime", "this month"),
        ("time", "morning"),
        ("decimal", "many"),
    ),
)
def test_declared_scalar_types_reject_unparseable_text(
    type_name: str,
    value: str,
) -> None:
    with pytest.raises(CatalogParameterValueError):
        parse_catalog_parameter_value(value, type_name=type_name)


@pytest.mark.parametrize(
    ("type_name", "value"),
    (
        ("date", "2026-03-14"),
        ("datetime", "2026-03-14T12:30:00+03:00"),
        ("time", "12:30:00"),
        ("decimal", "123.45"),
        ("string", "completed"),
    ),
)
def test_declared_scalar_types_preserve_valid_wire_values(
    type_name: str,
    value: str,
) -> None:
    assert parse_catalog_parameter_value(value, type_name=type_name) == value
