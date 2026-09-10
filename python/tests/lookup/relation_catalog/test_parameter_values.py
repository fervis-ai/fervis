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


@pytest.mark.parametrize('type_name,value,choices', [
    ('integer',2,('1',)),('boolean',False,('true',)),('float',2.5,('1.5',)),
])
def test_typed_scalar_parameter_enums_are_enforced(type_name,value,choices):
    with pytest.raises(ValueError):
        parse_catalog_parameter_value(value,type_name=type_name,choices=choices)


@pytest.mark.parametrize('type_name,value,choices', [
    ('integer',1,('1',)),('boolean',True,('true',)),('float',1.5,('1.50',)),
])
def test_parameter_enum_comparison_uses_declared_scalar_types(type_name,value,choices):
    assert parse_catalog_parameter_value(value,type_name=type_name,choices=choices) == value
