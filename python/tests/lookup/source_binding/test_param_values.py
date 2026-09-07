from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.canonical_data import EntityKeyComponentValue, EntityKeyValue
from fervis.lookup.relation_catalog import EntityKeyComponentTarget
from fervis.lookup.source_binding.param_values import compatible_fact_value_projections


def _staff_identity() -> FactValue:
    return FactValue.identity(
        id="canonical_staff_1",
        known_input_id="i1",
        key=EntityKeyValue(
            entity_kind="staff",
            key_id="primary_key",
            components=(EntityKeyComponentValue("staff_id", "staff_1"),),
        ),
        display_value="Ada",
        proof_refs=("resolver:list_staff_list",),
    )


def test_identity_requires_declared_entity_metadata() -> None:
    assert compatible_fact_value_projections(
        _staff_identity(),
        type_name="uuid",
        choices=(),
        entity_target=None,
    ) == ()


def test_declared_entity_metadata_rejects_a_different_identity_type() -> None:
    assert compatible_fact_value_projections(
        _staff_identity(),
        type_name="uuid",
        choices=(),
        entity_target=EntityKeyComponentTarget(
            entity_kind="location",
            key_id="primary_key",
            component_id="location_id",
        ),
    ) == ()


def test_json_numeric_projection_preserves_the_canonical_value():
    from fervis.lookup.answer_program.values import LiteralType, ValueProjectionKind
    for type_name in ('number', 'float', 'double'):
        for text, compatible in (('2', True), ('0.1', True), ('9007199254740993', type_name == 'number'), ('1e-400', False), ('1e400', type_name == 'number')):
            value = FactValue.literal(id='threshold', literal_type=LiteralType.NUMBER,
                                      value=text, label='threshold', proof_refs=('question',))
            options = compatible_fact_value_projections(value, type_name=type_name,
                                                       choices=(), entity_target=None)
            if compatible:
                from fervis.lookup.source_binding.param_values import fact_value_parameter_projection
                from decimal import Decimal
                projected = fact_value_parameter_projection(value, projection=ValueProjectionKind.WHOLE_VALUE, component_id=None, type_name=type_name, choices=())
                assert Decimal(str(projected)) == Decimal(text)
            assert options == (((ValueProjectionKind.WHOLE_VALUE, None),) if compatible else ()), (type_name, text)
