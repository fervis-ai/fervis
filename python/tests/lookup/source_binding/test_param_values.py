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
