from fervis.lookup.answer_program.source_materialization import _bound_row
from fervis.lookup.answer_program.relations import Relation, RelationField, RelationSource, SourceKind, FieldBindingRole
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from tests.lookup.grounding._fixtures import _staff_read


def test_declared_row_kind_is_fixed_data_without_a_response_column():
    read = _staff_read()
    catalog = RelationCatalog(reads=(read,))
    sources = build_api_row_source_catalog(catalog).sources
    source, = (s for s in sources if s.candidate_keys)
    field, = (f for f in source.fields if f.declared_entity_kind)
    assert field.declared_entity_kind == 'staff'
    assert field.choices == ('staff',)
    assert not field.can_carry_lookup_text
    assert FieldBindingRole.IDENTITY not in field.allowed_roles
    relation = Relation('rows', RelationSource(SourceKind.API_READ, read_id=read.id, row_source_id=source.id), (RelationField(field.id, (FieldBindingRole.PREDICATE,)),))
    assert _bound_row({}, relation=relation, catalog=catalog, row_source=source, request_args={}) == {field.id:'staff'}
