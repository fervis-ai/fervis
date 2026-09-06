"""A root object is a real parent row, even though its path is empty."""
from dataclasses import replace

from fervis.lookup.relation_catalog import (
    CandidateKey, CandidateKeyComponent, CatalogField, EndpointRead,
    EntityReference, EntityReferenceComponent, RelationCatalog, RowCardinality, RowPath,
)
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.response import extract_row_source_rows


def child_source(*, shadow_owner=False):
    read = EndpointRead(id='read', endpoint_name='read', resource_names=('observation',),
        row_paths=(RowPath(id='root', path='', cardinality=RowCardinality.ONE),
                   RowPath(id='items', path='items', parent_path='', cardinality=RowCardinality.MANY)),
        fields=(CatalogField(ref='owner', path='owner_id', type='string', row_path_id='root'),
                CatalogField(ref='item', path='items.id', type='string', row_path_id='items'),
                *((CatalogField(ref='local_owner', path='items.owner_id', type='string', row_path_id='items'),) if shadow_owner else ())),
        candidate_keys=(CandidateKey(id='pk', entity_kind='item', primary=True,
            components=(CandidateKeyComponent(id='id', field_ref='item'),)),),
        entity_references=(EntityReference(id='owner_ref', target_entity_kind='owner', target_key_id='pk',
            components=(EntityReferenceComponent(target_component_id='id', local_field_ref='owner'),)),))
    return next(source for source in build_api_row_source_catalog(RelationCatalog(reads=(read,))).sources if source.row_path == 'items')


def test_root_parent_identity_reference_reaches_each_child_row():
    source = child_source()
    assert source.parent_row_cardinality == RowCardinality.ONE
    assert source.field('owner_id').path == 'owner_id'
    assert [reference.id for reference in source.entity_references] == ['owner_ref']
    assert [key.entity_kind for key in source.candidate_keys] == ['item']
    assert extract_row_source_rows({'owner_id': 'owner-A', 'items': [{'id': 'a'}, {'id': 'b'}]}, row_source=source) == (
        {'owner_id': 'owner-A', 'id': 'a'}, {'owner_id': 'owner-A', 'id': 'b'},
    )
    assert extract_row_source_rows({'owner_id': 'owner-A', 'items': []}, row_source=source) == ()


def test_shadowed_parent_field_cannot_certify_the_child_value_as_a_reference():
    source = child_source(shadow_owner=True)
    assert source.entity_references == ()
    assert source.field('owner_id').path == 'items.owner_id'
    assert extract_row_source_rows({'owner_id': 'parent-owner', 'items': [{'id': 'a', 'owner_id': 'local-value'}]}, row_source=source) == (
        {'owner_id': 'local-value', 'id': 'a'},
    )


def test_grandparent_reference_is_not_reassigned_to_an_intermediate_row():
    read = EndpointRead(id='nested', endpoint_name='nested', resource_names=('observation',),
        row_paths=(RowPath(id='root', path='', cardinality=RowCardinality.ONE),
                   RowPath(id='groups', path='groups', parent_path='', cardinality=RowCardinality.MANY),
                   RowPath(id='items', path='groups.items', parent_path='groups', cardinality=RowCardinality.MANY)),
        fields=(CatalogField(ref='root_owner', path='owner_id', type='string', row_path_id='root'),
                CatalogField(ref='group_owner', path='groups.owner_id', type='string', row_path_id='groups'),
                CatalogField(ref='item', path='groups.items.id', type='string', row_path_id='items')),
        entity_references=tuple(EntityReference(id=ref, target_entity_kind=ref, target_key_id='pk',
            components=(EntityReferenceComponent(target_component_id='id', local_field_ref=ref),))
            for ref in ('root_owner', 'group_owner')))
    source = next(s for s in build_api_row_source_catalog(RelationCatalog(reads=(read,))).sources if s.row_path == 'groups.items')
    assert [ref.id for ref in source.entity_references] == ['group_owner']
    assert source.field('owner_id').path == 'groups.owner_id'
    assert extract_row_source_rows({'owner_id': 'ROOT', 'groups': [{'owner_id': 'GROUP', 'items': [{'id': 'child'}]}]}, row_source=source) == (
        {'owner_id': 'GROUP', 'id': 'child'},
    )

    without_group_owner = replace(read,
        fields=tuple(field for field in read.fields if field.ref != 'group_owner'),
        entity_references=tuple(ref for ref in read.entity_references if ref.id != 'group_owner'))
    source = next(s for s in build_api_row_source_catalog(RelationCatalog(reads=(without_group_owner,))).sources if s.row_path == 'groups.items')
    assert source.entity_references == ()
    assert {field.id for field in source.fields} == {'id'}
    assert extract_row_source_rows({'owner_id': 'ROOT', 'groups': [{'items': [{'id': 'child'}]}]}, row_source=source) == ({'id': 'child'},)


def test_child_object_shadows_all_parent_fields_inside_that_container():
    read = EndpointRead(id='object', endpoint_name='object', resource_names=('observation',),
        row_paths=(RowPath(id='root', path='', cardinality=RowCardinality.ONE),
                   RowPath(id='items', path='items', parent_path='', cardinality=RowCardinality.MANY)),
        fields=(CatalogField(ref='parent_id', path='owner.id', type='string', row_path_id='root'),
                CatalogField(ref='child_name', path='items.owner.name', type='string', row_path_id='items')),
        entity_references=(EntityReference(id='parent_ref', target_entity_kind='owner', target_key_id='pk',
            components=(EntityReferenceComponent(target_component_id='id', local_field_ref='parent_id'),)),))
    source = next(s for s in build_api_row_source_catalog(RelationCatalog(reads=(read,))).sources if s.row_path == 'items')
    assert source.entity_references == ()
    assert {field.path for field in source.fields} == {'items.owner.name'}
    assert extract_row_source_rows({'owner': {'id': 'ROOT'}, 'items': [{'owner': {'name': 'child'}}]}, row_source=source) == ({'owner': {'name': 'child'}},)
