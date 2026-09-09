from dataclasses import replace

import pytest

from tests.lookup.relational_engine.test_dependent_reads import _program
from fervis.lookup.relation_catalog import EntityKeyComponentTarget
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_model import ReadAccessCatalog, ReadDependency, AccessArgument
from fervis.lookup.source_reads.access_execution import execute_access_program
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.answer_program.model import RelationProgram


def configured(target_kind):
    program, bindings, catalog = _program()
    child = catalog.read('instruments')
    child = replace(child, params=(replace(child.params[0], entity_target=EntityKeyComponentTarget(target_kind, 'primary', 'id')),))
    catalog = replace(catalog, reads=tuple(child if read.id == child.id else read for read in catalog.reads))
    sources = build_api_row_source_catalog(catalog)
    parent, child = sources.sources
    access = ReadAccessCatalog(sources.sources, (ReadDependency(child.id, parent.id,
        (AccessArgument('facility_id', 'facilities.id'),), 'Each child invocation consumes its declared parent key.'),))
    return program, bindings, catalog, access


@pytest.mark.parametrize('target', ['facilities', 'instruments'])
def test_access_validation_enforces_declared_key_authority(target):
    _, _, _, access = configured(target)
    if target == 'facilities':
        access.validate()
    else:
        with pytest.raises(ValueError, match='authority'):
            access.validate()


@pytest.mark.parametrize('target', ['facilities', 'instruments'])
def test_plain_access_execution_checks_identity_before_child_transport(target):
    program, bindings, catalog, _ = configured(target)
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(endpoint_name)
            return {'responseStatus':200,'responseBody':[{'id':1}]}
    plain = RelationProgram(relations=program.relations, operations=program.operations, parameters=program.parameters)
    def execute():
        return execute_access_program(plain, catalog=catalog, bindings=bindings, read_session=ApiReadSession(Port()))
    if target == 'instruments':
        with pytest.raises(ValueError, match='authority'):
            execute()
        assert calls == []
    else:
        assert execute().engine_output.issue is None
        assert calls == ['facilities', 'instruments']


@pytest.mark.parametrize('source_kind', ['facilities', 'instruments'])
def test_plain_access_checks_certified_scalar_identity_before_transport(source_kind):
    from fervis.lookup.answer_program.values import ConstantRef, FactValue
    from fervis.lookup.canonical_data import EntityKeyValue, EntityKeyComponentValue
    from fervis.lookup.answer_program.relations import EndpointParamBinding
    program, bindings, catalog, _ = configured('facilities')
    value = FactValue.identity(id='selected', key=EntityKeyValue(source_kind, 'primary',
        (EntityKeyComponentValue('id', 1),)), proof_refs=('current-grounding',))
    child = program.relations[1]
    child = replace(child, source=replace(child.source, argument_relation_id='',
        param_bindings=(EndpointParamBinding(child.source.param_bindings[0].param_id,
            ConstantRef(value.id, 'current-grounding', value, component='key_component:id')),)))
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, args))
            return {'responseStatus':200,'responseBody':[{'id':1}]}
    plain = RelationProgram(relations=(child,))
    def execute():
        return execute_access_program(plain, catalog=catalog, bindings=bindings, read_session=ApiReadSession(Port()))
    if source_kind == 'instruments':
        with pytest.raises(ValueError, match='authority'):
            execute()
        assert calls == []
    else:
        assert execute().engine_output.issue is None
        assert calls == [('instruments', {'facility_id':1})]


@pytest.mark.parametrize('target_type,values,accepted', [
    ('integer', (1, 2), True), ('integer', ('wrong',), False), ('integer', (1, 'wrong'), False),
    ('uuid', ('00000000-0000-0000-0000-000000000001',), True), ('uuid', ('wrong',), False)])
@pytest.mark.parametrize('collection', [False, True])
def test_certified_components_obey_actual_scalar_type_before_transport(target_type, values, accepted, collection):
    from fervis.lookup.answer_program.values import ConstantRef, FactValue
    from fervis.lookup.canonical_data import EntityKeyValue, EntityKeyComponentValue
    from fervis.lookup.answer_program.relations import EndpointParamBinding
    program, bindings, catalog, _ = configured('facilities')
    read = catalog.read('instruments')
    read = replace(read, params=(replace(read.params[0], type=target_type),))
    catalog = replace(catalog, reads=tuple(read if item.id == read.id else item for item in catalog.reads))
    keys = tuple(EntityKeyValue('facilities', 'primary', (EntityKeyComponentValue('id', value),)) for value in values)
    value = (FactValue.identity_set(id='selected', keys=keys, proof_refs=('current-grounding',)) if collection else
             FactValue.identity(id='selected', key=keys[-1], proof_refs=('current-grounding',)))
    child = program.relations[1]
    child = replace(child, source=replace(child.source, argument_relation_id='',
        param_bindings=(EndpointParamBinding(child.source.param_bindings[0].param_id,
            ConstantRef(value.id, 'current-grounding', value, component='key_component:id')),)))
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(args['facility_id'])
            return {'responseStatus':200,'responseBody':[{'id':1}]}
    def execute():
        return execute_access_program(RelationProgram(relations=(child,)), catalog=catalog,
            bindings=bindings, read_session=ApiReadSession(Port()))
    if accepted:
        assert execute().engine_output.issue is None
        assert set(calls) == set(values if collection else values[-1:])
    else:
        with pytest.raises(ValueError):
            execute()
        assert calls == []
