from tests.lookup.relational_engine.test_dependent_reads import _program
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_model import ReadAccessCatalog, ReadDependency, AccessArgument
from fervis.lookup.relational_sql.acquisition import ApiView, materialize_views
from fervis.lookup.relational_sql.execution import execute_query


def test_sql_observations_use_dependent_api_reads_and_retain_their_proof():
    _, _, catalog = _program()
    parent, child = build_api_row_source_catalog(catalog).sources
    access = ReadAccessCatalog((parent, child), (ReadDependency(child.id, parent.id,
        (AccessArgument('facility_id', 'facilities.id'),), 'Complete traversal.'),))
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, args))
            rows = [{'id': 1}, {'id': 2}] if endpoint_name == 'facilities' else [{'id': args['facility_id'] * 10}]
            return {'responseStatus': 200, 'responseBody': rows}
    observed = materialize_views((ApiView('instruments', child.id, {'id': child.fields[0].id}, {}),),
        catalog=catalog, read_session=ApiReadSession(Port()), access=access)
    result = execute_query('SELECT COUNT(*) AS count FROM instruments', tables=observed.tables)
    assert result.rows == ((2,),)
    assert calls == [('facilities', {}), ('instruments', {'facility_id': 1}), ('instruments', {'facility_id': 2})]
    assert {'read:facilities', 'read:instruments'} <= set(observed.proof_refs['instruments'])


def test_acquired_numeric_and_timestamp_values_keep_their_observed_precision():
    from datetime import datetime, timezone
    from decimal import Decimal
    from fervis.lookup.relation_catalog.model import EndpointRead, RelationCatalog, CatalogField, RowPath, RowCardinality
    read = EndpointRead('observations', 'observations',
        row_paths=(RowPath('root', '', RowCardinality.MANY),),
        fields=(CatalogField('value', 'number', path='value', row_path_id='root'),
                CatalogField('observed_at', 'datetime', path='observed_at', row_path_id='root')))
    catalog = RelationCatalog(reads=(read,))
    source = build_api_row_source_catalog(catalog).sources[0]
    fields = {field.path: field.id for field in source.fields}
    class Port:
        def read(self, **kwargs):
            return {'responseStatus': 200, 'responseBody': [
                {'value': '0.000000000000000000001', 'observed_at': '2026-09-08T01:00:00+03:00'}]}
    observed = materialize_views((ApiView('observations', source.id,
        {'value': fields['value'], 'observed_at': fields['observed_at']}, {}),),
        catalog=catalog, read_session=ApiReadSession(Port()))
    result = execute_query('SELECT value, observed_at FROM observations', tables=observed.tables)
    assert result.rows == ((Decimal('0.000000000000000000001'), datetime(2026,9,7,22,tzinfo=timezone.utc)),)


def test_request_arguments_carry_existing_grounding_instead_of_fabricated_proof():
    import pytest
    from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType
    from fervis.lookup.relational_sql.execution import QueryValidationError
    _, _, catalog = _program()
    _, child = build_api_row_source_catalog(catalog).sources
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, args))
            return {'responseStatus': 200, 'responseBody': [{'id': 10}]}
    def view(value):
        return ApiView('instruments', child.id, {'id': child.fields[0].id}, {'facility_id': value})
    with pytest.raises(QueryValidationError, match='grounded'):
        materialize_views((view(1),), catalog=catalog, read_session=ApiReadSession(Port()))
    assert calls == []
    grounded = FactValue.literal(id='facility_input', literal_type=LiteralType.NUMBER,
                                value='1', proof_refs=('grounding:facility-selection',))
    observed = materialize_views((view(ConstantRef(grounded.id, 'grounding:current-run', grounded)),),
                                catalog=catalog, read_session=ApiReadSession(Port()))
    assert calls == [('instruments', {'facility_id': 1})]
    assert execute_query('SELECT COUNT(*) FROM instruments', tables=observed.tables).rows == ((1,),)


def test_access_discovery_uses_only_unbound_requirements_of_selected_queries():
    from types import SimpleNamespace
    from fervis.lookup.relational_sql.authoring import QueryArgument
    from fervis.lookup.relational_sql.catalog import build_query_view_catalog
    from fervis.lookup.relational_sql.binding import reads_requiring_access_discovery
    _,_,catalog=_program()
    views=build_query_view_catalog(catalog)
    child=next(view for view in views.views if views.tables[view.name]['read_id']=='instruments')
    parent=next(view for view in views.views if views.tables[view.name]['read_id']=='facilities')
    authored=SimpleNamespace(referenced_views=(parent.name,),request_arguments=())
    assert reads_requiring_access_discovery(authored,views.views,catalog=catalog)==()
    authored=SimpleNamespace(referenced_views=(child.name,),request_arguments=())
    assert reads_requiring_access_discovery(authored,views.views,catalog=catalog)==('instruments',)
    authored.request_arguments=(QueryArgument(child.name,'facility_id','reference_key'),)
    assert reads_requiring_access_discovery(authored,views.views,catalog=catalog)==()
