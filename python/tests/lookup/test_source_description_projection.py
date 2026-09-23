"""Grounding and read selection must receive the declared source semantics."""
from fervis.lookup.relation_catalog.model import EndpointRead, CatalogParam, ParamSource
from fervis.lookup.turn_prompts.projections.response_shape import ApiReadResponseShapeProjector, semantic_read_sources_xml


def test_api_read_xml_preserves_description_and_parameter_meaning():
    read = EndpointRead('records', 'list_records', source_metadata={
        'description': 'Operational places include retail outlets & warehouses.',
    }, params=(CatalogParam('records.query.q', 'q', ParamSource.QUERY, 'string',
                            description='Search labels only, not lifecycle state.', default='all'),))
    shape = ApiReadResponseShapeProjector(read).prompt_payload()
    assert shape['description'] == 'Operational places include retail outlets & warehouses.'
    rendered = semantic_read_sources_xml({'sources': [{'api_read': shape}]})
    assert 'Operational places include retail outlets &amp; warehouses.' in rendered
    assert 'Search labels only, not lifecycle state.' in rendered
    assert 'default="all"' in rendered
