"""Computed API projections can declare the same identities as OpenAPI reads."""
import pytest
from rest_framework import serializers
from rest_framework.views import APIView
from fervis.host_api.adapters.django.catalog import _build_contract


class ResponseSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    category_code = serializers.CharField()


def contract(metadata):
    class View(APIView):
        serializer_class = ResponseSerializer
        fervis_response_cardinality = 'many'
        fervis_relation_metadata = metadata
    return _build_contract(path='/records/', url_name='records', view_class=View, converters={})


def test_computed_projection_declares_owned_keys_and_references():
    result = contract({
        'candidateKeys': [{'keyId': 'pk', 'entityKind': 'record', 'primary': True,
                           'components': [{'componentId': 'id', 'fieldPath': 'id'}]}],
        'entityReferences': [{'referenceId': 'category', 'targetEntityKind': 'category', 'targetKeyId': 'code',
                              'components': [{'targetComponentId': 'code', 'localFieldPath': 'category_code'}]}],
    })
    assert result.candidate_keys[0].entity_kind == 'record'
    assert result.entity_references[0].components[0].local_field_path == 'category_code'


@pytest.mark.parametrize('field', ['missing', 'category.code'])
def test_declared_reference_cannot_claim_an_unreturned_field(field):
    with pytest.raises(ValueError, match='unknown field'):
        contract({'entityReferences': [{'referenceId': 'category', 'targetEntityKind': 'category', 'targetKeyId': 'code',
                                       'components': [{'targetComponentId': 'code', 'localFieldPath': field}]}]})


def test_malformed_relation_metadata_is_not_silently_ignored():
    with pytest.raises(ValueError):
        contract('not a metadata object')
