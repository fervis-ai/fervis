"""A route's parent key need not be the same-named key in its result rows."""
import pytest
from django.db import models
from django.urls.converters import UUIDConverter
from rest_framework import serializers
from rest_framework.views import APIView
from fervis.host_api.adapters.django.catalog import _build_contract


class GlobalBuyer(models.Model):
    buyer_id = models.UUIDField(primary_key=True)
    class Meta:
        app_label = 'path_target'


class LocalBuyer(models.Model):
    buyer_id = models.UUIDField(primary_key=True)
    class Meta:
        app_label = 'path_target'


class Sale(models.Model):
    sale_id = models.UUIDField(primary_key=True)
    buyer = models.ForeignKey(LocalBuyer, on_delete=models.CASCADE)
    class Meta:
        app_label = 'path_target'


class SaleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sale
        fields = ('sale_id', 'buyer_id')


def contract(fields):
    class View(APIView):
        serializer_class = SaleSerializer
        fervis_path_parameter_fields = fields
    return _build_contract(path='/buyers/<buyer_id>/sales/', url_name='sales', view_class=View,
                           converters={'buyer_id': UUIDConverter()})


def test_path_key_authority_is_independent_of_returned_foreign_key():
    result = contract({'buyer_id': GlobalBuyer._meta.get_field('buyer_id')})
    assert result.path_params[0].entity_target.entity_kind == GlobalBuyer._meta.label_lower
    assert result.entity_references[0].target_entity_kind == LocalBuyer._meta.label_lower
    assert {key.entity_kind for key in result.candidate_keys} == {Sale._meta.label_lower}
    assert GlobalBuyer._meta.label_lower in {key.entity_kind for key in result.candidate_key_authorities}


@pytest.mark.parametrize('fields', [{'unknown': GlobalBuyer._meta.pk}, {'buyer_id': 'buyer_id'}, ['buyer_id']])
def test_invalid_path_key_declarations_are_rejected(fields):
    with pytest.raises(ValueError, match='path parameter'):
        contract(fields)
