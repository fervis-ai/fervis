from __future__ import annotations

from django.db import models
from rest_framework import serializers

from fervis.host_api.adapters.django.schema_introspection import (
    response_fields_from_serializer,
)


def test_serializer_method_field_uses_optional_return_annotation_for_response_type():
    class MethodTypeSerializer(serializers.Serializer):
        full_name = serializers.SerializerMethodField()

        def get_full_name(self, obj) -> str | None:
            return None

    fields = {
        field.name: field
        for field in response_fields_from_serializer(
            MethodTypeSerializer,
            model_context=None,
        )
    }

    assert fields["full_name"].type == "string"


def test_serializer_method_field_uses_dict_return_annotation_for_response_type():
    class MethodTypeSerializer(serializers.Serializer):
        payment = serializers.SerializerMethodField()

        def get_payment(self, obj) -> dict:
            return {}

    fields = {
        field.name: field
        for field in response_fields_from_serializer(
            MethodTypeSerializer,
            model_context=None,
        )
    }

    assert fields["payment"].type == "object"


def test_url_and_slug_fields_project_as_strings():
    class UrlSerializer(serializers.Serializer):
        url = serializers.URLField()
        slug = serializers.SlugField()

    fields = {
        field.name: field
        for field in response_fields_from_serializer(
            UrlSerializer,
            model_context=None,
        )
    }

    assert fields["url"].type == "string"
    assert fields["slug"].type == "string"


def test_plain_response_serializer_infers_fk_identity_from_model_context():
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        location = models.ForeignKey(Location, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection"

    class PlainSaleSerializer(serializers.Serializer):
        sale_id = serializers.UUIDField()
        location_id = serializers.UUIDField()

    fields = {
        field.name: field
        for field in response_fields_from_serializer(
            PlainSaleSerializer,
            model_context=Sale,
        )
    }

    assert fields["sale_id"].identity is not None
    assert fields["sale_id"].identity["entityRef"] == "sale"
    assert fields["sale_id"].identity["idField"] == "sale_id"

    assert fields["location_id"].identity is not None
    assert fields["location_id"].identity["entityRef"] == "location"
    assert fields["location_id"].identity["idField"] == "location_id"
