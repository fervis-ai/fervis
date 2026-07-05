from __future__ import annotations

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
        for field in response_fields_from_serializer(MethodTypeSerializer)
    }

    assert fields["full_name"].type == "string"


def test_serializer_method_field_uses_dict_return_annotation_for_response_type():
    class MethodTypeSerializer(serializers.Serializer):
        payment = serializers.SerializerMethodField()

        def get_payment(self, obj) -> dict:
            return {}

    fields = {
        field.name: field
        for field in response_fields_from_serializer(MethodTypeSerializer)
    }

    assert fields["payment"].type == "object"


def test_url_and_slug_fields_project_as_strings():
    class UrlSerializer(serializers.Serializer):
        url = serializers.URLField()
        slug = serializers.SlugField()

    fields = {
        field.name: field for field in response_fields_from_serializer(UrlSerializer)
    }

    assert fields["url"].type == "string"
    assert fields["slug"].type == "string"
