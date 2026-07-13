from __future__ import annotations

from django.db import models
from rest_framework import serializers

from fervis.host_api.adapters.django.schema_introspection import (
    entity_references_from_serializer,
    inspect_response_serializer,
    path_param_candidate_key_authority,
    path_param_entity_target,
    query_params_from_serializer,
    relation_keys_from_serializer,
    response_fields_from_serializer,
)


def test_foreign_key_path_param_targets_the_related_candidate_key() -> None:
    class Staff(models.Model):
        staff_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection"

    class ShiftRecord(models.Model):
        shift_record_id = models.UUIDField(primary_key=True)
        staff = models.ForeignKey(Staff, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection"

    target = path_param_entity_target(ShiftRecord, param_name="staff_id")

    assert target is not None
    assert target.entity_kind == "staff"
    assert target.key_id == "primary_key"
    assert target.component_id == "staff_id"
    authority = path_param_candidate_key_authority(
        ShiftRecord,
        param_name="staff_id",
    )
    assert authority is not None
    assert authority.entity_kind == "staff"
    assert authority.key_id == "primary_key"
    assert authority.components[0].component_id == "staff_id"


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


def test_plain_response_serializer_derives_keys_and_references_from_model_context():
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

    keys = relation_keys_from_serializer(
        PlainSaleSerializer,
        model_context=Sale,
    )
    references = entity_references_from_serializer(
        PlainSaleSerializer,
        model_context=Sale,
    )

    assert keys[0].entity_kind == "sale"
    assert keys[0].components[0].field_path == "sale_id"
    assert references[0].target_entity_kind == "location"
    assert references[0].components[0].local_field_path == "location_id"

    inspection = inspect_response_serializer(
        PlainSaleSerializer,
        model_context=Sale,
    )
    authority = inspection.candidate_key_authorities[0]
    assert authority.entity_kind == "location"
    assert authority.key_id == "primary_key"
    assert authority.components[0].type == "uuid"


def test_method_field_name_does_not_override_declared_relation_structure():
    class Staff(models.Model):
        staff_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_method_reference"

    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_method_reference"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        staff = models.ForeignKey(Staff, on_delete=models.CASCADE)
        location = models.ForeignKey(Location, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection_method_reference"

    class SaleSerializer(serializers.ModelSerializer):
        staff_id = serializers.UUIDField()
        location_id = serializers.SerializerMethodField()

        class Meta:
            model = Sale
            fields = ("sale_id", "staff_id", "location_id")

        def get_location_id(self, obj) -> str:
            return "computed-output"

    references = entity_references_from_serializer(SaleSerializer)

    assert tuple(reference.target_entity_kind for reference in references) == (
        "staff",
    )


def test_relation_keys_include_only_total_declared_uniqueness():
    class Campaign(models.Model):
        campaign_id = models.UUIDField(primary_key=True)
        external_code = models.CharField(max_length=32, unique=True)
        optional_code = models.CharField(max_length=32, unique=True, null=True)
        regional_code = models.CharField(max_length=32)
        active = models.BooleanField(default=True)

        class Meta:
            app_label = "test_schema_introspection_total_keys"
            constraints = (
                models.UniqueConstraint(
                    fields=("regional_code",),
                    condition=models.Q(active=True),
                    name="unique_active_regional_code",
                ),
            )

    class CampaignSerializer(serializers.Serializer):
        campaign_id = serializers.UUIDField()
        external_code = serializers.CharField()
        optional_code = serializers.CharField(allow_null=True)
        regional_code = serializers.CharField()
        active = serializers.BooleanField()

    keys = relation_keys_from_serializer(
        CampaignSerializer,
        model_context=Campaign,
    )

    assert tuple(key.key_id for key in keys) == (
        "primary_key",
        "unique_external_code",
    )


def test_relation_keys_require_a_scalar_representation_of_each_component():
    class Payment(models.Model):
        payment_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_scalar_keys"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        payment = models.OneToOneField(Payment, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection_scalar_keys"

    class SaleSerializer(serializers.ModelSerializer):
        payment = serializers.SerializerMethodField()

        class Meta:
            model = Sale
            fields = ("sale_id", "payment")

        def get_payment(self, obj) -> dict:
            return {"payment_id": str(obj.payment_id)}

    keys = relation_keys_from_serializer(SaleSerializer)

    assert tuple(key.key_id for key in keys) == ("primary_key",)


def test_flattened_related_key_remains_outside_the_owning_relation_key():
    class ShiftRecord(models.Model):
        shift_record_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_relation_grain"

    class ShiftCompensation(models.Model):
        shift_compensation_id = models.UUIDField(primary_key=True)
        shift_record = models.ForeignKey(ShiftRecord, on_delete=models.CASCADE)
        closure_version = models.PositiveIntegerField()

        class Meta:
            app_label = "test_schema_introspection_relation_grain"
            constraints = (
                models.UniqueConstraint(
                    fields=("shift_record", "closure_version"),
                    name="unique_compensation_closure",
                ),
            )

    class CompensationSerializer(serializers.ModelSerializer):
        shift_record_id = serializers.UUIDField(
            source="shift_record.shift_record_id"
        )

        class Meta:
            model = ShiftCompensation
            fields = (
                "shift_compensation_id",
                "shift_record_id",
                "closure_version",
            )

    keys = relation_keys_from_serializer(CompensationSerializer)

    assert tuple(
        (key.entity_kind, key.key_id, tuple(c.field_path for c in key.components))
        for key in keys
    ) == (
        (
            "shift_compensation",
            "primary_key",
            ("shift_compensation_id",),
        ),
        (
            "shift_compensation",
            "unique_compensation_closure",
            ("shift_record_id", "closure_version"),
        ),
    )


def test_nested_to_one_key_is_a_reference_not_a_second_relation_key():
    class Area(models.Model):
        area_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_nested_to_one"

    class Staff(models.Model):
        staff_id = models.UUIDField(primary_key=True)
        default_area = models.ForeignKey(Area, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection_nested_to_one"

    class AreaSerializer(serializers.ModelSerializer):
        class Meta:
            model = Area
            fields = ("area_id",)

    class StaffSerializer(serializers.ModelSerializer):
        default_area = AreaSerializer()

        class Meta:
            model = Staff
            fields = ("staff_id", "default_area")

    inspection = inspect_response_serializer(StaffSerializer)

    assert inspection.relation_model is Staff
    assert tuple(
        (key.entity_kind, key.key_id) for key in inspection.candidate_keys
    ) == (("staff", "primary_key"),)
    assert tuple(
        (
            reference.target_entity_kind,
            reference.target_key_id,
            reference.components[0].local_field_path,
        )
        for reference in inspection.entity_references
    ) == (("area", "primary_key", "default_area.area_id"),)


def test_foreign_key_reference_targets_the_declared_unique_key():
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)
        code = models.CharField(max_length=32, unique=True)

        class Meta:
            app_label = "test_schema_introspection_unique"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        location = models.ForeignKey(
            Location,
            to_field="code",
            on_delete=models.CASCADE,
        )

        class Meta:
            app_label = "test_schema_introspection_unique"

    class PlainSaleSerializer(serializers.Serializer):
        sale_id = serializers.UUIDField()
        location_code = serializers.CharField(source="location_id")

    references = entity_references_from_serializer(
        PlainSaleSerializer,
        model_context=Sale,
    )

    assert references[0].target_entity_kind == "location"
    assert references[0].target_key_id == "unique_code"
    assert references[0].components[0].target_component_id == "code"
    assert references[0].components[0].local_field_path == "location_code"


def test_slug_related_query_parameter_targets_the_declared_unique_key():
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)
        code = models.CharField(max_length=32, unique=True)

        class Meta:
            app_label = "test_schema_introspection_query_unique"

    class QuerySerializer(serializers.Serializer):
        location = serializers.SlugRelatedField(
            slug_field="code",
            queryset=Location.objects.all(),
        )

    params = query_params_from_serializer(QuerySerializer)

    assert params[0].entity_target is not None
    assert params[0].entity_target.entity_kind == "location"
    assert params[0].entity_target.key_id == "unique_code"
    assert params[0].entity_target.component_id == "code"
