from __future__ import annotations

from django.db import models
import pytest
from rest_framework import generics, serializers

from fervis.host_api.adapters.django.schema_introspection import (
    entity_references_from_serializer,
    inspect_response_serializer,
    path_param_candidate_key_authority,
    path_param_entity_target,
    query_params_from_serializer,
    relation_keys_from_serializer,
    response_fields_from_serializer,
)


@pytest.mark.parametrize("key_field", [models.UUIDField, models.CharField])
def test_related_object_display_is_not_certified_as_its_primary_key(key_field):
    class Target(models.Model):
        target_id = key_field(primary_key=True)

        class Meta:
            app_label = f"test_display_reference_{key_field.__name__.lower()}"

    class Observation(models.Model):
        target = models.ForeignKey(Target, on_delete=models.CASCADE)

        class Meta:
            app_label = f"test_display_reference_{key_field.__name__.lower()}"

    class ObservationSerializer(serializers.ModelSerializer):
        label = serializers.CharField(source="target", read_only=True)
        explicit_key = serializers.ReadOnlyField(source="target.target_id")

        class Meta:
            model = Observation
            fields = ("target", "target_id", "explicit_key", "label")

    inspection = inspect_response_serializer(ObservationSerializer)

    assert {component.local_field_path for reference in inspection.entity_references
            for component in reference.components} == {
        "target", "target_id", "explicit_key",
    }
    assert next(field for field in inspection.response_fields if field.name == "label").type == "string"


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
    assert target.entity_kind == Staff._meta.label_lower
    assert target.key_id == "primary_key"
    assert target.component_id == "staff_id"
    authority = path_param_candidate_key_authority(
        ShiftRecord,
        param_name="staff_id",
    )
    assert authority is not None
    assert authority.entity_kind == Staff._meta.label_lower
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

    assert keys[0].entity_kind == Sale._meta.label_lower
    assert keys[0].components[0].field_path == "sale_id"
    assert references[0].target_entity_kind == Location._meta.label_lower
    assert references[0].components[0].local_field_path == "location_id"

    inspection = inspect_response_serializer(
        PlainSaleSerializer,
        model_context=Sale,
    )
    authority = inspection.candidate_key_authorities[0]
    assert authority.entity_kind == Location._meta.label_lower
    assert authority.key_id == "primary_key"
    assert authority.components[0].type == "uuid"


def test_nested_many_serializer_declares_its_own_row_identity() -> None:
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_nested_many_identity"

    class LocationRowSerializer(serializers.Serializer):
        location_id = serializers.UUIDField()

        class Meta:
            model = Location

    class ReportSerializer(serializers.Serializer):
        data = LocationRowSerializer(many=True)

    inspection = inspect_response_serializer(ReportSerializer)

    assert tuple(
        (
            key.entity_kind,
            key.key_id,
            tuple(component.field_path for component in key.components),
        )
        for key in inspection.candidate_keys
    ) == ((Location._meta.label_lower, "primary_key", ("data.location_id",)),)


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

    assert tuple(reference.target_entity_kind for reference in references) == (Staff._meta.label_lower,)


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
        shift_record_id = serializers.UUIDField(source="shift_record.shift_record_id")

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
            ShiftCompensation._meta.label_lower,
            "primary_key",
            ("shift_compensation_id",),
        ),
        (
            ShiftCompensation._meta.label_lower,
            "unique_compensation_closure",
            ("shift_record_id", "closure_version"),
        ),
    )


def test_nested_to_one_key_is_a_reference_not_a_second_relation_key():
    class Area(models.Model):
        area_id = models.UUIDField(primary_key=True)
        name = models.CharField(max_length=255)

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
            fields = ("area_id", "name")

    class StaffSerializer(serializers.ModelSerializer):
        default_area = AreaSerializer()

        class Meta:
            model = Staff
            fields = ("staff_id", "default_area")

    inspection = inspect_response_serializer(StaffSerializer)

    assert inspection.relation_model is Staff
    assert tuple(
        (key.entity_kind, key.key_id) for key in inspection.candidate_keys
    ) == ((Staff._meta.label_lower, "primary_key"),)
    assert tuple(
        (
            reference.target_entity_kind,
            reference.target_key_id,
            reference.components[0].local_field_path,
            reference.context_field_paths,
        )
        for reference in inspection.entity_references
    ) == (
        (
            Area._meta.label_lower,
            "primary_key",
            "default_area.area_id",
            ("default_area.name",),
        ),
    )


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

    assert references[0].target_entity_kind == Location._meta.label_lower
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
    assert params[0].entity_target.entity_kind == Location._meta.label_lower
    assert params[0].entity_target.key_id == "unique_code"
    assert params[0].entity_target.component_id == "code"


def test_scalar_query_parameter_targets_declared_foreign_key_filter() -> None:
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_query_filter"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        location = models.ForeignKey(Location, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection_query_filter"

    class QuerySerializer(serializers.Serializer):
        location_id = serializers.UUIDField(required=False)

    class LocationFilter:
        field_name = "location_id"
        lookup_expr = "exact"
        exclude = False
        method = None

    class SaleFilterSet:
        base_filters = {"location_id": LocationFilter()}

        class _meta:
            model = Sale

    class DeclaredFilterBackend:
        def get_filterset_class(self, view, queryset):
            return view.filterset_class

    class SaleListView(generics.ListAPIView):
        queryset = Sale.objects.all()
        filter_backends = (DeclaredFilterBackend,)
        filterset_class = SaleFilterSet

    params = query_params_from_serializer(
        QuerySerializer,
        model_context=Sale,
        view_class=SaleListView,
    )

    assert params[0].entity_target is not None
    assert params[0].entity_target.entity_kind == Location._meta.label_lower
    assert params[0].entity_target.key_id == "primary_key"
    assert params[0].entity_target.component_id == "location_id"


def test_negated_query_parameter_filter_has_no_entity_target() -> None:
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_negated_query_filter"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        location = models.ForeignKey(Location, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection_negated_query_filter"

    class QuerySerializer(serializers.Serializer):
        location_id = serializers.UUIDField(required=False)

    class SaleListView(generics.ListAPIView):
        queryset = Sale.objects.all()

        def get_queryset(self):
            queryset = super().get_queryset()
            location_id = self.request.query_params.get("location_id")
            return (
                queryset.exclude(location_id=location_id) if location_id else queryset
            )

    params = query_params_from_serializer(
        QuerySerializer,
        model_context=Sale,
        view_class=SaleListView,
    )

    assert params[0].entity_target is None


def test_disjunctive_query_parameter_filter_has_no_entity_target() -> None:
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_disjunctive_query_filter"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        location = models.ForeignKey(Location, on_delete=models.CASCADE)
        is_public = models.BooleanField(default=False)

        class Meta:
            app_label = "test_schema_introspection_disjunctive_query_filter"

    class QuerySerializer(serializers.Serializer):
        location_id = serializers.UUIDField(required=False)

    class SaleListView(generics.ListAPIView):
        queryset = Sale.objects.all()

        def get_queryset(self):
            queryset = super().get_queryset()
            location_id = self.request.query_params.get("location_id")
            return (
                queryset.filter(
                    models.Q(location_id=location_id) | models.Q(is_public=True)
                )
                if location_id
                else queryset
            )

    params = query_params_from_serializer(
        QuerySerializer,
        model_context=Sale,
        view_class=SaleListView,
    )

    assert params[0].entity_target is None


def test_scalar_query_parameter_without_proven_filter_has_no_entity_target() -> None:
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_schema_introspection_unconnected_query"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        location = models.ForeignKey(Location, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_schema_introspection_unconnected_query"

    class QuerySerializer(serializers.Serializer):
        location_id = serializers.UUIDField(required=False)

    class SaleListView(generics.ListAPIView):
        queryset = Sale.objects.all()

    params = query_params_from_serializer(
        QuerySerializer,
        model_context=Sale,
        view_class=SaleListView,
    )

    assert params[0].entity_target is None


def test_callable_query_default_remains_unknown_without_execution():
    def must_not_run():
        raise AssertionError('Discovery must not evaluate request defaults')

    class Query(serializers.Serializer):
        active = serializers.BooleanField(default=must_not_run)

    [param] = query_params_from_serializer(Query)
    assert param.default is None
    assert param.default_is_known is False


def test_optional_response_boolean_is_not_a_closed_two_value_domain():
    class Response(serializers.Serializer):
        active = serializers.BooleanField(required=False)

    assert Response({}).data == {}
    [field] = response_fields_from_serializer(Response, model_context=None)
    assert field.nullable is True


@pytest.mark.parametrize('nested', [False, True])
def test_omittable_boolean_cannot_certify_finite_population_coverage(nested):
    from dataclasses import replace
    from fervis.host_api.contracts import EndpointContract, ParameterContract
    from fervis.host_api.contracts.population import ParameterPopulation, ParameterRowValues
    from fervis.lookup.relation_catalog.from_host_api import relation_catalog_from_endpoint_contracts
    from fervis.lookup.relation_catalog.row_sources.builder import build_api_row_source_catalog
    from fervis.lookup.source_binding.population_values import population_values

    class Inner(serializers.Serializer):
        active = serializers.BooleanField(required=True)

    class Flat(serializers.Serializer):
        active = serializers.BooleanField(required=False)

    class Nested(serializers.Serializer):
        details = Inner(required=False)

    response = Nested if nested else Flat
    assert response({}).data == {}
    endpoint = EndpointContract('entries', 'entries', 'GET', '/entries', '', '',
        resource_names=('entries',), response_cardinality='many',
        query_params=(ParameterContract('active', 'boolean', default=True),),
        response_fields=response_fields_from_serializer(response, model_context=None))
    catalog = relation_catalog_from_endpoint_contracts((endpoint,))
    sources = build_api_row_source_catalog(catalog).sources
    source = next(s for s in sources if any(f.id.endswith('active') for f in s.fields))
    field = next(f for f in source.fields if f.id.endswith('active'))
    effect = ParameterPopulation(field_path=field.field_ref, value_mapping=(
        ParameterRowValues('false', ('false',)), ParameterRowValues('true', ('true',))))
    assert population_values(source, replace(source.params[0], population=effect)) == ()


def test_declared_query_defaults_preserve_catalog_types():
    from datetime import date
    from fervis.host_api.contracts import EndpointContract
    from fervis.lookup.relation_catalog.from_host_api import relation_catalog_from_endpoint_contracts

    class Query(serializers.Serializer):
        states = serializers.ListField(child=serializers.CharField(), default=['active'])
        since = serializers.DateField(default=date(2026, 1, 1))
        mode = serializers.ChoiceField(choices=[(1, 'First'), (2, 'Second')], default=1)

    params = query_params_from_serializer(Query)
    assert {p.name: p.default for p in params} == {
        'states': ['active'], 'since': '2026-01-01', 'mode': '1'}
    endpoint = EndpointContract('entries', 'entries', 'GET', '/entries', '', '', resource_names=('entries',), query_params=params)
    [read] = relation_catalog_from_endpoint_contracts((endpoint,)).reads
    assert {p.name: p.default for p in read.params} == {
        'states': ('active',), 'since': '2026-01-01', 'mode': '1'}


def test_readonly_model_property_uses_declared_type_without_executing_getter():
    class Observation(models.Model):
        @property
        def eligible(self) -> bool:
            raise AssertionError("Catalog discovery must not evaluate model properties")

        @property
        def undocumented(self):
            raise AssertionError("Catalog discovery must not sample model properties")

        class Meta:
            app_label = "test_declared_property_type"

    class ObservationSerializer(serializers.ModelSerializer):
        renamed = serializers.ReadOnlyField(source="eligible")

        class Meta:
            model = Observation
            fields = ("eligible", "renamed", "undocumented")

    inspection = inspect_response_serializer(ObservationSerializer)
    fields = {field.name: field for field in inspection.response_fields}
    assert fields["eligible"].type == "boolean"
    assert fields["renamed"].type == "boolean"
    assert fields["undocumented"].type == "any"


def test_readonly_property_typing_follows_declared_related_source_path():
    class Detail(models.Model):
        @property
        def quantity(self) -> int | None:
            raise AssertionError("Do not evaluate a related object's getter")

        class Meta:
            app_label = "test_related_property_type"

    class Record(models.Model):
        detail = models.ForeignKey(Detail, on_delete=models.CASCADE, null=True)

        class Meta:
            app_label = "test_related_property_type"

    class RecordSerializer(serializers.ModelSerializer):
        quantity = serializers.ReadOnlyField(source="detail.quantity")

        class Meta:
            model = Record
            fields = ("quantity",)

    [field] = inspect_response_serializer(RecordSerializer).response_fields
    assert field.type == "integer"
    assert field.nullable
