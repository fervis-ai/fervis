from __future__ import annotations

from django.db import models
from rest_framework import generics, serializers

from fervis.host_api.adapters.django.catalog import _build_contract


def test_catalog_uses_declared_filter_metadata_without_calling_get_queryset() -> None:
    class Location(models.Model):
        location_id = models.UUIDField(primary_key=True)

        class Meta:
            app_label = "test_catalog_declared_filter"

    class Sale(models.Model):
        sale_id = models.UUIDField(primary_key=True)
        location = models.ForeignKey(Location, on_delete=models.CASCADE)

        class Meta:
            app_label = "test_catalog_declared_filter"

    class QuerySerializer(serializers.Serializer):
        location_id = serializers.UUIDField(required=False)

    class SaleSerializer(serializers.ModelSerializer):
        class Meta:
            model = Sale
            fields = ("sale_id", "location_id")

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
            assert queryset.model is Sale
            return view.filterset_class

    class SaleListView(generics.ListAPIView):
        query_serializer_class = QuerySerializer
        serializer_class = SaleSerializer
        filter_backends = (DeclaredFilterBackend,)
        filterset_class = SaleFilterSet
        get_queryset_calls = 0

        def get_queryset(self):
            type(self).get_queryset_calls += 1
            return Sale.objects.filter(organization=self.request.user.organization)

    contract = _build_contract(
        path="sales/",
        url_name="sale-list",
        view_class=SaleListView,
        converters={},
    )

    location_param = next(
        param for param in contract.query_params if param.name == "location_id"
    )
    assert SaleListView.get_queryset_calls == 0
    assert location_param.entity_target is not None
    assert location_param.entity_target.entity_kind == "location"
    assert location_param.entity_target.key_id == "primary_key"
    assert location_param.entity_target.component_id == "location_id"
