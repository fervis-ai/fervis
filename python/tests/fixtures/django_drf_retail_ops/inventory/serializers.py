from rest_framework import serializers

from .models import InventoryLocation, StockRecord, Store


class StoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Store
        fields = ["id", "name", "region"]


class InventoryLocationSerializer(serializers.ModelSerializer):
    store_id = serializers.IntegerField(source="store.id", read_only=True)
    store_name = serializers.CharField(source="store.name", read_only=True)

    class Meta:
        model = InventoryLocation
        fields = ["id", "name", "kind", "store_id", "store_name"]


class StockRecordSerializer(serializers.ModelSerializer):
    product_id = serializers.IntegerField(source="product.id", read_only=True)
    product_sku = serializers.CharField(source="product.sku", read_only=True)
    product_name = serializers.CharField(source="product.name", read_only=True)
    location_id = serializers.IntegerField(source="location.id", read_only=True)
    location_name = serializers.CharField(source="location.name", read_only=True)

    class Meta:
        model = StockRecord
        fields = [
            "id",
            "product_id",
            "product_sku",
            "product_name",
            "location_id",
            "location_name",
            "quantity_on_hand",
            "reorder_point",
        ]


class StockRecordQuerySerializer(serializers.Serializer):
    product_id = serializers.IntegerField(required=False)
    location_id = serializers.IntegerField(required=False)
    below_reorder_point = serializers.BooleanField(required=False)
