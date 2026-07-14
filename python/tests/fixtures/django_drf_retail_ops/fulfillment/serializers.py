from rest_framework import serializers

from .models import StockTransfer


class StockTransferSerializer(serializers.ModelSerializer):
    product_id = serializers.IntegerField(source="product.id", read_only=True)
    product_sku = serializers.CharField(source="product.sku", read_only=True)
    from_location_id = serializers.IntegerField(
        source="from_location.id", read_only=True
    )
    from_location_name = serializers.CharField(
        source="from_location.name", read_only=True
    )
    to_location_id = serializers.IntegerField(source="to_location.id", read_only=True)
    to_location_name = serializers.CharField(source="to_location.name", read_only=True)

    class Meta:
        model = StockTransfer
        fields = [
            "id",
            "reference",
            "product_id",
            "product_sku",
            "from_location_id",
            "from_location_name",
            "to_location_id",
            "to_location_name",
            "quantity",
            "status",
            "requested_at",
        ]


class StockTransferQuerySerializer(serializers.Serializer):
    status = serializers.ChoiceField(
        required=False, choices=["draft", "in_transit", "received"]
    )
    product_id = serializers.IntegerField(required=False)
    to_location_id = serializers.IntegerField(required=False)
