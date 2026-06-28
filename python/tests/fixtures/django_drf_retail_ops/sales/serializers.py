from rest_framework import serializers

from .models import Order, OrderItem


class OrderItemSerializer(serializers.ModelSerializer):
    product_id = serializers.IntegerField(source="product.id", read_only=True)
    product_sku = serializers.CharField(source="product.sku", read_only=True)
    product_name = serializers.CharField(source="product.name", read_only=True)

    class Meta:
        model = OrderItem
        fields = ["id", "product_id", "product_sku", "product_name", "quantity", "unit_price"]


class OrderSerializer(serializers.ModelSerializer):
    store_id = serializers.IntegerField(source="store.id", read_only=True)
    store_name = serializers.CharField(source="store.name", read_only=True)
    items = OrderItemSerializer(many=True, read_only=True)

    class Meta:
        model = Order
        fields = [
            "id",
            "order_number",
            "store_id",
            "store_name",
            "status",
            "ordered_at",
            "total_amount",
            "items",
        ]


class OrderQuerySerializer(serializers.Serializer):
    store_id = serializers.IntegerField(required=False)
    status = serializers.ChoiceField(required=False, choices=["open", "paid", "cancelled"])
    start_date = serializers.DateField(required=False)
    end_date = serializers.DateField(required=False)

