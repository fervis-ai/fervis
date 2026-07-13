from rest_framework import serializers


class SalesSummaryQuerySerializer(serializers.Serializer):
    start_date = serializers.DateField(required=False)
    end_date = serializers.DateField(required=False)
    store_id = serializers.IntegerField(required=False)
    group_by = serializers.ChoiceField(
        required=False, choices=["store", "day", "status"]
    )


class SalesSummaryRowSerializer(serializers.Serializer):
    label = serializers.CharField()
    total_orders = serializers.IntegerField()
    total_amount = serializers.DecimalField(max_digits=12, decimal_places=2)


class LowStockQuerySerializer(serializers.Serializer):
    location_id = serializers.IntegerField(required=False)
    category_id = serializers.IntegerField(required=False)


class LowStockRowSerializer(serializers.Serializer):
    product_id = serializers.IntegerField()
    product_sku = serializers.CharField()
    location_id = serializers.IntegerField()
    location_name = serializers.CharField()
    quantity_on_hand = serializers.IntegerField()
    reorder_point = serializers.IntegerField()
