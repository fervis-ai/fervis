from rest_framework import serializers

from .models import Category, Product


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ["id", "name"]


class ProductSerializer(serializers.ModelSerializer):
    category_id = serializers.IntegerField(source="category.id", read_only=True)
    category_name = serializers.CharField(source="category.name", read_only=True)

    def __init__(self, *args, **kwargs):
        fields = kwargs.pop("fields", None)
        super().__init__(*args, **kwargs)
        if fields:
            selected = set(fields)
            for field_name in tuple(self.fields):
                if field_name not in selected:
                    self.fields.pop(field_name)

    class Meta:
        model = Product
        fields = [
            "id",
            "sku",
            "name",
            "category_id",
            "category_name",
            "unit_price",
            "active",
        ]


class ProductQuerySerializer(serializers.Serializer):
    fields = serializers.CharField(required=False)
    category_id = serializers.IntegerField(required=False)
    active = serializers.BooleanField(required=False)
    ordering = serializers.ChoiceField(
        required=False,
        choices=["sku", "-sku", "name", "-name", "unit_price", "-unit_price"],
    )
