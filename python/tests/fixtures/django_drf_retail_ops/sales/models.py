from django.db import models

from tests.fixtures.django_drf_retail_ops.catalog.models import Product
from tests.fixtures.django_drf_retail_ops.inventory.models import Store


class Order(models.Model):
    STATUS_CHOICES = (
        ("open", "Open"),
        ("paid", "Paid"),
        ("cancelled", "Cancelled"),
    )

    order_number = models.CharField(max_length=32, unique=True)
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    ordered_at = models.DateField()
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        app_label = "retail_ops_sales"
        ordering = ["order_number"]


class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name="items", on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.IntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        app_label = "retail_ops_sales"
        ordering = ["order__order_number", "product__sku"]

