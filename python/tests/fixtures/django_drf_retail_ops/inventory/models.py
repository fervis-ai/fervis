from django.db import models

from tests.fixtures.django_drf_retail_ops.catalog.models import Product


class Store(models.Model):
    name = models.CharField(max_length=80)
    region = models.CharField(max_length=80)

    class Meta:
        app_label = "retail_ops_inventory"
        ordering = ["name"]


class InventoryLocation(models.Model):
    KIND_CHOICES = (("store", "Store"), ("warehouse", "Warehouse"))

    name = models.CharField(max_length=80)
    kind = models.CharField(max_length=20, choices=KIND_CHOICES)
    store = models.ForeignKey(Store, null=True, blank=True, on_delete=models.CASCADE)

    class Meta:
        app_label = "retail_ops_inventory"
        ordering = ["name"]


class StockRecord(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    location = models.ForeignKey(InventoryLocation, on_delete=models.CASCADE)
    quantity_on_hand = models.IntegerField()
    reorder_point = models.IntegerField(default=0)

    class Meta:
        app_label = "retail_ops_inventory"
        ordering = ["product__sku", "location__name"]
        unique_together = [("product", "location")]

