from django.db import models

from tests.fixtures.django_drf_retail_ops.catalog.models import Product
from tests.fixtures.django_drf_retail_ops.inventory.models import InventoryLocation


class StockTransfer(models.Model):
    STATUS_CHOICES = (
        ("draft", "Draft"),
        ("in_transit", "In Transit"),
        ("received", "Received"),
    )

    reference = models.CharField(max_length=32, unique=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    from_location = models.ForeignKey(
        InventoryLocation,
        related_name="outgoing_transfers",
        on_delete=models.CASCADE,
    )
    to_location = models.ForeignKey(
        InventoryLocation,
        related_name="incoming_transfers",
        on_delete=models.CASCADE,
    )
    quantity = models.IntegerField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    requested_at = models.DateField()

    class Meta:
        app_label = "retail_ops_fulfillment"
        ordering = ["reference"]
