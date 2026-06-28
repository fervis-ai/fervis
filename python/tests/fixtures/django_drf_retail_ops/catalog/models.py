from django.db import models


class Category(models.Model):
    name = models.CharField(max_length=80)

    class Meta:
        app_label = "retail_ops_catalog"
        ordering = ["name"]


class Product(models.Model):
    sku = models.CharField(max_length=32, unique=True)
    name = models.CharField(max_length=120)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    active = models.BooleanField(default=True)

    class Meta:
        app_label = "retail_ops_catalog"
        ordering = ["sku"]

