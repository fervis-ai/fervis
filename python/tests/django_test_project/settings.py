from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SECRET_KEY = "fervis-test-secret"
DEBUG = True
USE_TZ = True
TIME_ZONE = "UTC"
ROOT_URLCONF = "tests.django_test_project.urls"
FERVIS_CONFIG_PATH = "config/fervis.json"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "rest_framework",
    "tests.fixtures.django_drf_retail_ops.catalog.apps.RetailCatalogConfig",
    "tests.fixtures.django_drf_retail_ops.inventory.apps.RetailInventoryConfig",
    "tests.fixtures.django_drf_retail_ops.sales.apps.RetailSalesConfig",
    "tests.fixtures.django_drf_retail_ops.fulfillment.apps.RetailFulfillmentConfig",
    "tests.fixtures.django_drf_retail_ops.reports.apps.RetailReportsConfig",
    "fervis.lineage.apps.FervisLineageConfig",
    "fervis.run_work.queue.django.apps.QuestionRunWorkDjangoConfig",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

MIDDLEWARE: list[str] = []
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

MIGRATION_MODULES = {
    "fervis_lineage": None,
    "fervis_jobs": None,
    "retail_ops_catalog": None,
    "retail_ops_inventory": None,
    "retail_ops_sales": None,
    "retail_ops_fulfillment": None,
    "retail_ops_reports": None,
}

REST_FRAMEWORK: dict[str, str | list[str] | dict[str, str] | None] = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_THROTTLE_RATES": {"fervis_question": "1000/min"},
    "UNAUTHENTICATED_USER": None,
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}
