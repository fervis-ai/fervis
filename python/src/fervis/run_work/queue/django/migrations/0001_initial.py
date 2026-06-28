# Generated manually for fervis durable run work items.

from django.db import migrations, models
from django.db.models import Q


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="RunWorkItem",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "run_id",
                    models.CharField(db_index=True, max_length=128, unique=True),
                ),
                ("conversation_id", models.CharField(db_index=True, max_length=128)),
                ("tenant_id", models.CharField(db_index=True, max_length=128)),
                ("user_id", models.CharField(max_length=128)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("QUEUED", "Queued"),
                            ("RUNNING", "Running"),
                            ("COMPLETED", "Completed"),
                            ("FAILED", "Failed"),
                        ],
                        db_index=True,
                        default="QUEUED",
                        max_length=32,
                    ),
                ),
                ("provider", models.CharField(blank=True, max_length=64, null=True)),
                ("model_key", models.CharField(default="HAIKU", max_length=64)),
                ("question", models.TextField()),
                ("session_mode", models.CharField(default="continue", max_length=32)),
                ("session_id", models.CharField(blank=True, max_length=128, null=True)),
                (
                    "approval_mode",
                    models.CharField(default="auto_allow", max_length=32),
                ),
                (
                    "approval_decision",
                    models.CharField(blank=True, max_length=128, null=True),
                ),
                ("max_turns", models.PositiveIntegerField(default=6)),
                (
                    "max_budget_usd",
                    models.DecimalField(decimal_places=4, default=0, max_digits=8),
                ),
                ("max_thinking_tokens", models.PositiveIntegerField(default=64)),
                ("conversation_context", models.JSONField(default=dict)),
                (
                    "idempotency_key",
                    models.CharField(blank=True, max_length=255, null=True),
                ),
                ("attempt_count", models.PositiveIntegerField(default=0)),
                ("active_attempt", models.PositiveIntegerField(default=0)),
                ("max_attempts", models.PositiveIntegerField(default=2)),
                (
                    "lease_owner",
                    models.CharField(blank=True, max_length=128, null=True),
                ),
                (
                    "lease_expires_at",
                    models.DateTimeField(blank=True, db_index=True, null=True),
                ),
                (
                    "next_attempt_at",
                    models.DateTimeField(blank=True, db_index=True, null=True),
                ),
                ("last_error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "db_table": "fervis_run_work_item",
                "ordering": ["created_at"],
            },
        ),
        migrations.AddConstraint(
            model_name="fervisrunworkitem",
            constraint=models.UniqueConstraint(
                condition=Q(("idempotency_key__isnull", False)),
                fields=("tenant_id", "conversation_id", "idempotency_key"),
                name="fervis_work_idempotency_unique",
            ),
        ),
        migrations.AddConstraint(
            model_name="fervisrunworkitem",
            constraint=models.UniqueConstraint(
                condition=Q(("status__in", ["QUEUED", "RUNNING"])),
                fields=("tenant_id", "conversation_id"),
                name="fervis_work_active_conversation_unique",
            ),
        ),
        migrations.AddIndex(
            model_name="fervisrunworkitem",
            index=models.Index(
                fields=["status", "next_attempt_at", "created_at"],
                name="fervis_work_claim_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="fervisrunworkitem",
            index=models.Index(
                fields=["status", "lease_expires_at"], name="fervis_work_lease_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="fervisrunworkitem",
            index=models.Index(
                fields=["tenant_id", "conversation_id"], name="fervis_work_conv_idx"
            ),
        ),
    ]
