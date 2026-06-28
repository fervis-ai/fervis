from __future__ import annotations

from django.db import models
from django.db.models import Q


class RunWorkStatus(models.TextChoices):
    QUEUED = "QUEUED", "Queued"
    RUNNING = "RUNNING", "Running"
    COMPLETED = "COMPLETED", "Completed"
    FAILED = "FAILED", "Failed"


def default_read_context_ref() -> dict[str, str | None]:
    return {
        "scheme": "unmigrated",
        "key": None,
        "tenant_key": None,
    }


class RunWorkItem(models.Model):
    run_id = models.CharField(max_length=128, unique=True, db_index=True)
    conversation_id = models.CharField(max_length=128, db_index=True)
    tenant_id = models.CharField(max_length=128, db_index=True)
    user_id = models.CharField(max_length=128)
    status = models.CharField(
        max_length=32,
        choices=RunWorkStatus.choices,
        default=RunWorkStatus.QUEUED,
        db_index=True,
    )
    provider = models.CharField(max_length=64, null=True, blank=True)
    model_key = models.CharField(max_length=64, default="HAIKU")
    question = models.TextField()
    session_mode = models.CharField(max_length=32, default="continue")
    session_id = models.CharField(max_length=128, null=True, blank=True)
    approval_mode = models.CharField(max_length=32, default="auto_allow")
    approval_decision = models.CharField(max_length=128, null=True, blank=True)
    max_budget_usd = models.DecimalField(max_digits=8, decimal_places=4, default=0)
    max_thinking_tokens = models.PositiveIntegerField(default=64)
    conversation_context = models.JSONField(default=dict)
    runtime_context = models.JSONField(default=dict)
    read_context_ref = models.JSONField(default=default_read_context_ref)
    idempotency_key = models.CharField(max_length=255, null=True, blank=True)
    attempt_count = models.PositiveIntegerField(default=0)
    active_attempt = models.PositiveIntegerField(default=0)
    max_attempts = models.PositiveIntegerField(default=2)
    lease_owner = models.CharField(max_length=128, null=True, blank=True)
    lease_expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    next_attempt_at = models.DateTimeField(null=True, blank=True, db_index=True)
    last_error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "fervis_run_work_item"
        app_label = "fervis_jobs"
        ordering = ["created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["tenant_id", "conversation_id", "idempotency_key"],
                condition=Q(idempotency_key__isnull=False),
                name="fervis_work_idempotency_unique",
            ),
            models.UniqueConstraint(
                fields=["tenant_id", "conversation_id"],
                condition=Q(status__in=["QUEUED", "RUNNING"]),
                name="fervis_work_active_conversation_unique",
            ),
        ]
        indexes = [
            models.Index(
                fields=["status", "next_attempt_at", "created_at"],
                name="fervis_work_claim_idx",
            ),
            models.Index(
                fields=["status", "lease_expires_at"],
                name="fervis_work_lease_idx",
            ),
            models.Index(
                fields=["tenant_id", "conversation_id"],
                name="fervis_work_conv_idx",
            ),
        ]
