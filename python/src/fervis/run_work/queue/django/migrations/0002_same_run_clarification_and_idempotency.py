import hashlib
import json

from django.db import migrations, models
from django.db.models import Q


def populate_idempotency_authority(apps, schema_editor):
    work_item = apps.get_model("fervis_jobs", "RunWorkItem")
    for item in work_item.objects.all().iterator():
        payload = json.dumps(
            {
                "tenant_id": item.tenant_id,
                "principal_id": item.user_id,
                "read_context_ref": item.read_context_ref,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        item.idempotency_authority_ref = (
            "idempotency-authority:sha256:"
            + hashlib.sha256(payload.encode()).hexdigest()
        )
        item.idempotency_scope = f"conversation:{item.conversation_id}"
        item.save(
            update_fields=("idempotency_authority_ref", "idempotency_scope")
        )


class Migration(migrations.Migration):
    dependencies = [("fervis_jobs", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="runworkitem",
            name="idempotency_authority_ref",
            field=models.CharField(blank=True, default="", max_length=96),
        ),
        migrations.AddField(
            model_name="runworkitem",
            name="idempotency_scope",
            field=models.CharField(blank=True, default="", max_length=160),
        ),
        migrations.RunPython(populate_idempotency_authority, migrations.RunPython.noop),
        migrations.RemoveConstraint(
            model_name="runworkitem",
            name="fervis_work_idempotency_unique",
        ),
        migrations.RemoveConstraint(
            model_name="runworkitem",
            name="fervis_work_active_conversation_unique",
        ),
        migrations.AlterField(
            model_name="runworkitem",
            name="status",
            field=models.CharField(
                choices=[
                    ("QUEUED", "Queued"),
                    ("RUNNING", "Running"),
                    ("WAITING_FOR_CLARIFICATION", "Waiting for clarification"),
                    ("COMPLETED", "Completed"),
                    ("FAILED", "Failed"),
                ],
                db_index=True,
                default="QUEUED",
                max_length=32,
            ),
        ),
        migrations.AddConstraint(
            model_name="runworkitem",
            constraint=models.UniqueConstraint(
                condition=Q(idempotency_key__isnull=False),
                fields=(
                    "idempotency_authority_ref",
                    "idempotency_scope",
                    "idempotency_key",
                ),
                name="fervis_work_idempotency_unique",
            ),
        ),
        migrations.AddConstraint(
            model_name="runworkitem",
            constraint=models.UniqueConstraint(
                condition=Q(
                    status__in=[
                        "QUEUED",
                        "RUNNING",
                        "WAITING_FOR_CLARIFICATION",
                    ]
                ),
                fields=("tenant_id", "conversation_id"),
                name="fervis_work_active_conversation_unique",
            ),
        ),
    ]
