from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fervis_jobs", "0002_same_run_clarification_and_idempotency")
    ]

    operations = [
        migrations.AlterField(
            model_name="runworkitem",
            name="status",
            field=models.CharField(
                choices=[
                    ("QUEUED", "Queued"),
                    ("RUNNING", "Running"),
                    ("WAITING_FOR_CLARIFICATION", "Waiting for clarification"),
                    ("SUPERSEDED", "Superseded"),
                    ("COMPLETED", "Completed"),
                    ("FAILED", "Failed"),
                ],
                db_index=True,
                default="QUEUED",
                max_length=32,
            ),
        ),
    ]
