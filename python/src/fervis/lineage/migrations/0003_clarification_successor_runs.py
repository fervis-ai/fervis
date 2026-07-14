from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("fervis_lineage", "0002_same_run_clarification")]

    operations = [
        migrations.AddField(
            model_name="questionrun",
            name="trigger_clarification_response_id",
            field=models.CharField(blank=True, default="", max_length=128),
        ),
        migrations.AlterField(
            model_name="questionrun",
            name="trigger_kind",
            field=models.CharField(
                choices=[
                    ("initial", "initial"),
                    ("clarification_response", "clarification_response"),
                    ("retry", "retry"),
                    ("rerun", "rerun"),
                ],
                default="initial",
                max_length=32,
            ),
        ),
    ]
