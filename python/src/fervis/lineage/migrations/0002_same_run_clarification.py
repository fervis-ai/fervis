from django.db import migrations, models
import django.db.models.deletion


def attach_clarifications_to_owning_steps(apps, schema_editor):
    request = apps.get_model("fervis_lineage", "ClarificationRequest")
    pending = request.objects.filter(step__isnull=True).select_related("fact_result")
    for clarification in pending:
        fact_result = clarification.fact_result
        if fact_result is None:
            raise RuntimeError(
                "cannot migrate clarification without owning step lineage"
            )
        clarification.step_id = fact_result.produced_by_step_id
        clarification.save(update_fields=("step",))


class Migration(migrations.Migration):
    dependencies = [("fervis_lineage", "0001_initial")]

    operations = [
        migrations.RunPython(
            attach_clarifications_to_owning_steps,
            migrations.RunPython.noop,
        ),
        migrations.AlterField(
            model_name="factresult",
            name="result_kind",
            field=models.CharField(
                choices=[
                    ("answered", "answered"),
                    ("impossible", "impossible"),
                    ("no_data", "no_data"),
                    ("undefined", "undefined"),
                ],
                max_length=64,
            ),
        ),
        migrations.RemoveField(
            model_name="clarificationrequest",
            name="fact_result",
        ),
        migrations.RemoveField(
            model_name="questionrun",
            name="trigger_clarification_response_id",
        ),
        migrations.AlterField(
            model_name="questionrun",
            name="trigger_kind",
            field=models.CharField(
                choices=[
                    ("initial", "initial"),
                    ("retry", "retry"),
                    ("rerun", "rerun"),
                ],
                default="initial",
                max_length=32,
            ),
        ),
        migrations.AlterField(
            model_name="clarificationrequest",
            name="step",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.PROTECT,
                related_name="clarification_requests",
                to="fervis_lineage.runstep",
            ),
        ),
    ]
