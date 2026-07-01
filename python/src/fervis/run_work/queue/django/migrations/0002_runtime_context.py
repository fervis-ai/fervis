from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fervis_jobs", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="runworkitem",
            name="runtime_context",
            field=models.JSONField(default=dict),
        ),
    ]
