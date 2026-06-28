from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("fervis_jobs", "0002_runtime_context"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="fervisrunworkitem",
            name="max_turns",
        ),
    ]
