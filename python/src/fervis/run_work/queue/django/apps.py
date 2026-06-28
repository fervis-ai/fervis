from django.apps import AppConfig


class QuestionRunWorkDjangoConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "fervis.run_work.queue.django"
    label = "fervis_jobs"
