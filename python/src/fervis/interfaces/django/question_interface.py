"""Django composition for the shared Fervis question interface."""

from __future__ import annotations

from fervis.interfaces.common.questions import QuestionInterface
from fervis.interfaces.common.admission import ConfiguredModelPolicy
from fervis.lineage.views.django import DjangoLineageQuery
from fervis.observability.django import DjangoObservabilityQuery
from fervis.model_io.providers.openai_answer_question import (
    configured_answer_question_service,
)
from fervis.project.source_scope import configured_fervis_config

from .composition import question_run_request_limits
from .question_run_ports import django_question_service


def django_question_interface() -> QuestionInterface:
    config = configured_fervis_config()
    lineage_query = DjangoLineageQuery()
    return QuestionInterface(
        questions=django_question_service(),
        answer_questions=configured_answer_question_service(
            lineage_query,
            DjangoObservabilityQuery(),
        ),
        limits=question_run_request_limits(),
        model_policy=ConfiguredModelPolicy.from_config(config.model),
    )
