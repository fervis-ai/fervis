"""Question interface readers backed by Fervis SQL storage."""

from __future__ import annotations

from fervis.interfaces.common.questions import QuestionInterface
from fervis.interfaces.common.admission import ConfiguredModelPolicy

from .bundle import sql_storage_bundle


def sql_question_interface(*, project, loaded_config) -> QuestionInterface:
    bundle = sql_storage_bundle(project=project, loaded_config=loaded_config)
    return QuestionInterface(
        questions=bundle.questions,
        model_policy=ConfiguredModelPolicy.from_config(loaded_config.config.model),
        close_callback=bundle.close,
    )
