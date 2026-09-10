"""Model-turn purpose constants."""

from __future__ import annotations
from fervis.types.enums import StrEnum


class ModelTurnPurpose(StrEnum):
    CONVERSATION_RESOLUTION = "conversation_resolution"
    QUESTION_CONTRACT = "question_contract"
    QUERY_ENRICHMENT = "query_enrichment"
    GROUNDING = "grounding"
    READ_ELIGIBILITY = "read_eligibility"
    PLAN_SELECTION = "plan_selection"
    SOURCE_ACCESS = "source_access"
    SOURCE_POPULATION = "source_population"
    SOURCE_REALIZATION = "source_realization"
    SOURCE_BINDING = "source_binding"
    ANSWER_SYNTHESIS = "answer_synthesis"
