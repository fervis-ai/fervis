"""End-user explanation payloads."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lineage.views.explain import ExplainView
from fervis.lineage.views.input_lineage import input_lineage_view
from fervis.lineage.views.model import (
    InputLineageView,
    LineageTimelineView,
)
from fervis.lineage.views.timeline import compact_lineage_timeline_view


@dataclass(frozen=True)
class LineageExplanationView:
    compact: LineageTimelineView
    verbose: LineageTimelineView


@dataclass(frozen=True)
class AnswerExplanationView:
    inputs: InputLineageView
    lineage: LineageExplanationView


def answer_explanation_view(explain: ExplainView) -> AnswerExplanationView:
    inputs = input_lineage_view(explain.lineage)
    return AnswerExplanationView(
        inputs=inputs,
        lineage=LineageExplanationView(
            compact=compact_lineage_timeline_view(explain.timeline),
            verbose=explain.timeline,
        ),
    )
