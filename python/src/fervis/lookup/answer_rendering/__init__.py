"""Deterministic answer rendering from verified fact results."""

from fervis.lookup.answer_rendering.model import RenderedFact
from fervis.lookup.answer_rendering.renderer import (
    render_fact_result,
    rendered_fact_payload,
    rendered_fact_text,
)

__all__ = [
    "RenderedFact",
    "render_fact_result",
    "rendered_fact_text",
    "rendered_fact_payload",
]
