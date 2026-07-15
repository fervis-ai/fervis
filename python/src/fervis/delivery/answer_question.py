"""Disposable spoken answers about a persisted answer computation."""

from __future__ import annotations

import io
import json
import wave
from dataclasses import dataclass
from typing import Protocol
from xml.etree import ElementTree as ET

from fervis.lineage.views.explain import ExplainViewService
from fervis.lineage.views.explanation import (
    AnswerExplanationView,
    answer_explanation_view,
)
from fervis.lineage.views.json_payload import view_json
from fervis.lineage.views.query import LineageQueryPort
from fervis.observability.query import ObservabilityQueryPort


ANSWER_QUESTION_INSTRUCTIONS = """
About Fervis:
Fervis is an AI-based runtime for answering factual questions. Fervis has just answered the latest question in a conversation.

The listener has already seen that answer and can see the conversation history. They are now asking how Fervis interpreted the latest question or computed its answer.

Fervis pipeline:
Conversation Resolution completes contextual follow-ups using prior turns.
Question Contract defines the requested facts and required inputs.
Query Enrichment connects concepts in the question to candidate backend resources.
Grounding resolves inputs such as dates and identities.
Read Eligibility retains sources that can validly contribute to the answer.
Plan Selection chooses the source strategy.
Source Binding determines the source parameters and constraints required to satisfy the request.
Fact Planning defines the deterministic operations used to derive the requested facts.
Execution reads evidence and computes outputs.

Your task:
Answer the listener’s recorded question about the latest displayed answer.

Use only the supplied answerComputation XML. If it does not establish something required to answer the question, state precisely what is unavailable. Never guess.

Use the fewest words that fully answer the question, and never exceed 60 words. Begin with the direct answer. Do not add a greeting, introduction, question restatement, or generic pipeline description.

Use only the evidence relevant to the listener’s question. This may include resolved inputs; accepted or rejected choices and their recorded reasons; source qualification; executed filters; returned evidence; the calculation method; and material limitations.

Do not read IDs, hashes, XML, or headings aloud. Use a sober, trustworthy, natural conversational tone. Speak dates, numbers, and other values naturally.
"""  # noqa: E501

MAX_QUESTION_BYTES = 1_500_000
MAX_QUESTION_SECONDS = 30
MIN_QUESTION_MILLISECONDS = 180
INPUT_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class RecordedAnswerQuestion:
    pcm: bytes
    duration_ms: int
    sample_rate: int = INPUT_SAMPLE_RATE


@dataclass(frozen=True)
class GeneratedAnswerAudio:
    data: bytes
    content_type: str = "audio/wav"


class InvalidRecordedAnswerQuestion(ValueError):
    """Raised when an uploaded microphone recording is outside the contract."""


class AnswerQuestionGenerationError(RuntimeError):
    """Raised when the disposable provider response cannot produce valid audio."""

    def __init__(
        self,
        message: str,
        *,
        public_message: str = "The answer explanation could not be generated.",
    ) -> None:
        super().__init__(message)
        self.public_message = public_message


class AnswerQuestionGenerator(Protocol):
    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
        question: RecordedAnswerQuestion,
    ) -> GeneratedAnswerAudio: ...


class AnswerExplanationLoader(Protocol):
    def for_run(self, run_id: str) -> AnswerExplanationView: ...


class AnswerQuestionDelivery(Protocol):
    def answer(
        self, run_id: str, question: RecordedAnswerQuestion
    ) -> GeneratedAnswerAudio: ...


class LineageAnswerExplanationLoader:
    """Load the same typed explanation projection used by question-run views."""

    def __init__(
        self,
        *,
        lineage_query: LineageQueryPort,
        observability_query: ObservabilityQueryPort,
    ) -> None:
        self._explain = ExplainViewService(
            lineage_query=lineage_query,
            observability_query=observability_query,
        )

    def for_run(self, run_id: str) -> AnswerExplanationView:
        return answer_explanation_view(self._explain.for_run(run_id))


class AnswerQuestionService:
    def __init__(
        self,
        *,
        explanations: AnswerExplanationLoader,
        generator: AnswerQuestionGenerator,
    ) -> None:
        self._explanations = explanations
        self._generator = generator

    def answer(
        self, run_id: str, question: RecordedAnswerQuestion
    ) -> GeneratedAnswerAudio:
        explanation = self._explanations.for_run(run_id)
        return self._generator.generate(
            instructions=ANSWER_QUESTION_INSTRUCTIONS,
            input_text=answer_computation_xml_v1(explanation),
            question=question,
        )


def parse_recorded_answer_question(
    data: bytes,
    *,
    content_type: str,
) -> RecordedAnswerQuestion:
    if content_type.partition(";")[0].strip().lower() not in {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
    }:
        raise InvalidRecordedAnswerQuestion("Use a WAV microphone recording.")
    if not data:
        raise InvalidRecordedAnswerQuestion("Record a question before releasing.")
    if len(data) > MAX_QUESTION_BYTES:
        raise InvalidRecordedAnswerQuestion("The recording is too large.")
    try:
        with wave.open(io.BytesIO(data), "rb") as recorded:
            channels = recorded.getnchannels()
            sample_width = recorded.getsampwidth()
            sample_rate = recorded.getframerate()
            frame_count = recorded.getnframes()
            pcm = recorded.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise InvalidRecordedAnswerQuestion(
            "The recording is not valid WAV audio."
        ) from exc
    if channels != 1 or sample_width != 2 or sample_rate != INPUT_SAMPLE_RATE:
        raise InvalidRecordedAnswerQuestion("Record mono PCM16 audio at 24 kHz.")
    expected_pcm_bytes = frame_count * channels * sample_width
    if len(pcm) != expected_pcm_bytes:
        raise InvalidRecordedAnswerQuestion("The recording is truncated.")
    duration_ms = len(pcm) * 1000 // (sample_rate * channels * sample_width)
    if duration_ms < MIN_QUESTION_MILLISECONDS:
        raise InvalidRecordedAnswerQuestion(
            "Hold the button long enough to ask a question."
        )
    if duration_ms > MAX_QUESTION_SECONDS * 1000:
        raise InvalidRecordedAnswerQuestion("Keep the question under 30 seconds.")
    return RecordedAnswerQuestion(pcm=pcm, duration_ms=duration_ms)


def answer_computation_xml_v1(
    explanation: AnswerExplanationView,
) -> str:
    """Wrap the exact desktop Inputs and verbose Explanation projections."""

    root = ET.Element("AnswerComputation", {"version": "1"})
    inputs = ET.SubElement(root, "Inputs", {"encoding": "application/json"})
    inputs.text = _compact_json(view_json(explanation.inputs))
    verbose = ET.SubElement(
        root,
        "Explanation",
        {"detail": "verbose", "encoding": "application/json"},
    )
    verbose.text = _compact_json(view_json(explanation.lineage.verbose))
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
