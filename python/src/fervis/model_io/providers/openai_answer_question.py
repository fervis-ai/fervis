"""OpenAI Realtime adapter for disposable answer-computation questions."""

from __future__ import annotations

import base64
import binascii
import io
import wave
from typing import Any

from fervis.delivery.answer_question import (
    AnswerQuestionGenerationError,
    AnswerQuestionService,
    GeneratedAnswerAudio,
    LineageAnswerExplanationLoader,
    RecordedAnswerQuestion,
)
from fervis.lineage.views.query import LineageQueryPort
from fervis.model_io.providers.chat_runtime import ChatProviderConfig
from fervis.model_io.providers.openai_compatible_adapter import (
    OPENAI_COMPATIBLE_PROVIDER_CONFIGS,
)
from fervis.model_io.providers.openai_compatible_adapter.loop_adapter import (
    openai_compatible_client,
)
from fervis.observability.query import ObservabilityQueryPort


DEFAULT_MODEL = "gpt-realtime-2.1"
DEFAULT_VOICE = "cedar"
PCM_SAMPLE_RATE = 24_000
MAX_AUDIO_SECONDS = 60
MAX_PCM_BYTES = PCM_SAMPLE_RATE * 2 * MAX_AUDIO_SECONDS


class OpenAIRealtimeAnswerQuestionGenerator:
    def __init__(
        self,
        *,
        provider_config: ChatProviderConfig,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        client: Any = None,
    ) -> None:
        self._provider_config = provider_config
        self._model = model
        self._voice = voice
        self._client = client

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
        question: RecordedAnswerQuestion,
    ) -> GeneratedAnswerAudio:
        pcm = bytearray()
        completed = False
        try:
            with self._open_connection() as connection:
                connection.response.create(
                    response={
                        "audio": {
                            "output": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": PCM_SAMPLE_RATE,
                                },
                                "voice": self._voice,
                            }
                        },
                        "conversation": "none",
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": input_text},
                                    {
                                        "type": "input_audio",
                                        "audio": base64.b64encode(question.pcm).decode(
                                            "ascii"
                                        ),
                                    },
                                ],
                            }
                        ],
                        "instructions": instructions,
                        "max_output_tokens": 1000,
                        "output_modalities": ["audio"],
                    }
                )
                for event in connection:
                    event_type = str(getattr(event, "type", ""))
                    if event_type == "response.output_audio.delta":
                        pcm.extend(_decode_audio_delta(getattr(event, "delta", "")))
                        if len(pcm) > MAX_PCM_BYTES:
                            raise AnswerQuestionGenerationError(
                                "The spoken answer exceeded 60 seconds."
                            )
                    elif event_type == "error":
                        message = _provider_error_message(event)
                        raise AnswerQuestionGenerationError(
                            message,
                            public_message=_public_failure_message(message),
                        )
                    elif event_type == "response.done":
                        status = str(
                            getattr(getattr(event, "response", None), "status", "")
                        )
                        if status != "completed":
                            raise AnswerQuestionGenerationError(
                                "OpenAI did not complete the spoken answer."
                            )
                        completed = True
                        break
        except AnswerQuestionGenerationError:
            raise
        except Exception as exc:
            raise AnswerQuestionGenerationError(
                "OpenAI spoken answer generation failed.",
                public_message=_public_failure_message(str(exc)),
            ) from exc
        if not completed or not pcm:
            raise AnswerQuestionGenerationError(
                "OpenAI returned no completed spoken answer."
            )
        return GeneratedAnswerAudio(data=_wav_bytes(bytes(pcm)))

    def _open_connection(self):
        client = self._client
        if client is None:
            client = openai_compatible_client(
                api_key=self._provider_config.api_key,
                base_url=self._provider_config.base_url,
                max_retries=1,
                timeout=70.0,
            )
        return client.realtime.connect(
            model=self._model,
            websocket_connection_options={
                "open_timeout": 15,
                "close_timeout": 5,
            },
        )


def configured_answer_question_service(
    lineage_query: LineageQueryPort,
    observability_query: ObservabilityQueryPort,
) -> AnswerQuestionService | None:
    provider_config = _openai_provider_config()
    if not provider_config.api_key:
        return None
    return AnswerQuestionService(
        explanations=LineageAnswerExplanationLoader(
            lineage_query=lineage_query,
            observability_query=observability_query,
        ),
        generator=OpenAIRealtimeAnswerQuestionGenerator(
            provider_config=provider_config,
        ),
    )


def _openai_provider_config() -> ChatProviderConfig:
    return next(
        config
        for config in OPENAI_COMPATIBLE_PROVIDER_CONFIGS
        if config.provider_name == "openai"
    )


def _decode_audio_delta(value: object) -> bytes:
    try:
        return base64.b64decode(str(value), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AnswerQuestionGenerationError(
            "OpenAI returned invalid spoken answer audio."
        ) from exc


def _provider_error_message(event: object) -> str:
    error = getattr(event, "error", None)
    message = str(getattr(error, "message", "") or "").strip()
    return message or "OpenAI rejected spoken answer generation."


def _public_failure_message(provider_detail: str) -> str:
    if "insufficient_quota" in provider_detail.lower():
        return (
            "OpenAI quota is exhausted. Add credits or increase the project budget, "
            "then try again."
        )
    return "The answer explanation could not be generated."


def _wav_bytes(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(PCM_SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()
