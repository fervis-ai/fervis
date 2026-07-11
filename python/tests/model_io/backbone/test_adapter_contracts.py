from dataclasses import dataclass
import json
from types import SimpleNamespace
from typing import Any

import pytest

from fervis.observability.usage_types import CostSource
from fervis.model_io.backbone.dto import (
    ToolSpec,
    ProviderOutputMode,
    ProviderRunRequest,
    ProviderRunResult,
)
from fervis.model_io.backbone.tool_codec import decode_json_object_argument
from fervis.lookup.question_contract import (
    build_answer_request_contract_schema,
    build_missing_input_clarification_schema,
    build_question_contract_decisions_schema,
    parse_question_contract,
    QuestionContractTurnPrompt,
)
from fervis.lookup.question_contract.model import QuestionContractRequest
from fervis.lookup.fact_planning.schema import build_fact_plan_schema
from fervis.model_io.structured_output.errors import RequiredToolOutputError
from fervis.model_io.structured_output.generation import (
    generate_one_of_tool_output,
)
from fervis.model_io.structured_output.provider_budget import provider_budget_tool_specs
from fervis.model_io.backbone.factory import (
    build_provider_backbone,
    reset_provider_backbone_for_tests,
)
from fervis.model_io.backbone.registry import (
    ProviderRegistration,
    register_provider,
)
from fervis.model_io.providers.chat_runtime import ChatProviderConfig
from fervis.model_io.providers.chat_runtime import build_provider_run_result
from fervis.model_io.providers.chat_runtime import chat_json_system_prompt
from fervis.model_io.providers.chat_runtime import chat_tool_system_prompt
from fervis.model_io.providers.chat_runtime import provider_error_payload
from fervis.model_io.providers.chat_runtime import _provider_error_context
from fervis.model_io.providers.chat_runtime import provider_hard_timeout_seconds
from fervis.model_io.providers.chat_runtime import run_provider_worker_with_timeout
from fervis.model_io.providers.chat_runtime import provider_sdk_status
from fervis.model_io.providers.chat_runtime import provider_timeout_seconds
from fervis.model_io.providers.chat_runtime import ProviderExecutionError
from fervis.model_io.pricing import ModelPricing
from fervis.model_io.providers.anthropic_adapter import (
    build_anthropic_registration,
)
from fervis.model_io.providers.anthropic_adapter import (
    loop_adapter as anthropic_loop,
)
from fervis.model_io.providers.anthropic_adapter.loop_adapter import (
    AnthropicLoopRuntime,
)
from fervis.model_io.providers.openai_compatible_adapter import (
    loop_adapter as openai_compatible_loop,
)
from fervis.model_io.providers.openai_compatible_adapter.loop_adapter import (
    OpenAICompatibleLoopRuntime,
    OpenAICompatibleRequestPayload,
)
from fervis import errors as api_errors
from tests.model_io.backbone.source_binding_fixtures import (
    source_binding_tool_spec,
)
from tests.testkit.provider_native import provider_native_test_arguments


def _question_contract_schema() -> dict[str, object]:
    return build_question_contract_decisions_schema()


def _fact_plan_tool_specs() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="submit_pattern_fact_plan",
            description="Submit one typed fact plan.",
            strict=True,
            input_schema=_selected_fact_plan_schema(),
        ),
    )


def _selected_fact_plan_schema() -> dict[str, object]:
    return build_fact_plan_schema(
        requested_fact_ids=("fact_1",),
        pattern_names=("direct_field_value",),
        selected_plan_shapes_by_requested_fact_id={
            "fact_1": "direct_field_value",
        },
        source_binding_ids_by_requested_fact_id={"fact_1": ("sb_1",)},
        answer_output_ids_by_requested_fact_id={"fact_1": ("answer_1",)},
        answer_output_ids_by_source_binding_id={"sb_1": ("answer_1",)},
        source_binding_ids_by_requirement_by_requested_fact_id={},
        grouped_ranked_choices_by_requested_fact_id={},
        scalar_aggregate_choices_by_requested_fact_id={},
        field_ids_by_source_binding_id={"sb_1": ("amount",)},
    )


def _question_contract_tool_specs() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="submit_answer_request_contract",
            description="Submit complete catalog-blind answer request contracts.",
            strict=True,
            input_schema=build_answer_request_contract_schema(),
        ),
        ToolSpec(
            name="submit_missing_input_clarification",
            description=(
                "Submit a missing-input clarification request for the question-contract turn."
            ),
            strict=True,
            input_schema=build_missing_input_clarification_schema(),
        ),
    )


def _openai_test_config() -> ChatProviderConfig:
    return ChatProviderConfig(
        provider_name="openai",
        model_name="gpt-5.4-mini",
        api_key_env_var="OPENAI_API_KEY",
        sdk_name="openai-compatible-chat-completions",
        default_base_url="https://api.openai.com/v1",
        max_output_tokens_parameter="max_completion_tokens",
    )


def test_question_contract_schema_is_decisions_only():
    schema = _question_contract_schema()
    branches = {
        branch["properties"]["kind"]["enum"][0]: branch for branch in schema["oneOf"]
    }
    answer_contract_schema = branches["question_contract"]
    clarification_schema = branches["needs_clarification"]
    answer_request_schema = answer_contract_schema["properties"]["answer_requests"][
        "items"
    ]

    assert {
        "has_one_of": "oneOf" in schema,
        "answer_required": answer_contract_schema["required"],
        "answer_kind_enum": answer_contract_schema["properties"]["kind"]["enum"],
        "clarification_required": clarification_schema["required"],
        "clarification_kind_enum": clarification_schema["properties"]["kind"]["enum"],
        "has_prior_answer_references": (
            "prior_answer_references" in answer_request_schema["properties"]
        ),
    } == {
        "has_one_of": True,
        "answer_required": [
            "kind",
            "answer_requests_count",
            "question_inputs",
            "answer_requests",
            "question_input_inventory_check",
        ],
        "answer_kind_enum": ["question_contract"],
        "clarification_required": ["kind", "missing"],
        "clarification_kind_enum": ["needs_clarification"],
        "has_prior_answer_references": False,
    }


def test_question_contract_inventory_check_schema_uses_boolean_shape_not_literal_enum():
    schema = build_answer_request_contract_schema()

    inventory_check_schema = schema["properties"]["question_input_inventory_check"]
    declared_schema = inventory_check_schema["properties"][
        "all_input_like_phrases_declared"
    ]

    assert declared_schema == {"type": "boolean"}


def test_question_contract_schema_omits_provider_unsupported_unique_items():
    canonical_schema = build_question_contract_decisions_schema()

    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=_question_contract_tool_specs(),
        )
    )

    projected_schema = anthropic_loop._message_kwargs(payload)["tools"][0][
        "input_schema"
    ]
    assert "uniqueItems" not in json.dumps(canonical_schema)
    assert "uniqueItems" not in json.dumps(projected_schema)


def test_provider_native_question_contract_fixture_matches_current_schema():
    from jsonschema import validate

    payload = provider_native_test_arguments(
        tool_name="submit_answer_request_contract",
        prompt="",
        tool_specs=(),
    )

    validate(instance=payload, schema=build_question_contract_decisions_schema())
    parse_question_contract(
        tool_name="submit_answer_request_contract",
        payload=payload,
        question_context="What is the test adapter answer?",
    )


def test_question_contract_schema_accepts_time_value_owned_by_input_decision():
    from jsonschema import validate

    payload = _question_contract_with_time_value_input()

    validate(instance=payload, schema=build_question_contract_decisions_schema())


def test_question_contract_turn_schema_rejects_unavailable_conversation_resolution_inputs():
    from jsonschema import ValidationError, validate

    prompt = QuestionContractTurnPrompt(
        QuestionContractRequest(
            current_question="How many sales happened today?",
            conversation_context={},
        )
    )
    schema = prompt.tool_contract().tool_specs[0].input_schema
    payload = _question_contract_with_time_value_input()
    payload["question_inputs"][0] = {
        "input_ref": "today_time_1",
        "source": "conversation_resolution",
        "reference_text": "today",
        "occurrence": 1,
        "resolved_input_ref": "today_time_1",
        "inventory_check": {
            "why_this_is_an_input": "This is the time phrase constraining the count."
        },
        "kind": "row_set_reference",
    }

    with pytest.raises(ValidationError):
        validate(instance=payload, schema=schema)


def test_question_contract_parser_fails_on_model_authored_requested_facts():
    payload = provider_native_test_arguments(
        tool_name="submit_answer_request_contract",
        prompt="",
        tool_specs=(),
    )
    payload["requested_facts"] = [
        {
            "id": "model_authored_fact",
            "description": "model-authored facts must not own the contract",
        }
    ]

    with pytest.raises(ValueError, match="unparsed fields: requested_facts"):
        parse_question_contract(
            tool_name="submit_answer_request_contract",
            payload=payload,
            question_context="What is the test adapter answer?",
        )

    with pytest.raises(ValueError, match="answer_requests"):
        parse_question_contract(
            tool_name="submit_answer_request_contract",
            payload={
                "kind": "question_contract",
                "answer_requests_count": 1,
                "question_inputs": [],
            },
            question_context="How much sales did we make?",
        )

    with pytest.raises(ValueError, match="unknown question contract tool"):
        parse_question_contract(
            tool_name="submit_requested_facts",
            payload={
                "kind": "question_contract",
                "answer_requests_count": 1,
                "answer_requests": [],
            },
            question_context="How much sales did we make?",
        )


def _question_contract_with_time_value_input() -> dict[str, object]:
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [
            {
                "kind": "literal_text",
                "input_ref": "time_1",
                "source": "question_context",
                "value_source_text": "today",
                "resolved_value_text": "today",
                "role": "time_value",
                "inventory_check": {
                    "why_this_is_an_input": "today constrains the sales count"
                },
            }
        ],
        "answer_requests": [
            {
                "answer_fact": "sales today",
                "answer_expression": {"family": "scalar_aggregate"},
                "answer_subject": {
                    "subject_text": "sales",
                    "instance_interpretation": {
                        "kind": "NORMAL_BUSINESS_INSTANCE",
                    },
                },
                "answer_population": {
                    "population_label": "sales today",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                            "owned_question_input_refs": [],
                        }
                    ],
                },
                "answer_outputs": [
                    {
                        "description": "sales count",
                    }
                ],
                "used_question_inputs": ["time_1"],
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }


def test_question_contract_schema_rejects_answer_text():
    schema = build_question_contract_decisions_schema()

    with pytest.raises(Exception):
        from jsonschema import validate

        validate(
            instance={
                "kind": "question_contract",
                "answer_requests_count": 1,
                "answer_text": "How much sales did we make?",
                "question_inputs": [],
                "answer_requests": [
                    {
                        "answer_fact": "sales",
                        "answer_outputs": [
                            {
                                "description": "sales",
                            }
                        ],
                        "used_question_inputs": [],
                    }
                ],
            },
            schema=schema,
        )


def test_fact_plan_schema_uses_structured_missing_catalog_inputs():
    serialized = json.dumps(_selected_fact_plan_schema())
    outcome_variants = _selected_fact_plan_schema()["properties"]["outcome"]["oneOf"]
    missing_input_schema = next(
        variant["properties"]["missing_catalog_inputs"]["items"]
        for variant in outcome_variants
        if "missing_catalog_inputs" in variant.get("properties", {})
    )
    variants = missing_input_schema["oneOf"]
    assert {
        "has_missing_catalog_inputs": "missing_catalog_inputs" in serialized,
        "variant_kinds": {
            variant["properties"]["kind"]["enum"][0] for variant in variants
        },
        "variant_min_lengths": [
            (
                variant["properties"]["id"]["minLength"],
                variant["properties"]["requested_fact_id"]["minLength"],
            )
            for variant in variants
        ],
    } == {
        "has_missing_catalog_inputs": True,
        "variant_kinds": {
            "missing_catalog_required_input",
            "missing_catalog_choice_input",
        },
        "variant_min_lengths": [(1, 1), (1, 1)],
    }


def _anthropic_test_config() -> ChatProviderConfig:
    return ChatProviderConfig(
        provider_name="anthropic",
        model_name="claude-haiku-4-5-20251001",
        api_key_env_var="ANTHROPIC_API_KEY",
        sdk_name="anthropic-messages",
    )


def test_provider_result_uses_provider_tokens_and_configured_cost_breakdown():
    result = build_provider_run_result(
        ChatProviderConfig(
            provider_name="test_provider",
            model_name="test-model",
            api_key_env_var="TEST_API_KEY",
            sdk_name="test-sdk",
            input_cost_per_million_tokens=1000,
            output_cost_per_million_tokens=2000,
            thinking_cost_per_million_tokens=3000,
            pricing_version="test-provider-2026-05",
        ),
        answer="ok",
        input_tokens=10,
        output_tokens=20,
        thinking_tokens=5,
    )

    assert result.usage == {
        "inputTokens": 10,
        "outputTokens": 20,
        "thinkingTokens": 5,
        "costUsd": 0.065,
        "inputCostUsd": 0.01,
        "outputCostUsd": 0.04,
        "thinkingCostUsd": 0.015,
        "costSource": "configured_provider_pricing",
        "pricingVersion": "test-provider-2026-05",
    }


def test_provider_result_prices_model_subcalls_with_same_provider_pricing():
    result = build_provider_run_result(
        ChatProviderConfig(
            provider_name="test_provider",
            model_name="test-model",
            api_key_env_var="TEST_API_KEY",
            sdk_name="test-sdk",
            input_cost_per_million_tokens=1000,
            output_cost_per_million_tokens=2000,
            thinking_cost_per_million_tokens=3000,
            pricing_version="test-provider-2026-05",
        ),
        answer="ok",
        input_tokens=30,
        output_tokens=12,
        thinking_tokens=5,
        usage_details={
            "modelSubcalls": [
                {
                    "callId": "source_binding.stage_1",
                    "phase": "source_binding.stage_1",
                    "inputTokens": 10,
                    "outputTokens": 4,
                    "thinkingTokens": 1,
                    "promptChars": 100,
                    "schemaChars": 20,
                    "toolSpecChars": 30,
                },
                {
                    "callId": "source_binding.stage_2.1",
                    "phase": "source_binding.stage_2",
                    "inputTokens": 20,
                    "outputTokens": 8,
                    "thinkingTokens": 4,
                    "promptChars": 200,
                    "schemaChars": 40,
                    "toolSpecChars": 60,
                },
            ]
        },
    )

    assert result.usage["modelSubcalls"] == [
        {
            "callId": "source_binding.stage_1",
            "phase": "source_binding.stage_1",
            "inputTokens": 10,
            "outputTokens": 4,
            "thinkingTokens": 1,
            "costUsd": 0.021,
            "inputCostUsd": 0.01,
            "outputCostUsd": 0.008,
            "thinkingCostUsd": 0.003,
            "promptChars": 100,
            "schemaChars": 20,
            "toolSpecChars": 30,
        },
        {
            "callId": "source_binding.stage_2.1",
            "phase": "source_binding.stage_2",
            "inputTokens": 20,
            "outputTokens": 8,
            "thinkingTokens": 4,
            "costUsd": 0.048,
            "inputCostUsd": 0.02,
            "outputCostUsd": 0.016,
            "thinkingCostUsd": 0.012,
            "promptChars": 200,
            "schemaChars": 40,
            "toolSpecChars": 60,
        },
    ]


def test_provider_result_does_not_fabricate_missing_provider_usage():
    with pytest.raises(ProviderExecutionError, match="input token usage"):
        build_provider_run_result(
            ChatProviderConfig(
                provider_name="test_provider",
                model_name="test-model",
                api_key_env_var="TEST_API_KEY",
                sdk_name="test-sdk",
                input_cost_per_million_tokens=1000,
                output_cost_per_million_tokens=2000,
                pricing_version="test-provider-2026-05",
            ),
            answer="ok",
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
        )


def test_provider_status_does_not_require_pricing_before_sdk_call(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret")

    status = provider_sdk_status(
        ChatProviderConfig(
            provider_name="test_provider",
            model_name="test-model",
            api_key_env_var="TEST_API_KEY",
            sdk_name="test-sdk",
        ),
        sdk_available=True,
    )

    assert status == "enabled"


def test_provider_result_prices_usage_from_models_dev(monkeypatch):
    from fervis.model_io.providers import chat_runtime

    monkeypatch.setattr(
        chat_runtime,
        "resolve_model_pricing",
        lambda *, provider, model_key: ModelPricing(
            input_cost_per_million_tokens=10,
            output_cost_per_million_tokens=20,
            thinking_cost_per_million_tokens=30,
            pricing_version="models.dev:test_provider/test-model",
            cost_source=CostSource.MODELS_DEV,
        ),
    )

    result = build_provider_run_result(
        ChatProviderConfig(
            provider_name="test_provider",
            model_name="test-model",
            api_key_env_var="TEST_API_KEY",
            sdk_name="test-sdk",
        ),
        answer="ok",
        input_tokens=10,
        output_tokens=20,
        thinking_tokens=5,
    )

    assert result.usage == {
        "inputTokens": 10,
        "outputTokens": 20,
        "thinkingTokens": 5,
        "costUsd": 0.00065,
        "inputCostUsd": 0.0001,
        "outputCostUsd": 0.0004,
        "thinkingCostUsd": 0.00015,
        "costSource": "models_dev",
        "pricingVersion": "models.dev:test_provider/test-model",
    }


def test_provider_result_marks_usage_unpriced_when_models_dev_has_no_price(monkeypatch):
    from fervis.model_io.providers import chat_runtime

    monkeypatch.setattr(
        chat_runtime,
        "resolve_model_pricing",
        lambda *, provider, model_key: ModelPricing.unpriced(
            pricing_version="models.dev:test_provider/missing-model"
        ),
    )

    result = build_provider_run_result(
        ChatProviderConfig(
            provider_name="test_provider",
            model_name="missing-model",
            api_key_env_var="TEST_API_KEY",
            sdk_name="test-sdk",
        ),
        answer="ok",
        input_tokens=10,
        output_tokens=20,
        thinking_tokens=5,
    )

    assert result.usage == {
        "inputTokens": 10,
        "outputTokens": 20,
        "thinkingTokens": 5,
        "costUsd": 0,
        "costSource": "provider_usage_unpriced",
        "pricingVersion": "models.dev:test_provider/missing-model",
    }


@dataclass
class StubLoopRuntime:
    def run(self, request: ProviderRunRequest) -> ProviderRunResult:
        return ProviderRunResult(
            provider="stub",
            answer=f"stub:{request.prompt}",
            usage={
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 1,
                "costUsd": 0.000003,
                "inputCostUsd": 0.000001,
                "outputCostUsd": 0.000001,
                "thinkingCostUsd": 0.000001,
                "costSource": CostSource.CONFIGURED_PROVIDER_PRICING,
                "pricingVersion": "test-provider-2026-05",
            },
            raw_payload={"provider": "stub"},
        )


class StubModelAdapter:
    provider_name = "stub"

    def __init__(self, loop_runtime: StubLoopRuntime):
        self.loop_runtime = loop_runtime

    def generate(
        self,
        *,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode=None,
        tool_specs=(),
    ) -> dict[str, Any]:
        result = self.loop_runtime.run(
            ProviderRunRequest(
                provider="stub",
                prompt=prompt,
                max_thinking_tokens=max_thinking_tokens,
                system_prompt=system_prompt,
            )
        )
        return {
            "provider": result.provider,
            "answer": result.answer,
            "usage": result.usage,
            "toolRequests": [],
            "raw": result.raw_payload,
        }


class StubStreamRuntime:
    def map_events(self, *, run_id: str, events: list[dict[str, Any]]):
        return []


class StubSessionRuntime:
    def continue_session(self, *, session_id: str | None):
        from fervis.model_io.backbone.dto import SessionRef

        sid = session_id or "stub-session"
        return SessionRef(session_id=sid, provider_session_id=f"stub:{sid}")

    def resume_session(self, *, session_id: str):
        from fervis.model_io.backbone.dto import SessionRef

        return SessionRef(
            session_id=session_id, provider_session_id=f"stub:{session_id}"
        )

    def fork_session(self, *, session_id: str, branch_point_event_id: str):
        from fervis.model_io.backbone.dto import SessionRef

        return SessionRef(session_id="stub-fork", provider_session_id="stub:stub-fork")


class StubHitlRuntime:
    def interruption_required(self, *, safety_classification: str) -> bool:
        return False

    def approve(self, *, reason: str | None = None):
        from fervis.model_io.backbone.dto import ToolDecision

        return ToolDecision(decision="approve", reason=reason)

    def reject(self, *, reason: str | None = None):
        from fervis.model_io.backbone.dto import ToolDecision

        return ToolDecision(decision="deny", reason=reason)


class StubHooksRuntime:
    def build_hooks(self):
        return []


class StubTraceRuntime:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


def _register_stub_provider() -> None:
    loop = StubLoopRuntime()
    register_provider(
        ProviderRegistration(
            name="stub",
            model_adapter=StubModelAdapter(loop),
            loop_runtime=loop,
            stream_runtime=StubStreamRuntime(),
            session_runtime=StubSessionRuntime(),
            hitl_runtime=StubHitlRuntime(),
            hooks_runtime=StubHooksRuntime(),
            trace_runtime=StubTraceRuntime(),
        )
    )


def test_tool_codec_preserves_malformed_json_object_for_planner_runtime_diagnostics():
    assert decode_json_object_argument("not-json") == {
        "_malformed_json_object": "not-json"
    }


def test_anthropic_adapter_implements_provider_contracts(fervis_foundation_reset):
    registration = build_anthropic_registration()

    session = registration.session_runtime.continue_session(session_id="session-1")
    assert {
        "model_adapter_generate": callable(
            getattr(registration.model_adapter, "generate", None)
        ),
        "loop_run": callable(registration.loop_runtime.run),
        "stream_map_events": callable(registration.stream_runtime.map_events),
        "hitl_approve": callable(registration.hitl_runtime.approve),
        "hooks_build": callable(registration.hooks_runtime.build_hooks),
        "trace_record": callable(registration.trace_runtime.record),
        "provider_session_id": session.provider_session_id,
        "session_provider": session.metadata["provider"],
    } == {
        "model_adapter_generate": True,
        "loop_run": True,
        "stream_map_events": True,
        "hitl_approve": True,
        "hooks_build": True,
        "trace_record": True,
        "provider_session_id": "anthropic:session-1",
        "session_provider": "anthropic",
    }


def test_anthropic_client_does_not_discover_environment_proxies_in_worker(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        anthropic_loop,
        "anthropic",
        type("FakeAnthropicModule", (), {"Anthropic": FakeAnthropic}),
    )

    payload = AnthropicLoopRuntime(config=_anthropic_test_config()).request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="hello",
            max_thinking_tokens=64,
            system_prompt="system",
        )
    )

    client = anthropic_loop._client(payload)

    assert isinstance(client, FakeAnthropic)
    assert captured["http_client"]._trust_env is False
    captured["http_client"].close()


def test_anthropic_runtime_hard_timeout_fails_before_gunicorn_abort(
    fervis_foundation_reset,
    monkeypatch,
):
    def blocking_worker(payload, result_queue):
        import time

        time.sleep(2)

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", blocking_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    monkeypatch.setenv("FERVIS_PROVIDER_HARD_TIMEOUT_SECONDS", "1")

    with pytest.raises(api_errors.Unavailable):
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="slow",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )


def test_anthropic_runtime_preserves_child_error_context(
    fervis_foundation_reset,
    monkeypatch,
):
    def failing_worker(payload, result_queue):
        result_queue.put(
            {
                "ok": False,
                "errorClass": "RateLimitError",
                "error": "rate limited by provider",
            }
        )

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", failing_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    with pytest.raises(api_errors.RateLimit) as exc:
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="hello",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )

    assert (
        exc.value.code,
        exc.value.context["error_class"],
        exc.value.context["reason"],
    ) == ("llm_api_rate_limited", "RateLimitError", "rate limited by provider")


def test_provider_runtime_maps_timeout_to_llm_timeout_error(
    fervis_foundation_reset,
    monkeypatch,
):
    def blocking_worker(payload, result_queue):
        import time

        time.sleep(2)

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", blocking_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    monkeypatch.setenv("FERVIS_PROVIDER_HARD_TIMEOUT_SECONDS", "1")

    with pytest.raises(api_errors.Unavailable) as exc:
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="slow",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )

    assert exc.value.code == "llm_api_timeout"


def test_provider_runtime_maps_bad_request_like_errors_to_llm_bad_request(
    fervis_foundation_reset,
    monkeypatch,
):
    def failing_worker(payload, result_queue):
        result_queue.put(
            {
                "ok": False,
                "errorClass": "BadRequestError",
                "error": "response_format schema invalid",
            }
        )

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", failing_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    with pytest.raises(api_errors.Unavailable) as exc:
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="hello",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )

    assert exc.value.code == "llm_api_bad_request"


def test_provider_runtime_maps_authentication_errors_to_typed_llm_error(
    fervis_foundation_reset,
    monkeypatch,
):
    def failing_worker(payload, result_queue):
        result_queue.put(
            {
                "ok": False,
                "errorClass": "AuthenticationError",
                "error": "invalid api key",
                "statusCode": "401",
                "errorType": "authentication_error",
            }
        )

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", failing_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    with pytest.raises(api_errors.Unavailable) as exc:
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="hello",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )

    assert exc.value.code == "llm_api_authentication_error"


def test_provider_runtime_maps_rate_limit_errors_to_rate_limit_error(
    fervis_foundation_reset,
    monkeypatch,
):
    def failing_worker(payload, result_queue):
        result_queue.put(
            {
                "ok": False,
                "errorClass": "RateLimitError",
                "error": "too many requests",
                "statusCode": "429",
                "errorType": "rate_limit_error",
            }
        )

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", failing_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    with pytest.raises(api_errors.RateLimit) as exc:
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="hello",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )

    assert exc.value.code == "llm_api_rate_limited"


def test_provider_runtime_extracts_anthropic_body_error_type_for_overload(
    fervis_foundation_reset,
    monkeypatch,
):
    def failing_worker(payload, result_queue):
        result_queue.put(
            {
                "ok": False,
                "errorClass": "InternalServerError",
                "error": "Grammar compilation is temporarily unavailable.",
                "statusCode": "503",
                "errorType": "overloaded_error",
                "requestId": "req_grammar",
            }
        )

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", failing_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    with pytest.raises(api_errors.RateLimit) as exc:
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="hello",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )

    assert exc.value.code == "llm_api_rate_limited"
    assert exc.value.context["provider_metadata"]["requestId"] == "req_grammar"


def test_provider_error_payload_extracts_nested_error_body_metadata():
    class ProviderError(Exception):
        status_code = 503
        body = {
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Grammar compilation is temporarily unavailable.",
            },
            "request_id": "req_nested",
        }

    payload = provider_error_payload(ProviderError("failed"))

    assert (
        payload["statusCode"],
        payload["errorType"],
        payload["requestId"],
    ) == ("503", "overloaded_error", "req_nested")


def test_provider_worker_uncaught_exception_returns_provider_error():
    def worker(_payload, _result_queue):
        raise RuntimeError("worker failed")

    with pytest.raises(ProviderExecutionError) as exc_info:
        run_provider_worker_with_timeout(
            worker,
            payload={},
            timeout_reason="timed out",
        )

    assert exc_info.value.error_class == "RuntimeError"
    assert exc_info.value.reason == "worker failed"


def test_provider_worker_no_result_reports_worker_completion():
    def worker(_payload, _result_queue):
        return None

    with pytest.raises(ProviderExecutionError) as exc_info:
        run_provider_worker_with_timeout(
            worker,
            payload={},
            timeout_reason="timed out",
        )

    assert exc_info.value.error_class == "ProviderNoResultError"
    assert exc_info.value.context == {
        "provider_metadata": {
            "worker_completion": "no_result",
        }
    }


def test_anthropic_runtime_does_not_send_tool_choice_without_tools():
    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=(),
        )
    )

    message_kwargs = anthropic_loop._message_kwargs(payload)

    assert "tools" not in message_kwargs
    assert "tool_choice" not in message_kwargs


def test_default_chat_json_system_prompt_retains_single_tool_call_contract():
    prompt = chat_json_system_prompt()

    assert "Return exactly one raw JSON object and nothing else." in prompt
    assert "exactly one tool call object" in prompt


def test_provider_native_tool_system_prompt_does_not_request_json_text():
    prompt = chat_tool_system_prompt()

    assert (
        "provider-native tool call" in prompt,
        "raw JSON object" in prompt,
        "tool call object" in prompt,
    ) == (True, False, False)


def test_runtime_system_prompt_is_prefixed_to_tool_system_prompt():
    prompt = chat_tool_system_prompt("You are Ask Ozai.")

    assert prompt.startswith("You are Ask Ozai. ")
    assert "Use the required provider-native tool call exactly once." in prompt


def test_provider_timeout_defaults_allow_strict_tool_latency(monkeypatch):
    monkeypatch.delenv("FERVIS_PROVIDER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("FERVIS_PROVIDER_HARD_TIMEOUT_SECONDS", raising=False)

    assert provider_timeout_seconds() == 120
    assert provider_hard_timeout_seconds() == 150


def test_anthropic_tool_call_uses_provider_native_system_prompt():
    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=(),
        )
    )

    message_kwargs = anthropic_loop._message_kwargs(payload)

    assert message_kwargs["system"] == chat_tool_system_prompt()
    assert "raw JSON object" not in message_kwargs["system"]


def test_openai_compatible_tool_call_uses_provider_native_system_prompt():
    payload = OpenAICompatibleRequestPayload(
        api_key="test",
        base_url="https://example.test",
        model="gpt-test",
        max_tokens=1024,
        max_output_tokens_parameter="max_tokens",
        max_retries=0,
        timeout=20,
        temperature=0,
        prompt="{}",
        output_mode=ProviderOutputMode.TOOL_CALL,
        tool_specs=[],
        json_object_arguments_by_tool={},
        system_prompt="",
    )

    completion_kwargs = openai_compatible_loop._base_completion_kwargs(payload)

    system_prompt = completion_kwargs["messages"][0]["content"]
    assert system_prompt == chat_tool_system_prompt()
    assert "raw JSON object" not in system_prompt


def test_openai_compatible_tool_call_includes_runtime_system_prompt():
    payload = OpenAICompatibleRequestPayload(
        api_key="test",
        base_url="https://example.test",
        model="gpt-test",
        max_tokens=1024,
        max_output_tokens_parameter="max_tokens",
        max_retries=0,
        timeout=20,
        temperature=0,
        prompt="{}",
        output_mode=ProviderOutputMode.TOOL_CALL,
        tool_specs=[],
        json_object_arguments_by_tool={},
        system_prompt="You are Ask Ozai.",
    )

    completion_kwargs = openai_compatible_loop._base_completion_kwargs(payload)

    assert completion_kwargs["messages"][0]["content"].startswith("You are Ask Ozai. ")


def test_anthropic_adapter_sends_fact_plan_tool_contract():
    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=_fact_plan_tool_specs(),
        )
    )

    message_kwargs = anthropic_loop._message_kwargs(payload)
    tool = message_kwargs["tools"][0]
    schema = tool["input_schema"]

    assert tool["name"] == "submit_pattern_fact_plan"
    assert schema["required"] == ["outcome"]


def test_anthropic_adapter_sends_question_contract_tool_contracts():
    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=_question_contract_tool_specs(),
        )
    )

    message_kwargs = anthropic_loop._message_kwargs(payload)
    tools = message_kwargs["tools"]

    assert {
        "tool_names": [tool["name"] for tool in tools],
        "has_one_of": ["oneOf" in tool["input_schema"] for tool in tools],
        "required": [tool["input_schema"]["required"] for tool in tools],
        "tool_choice_type": message_kwargs["tool_choice"]["type"],
        "disable_parallel": message_kwargs["tool_choice"]["disable_parallel_tool_use"],
    } == {
        "tool_names": [
            "submit_answer_request_contract",
            "submit_missing_input_clarification",
        ],
        "has_one_of": [False, False],
        "required": [
            [
                "kind",
                "answer_requests_count",
                "question_inputs",
                "answer_requests",
                "question_input_inventory_check",
            ],
            ["kind", "missing"],
        ],
        "tool_choice_type": "any",
        "disable_parallel": True,
    }


def test_anthropic_adapter_strips_validator_only_schema_constraints():
    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=(
                ToolSpec(
                    name="submit_contract_payload",
                    description="Generic strict tool contract",
                    strict=True,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "stepId": {
                                "type": "string",
                                "pattern": "^endpoint_",
                            },
                            "items": {
                                "type": "array",
                                "items": {"type": "string", "pattern": "^field\\."},
                                "minItems": 1,
                                "maxItems": 3,
                            },
                            "rank": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["stepId", "items", "rank"],
                        "additionalProperties": False,
                        "modelSchemas": {
                            "stepId": {"type": "string", "minLength": 1},
                        },
                    },
                ),
            ),
        )
    )

    schema = anthropic_loop._message_kwargs(payload)["tools"][0]["input_schema"]
    serialized = str(schema)
    assert {
        "has_pattern": "pattern" in serialized,
        "has_model_schemas": "modelSchemas" in serialized,
        "step_id": schema["properties"]["stepId"],
        "item_min": schema["properties"]["items"]["minItems"],
        "item_max": schema["properties"]["items"]["maxItems"],
        "rank": schema["properties"]["rank"],
    } == {
        "has_pattern": False,
        "has_model_schemas": False,
        "step_id": {"type": "string"},
        "item_min": 1,
        "item_max": 3,
        "rank": {"type": "integer", "minimum": 1, "maximum": 10},
    }


def test_openai_adapter_strips_validator_only_schema_metadata():
    runtime = OpenAICompatibleLoopRuntime(config=_openai_test_config())
    input_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
        "modelSchemas": {
            "value": {"type": "string", "minLength": 1},
        },
    }
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="openai",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=(
                ToolSpec(
                    name="submit_contract_payload",
                    description="Generic strict tool contract",
                    strict=True,
                    input_schema=input_schema,
                ),
            ),
        )
    )

    schema = openai_compatible_loop._completion_kwargs(payload)["tools"][0]["function"][
        "parameters"
    ]

    assert (
        "modelSchemas" in input_schema,
        "modelSchemas" in json.dumps(schema),
        schema["properties"]["value"],
    ) == (True, False, {"type": "string"})


def test_anthropic_adapter_projects_one_of_to_supported_strict_schema():
    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=(
                ToolSpec(
                    name="submit_union_payload",
                    description="Union tool contract",
                    strict=True,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "oneOf": [
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "kind": {"enum": ["success"]},
                                            "value": {"type": "string"},
                                        },
                                        "required": ["kind", "value"],
                                    },
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "kind": {"enum": ["impossible"]},
                                            "reason": {"type": "string"},
                                        },
                                        "required": ["kind", "reason"],
                                    },
                                ]
                            }
                        },
                        "required": ["outcome"],
                        "additionalProperties": False,
                    },
                ),
            ),
        )
    )

    schema = anthropic_loop._message_kwargs(payload)["tools"][0]["input_schema"]
    outcome = schema["properties"]["outcome"]

    assert "oneOf" not in str(schema)
    assert outcome == {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"enum": ["success"]},
                    "value": {"type": "string"},
                },
                "required": ["kind", "value"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"enum": ["impossible"]},
                    "reason": {"type": "string"},
                },
                "required": ["kind", "reason"],
            },
        ]
    }


def test_anthropic_adapter_preserves_branch_required_fields_in_nested_unions():
    schema = anthropic_loop._anthropic_strict_schema(
        {
            "type": "object",
            "required": ["metric"],
            "properties": {
                "metric": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["kind", "record_id_field_id"],
                            "properties": {
                                "kind": {"enum": ["count_records"]},
                                "record_id_field_id": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "required": ["kind", "function", "field_id"],
                            "properties": {
                                "kind": {"enum": ["aggregate_field"]},
                                "field_id": {"type": "string"},
                                "function": {"enum": ["sum", "min", "max", "avg"]},
                            },
                            "additionalProperties": False,
                        },
                    ]
                }
            },
            "additionalProperties": False,
        }
    )

    metric_variants = schema["properties"]["metric"]["anyOf"]

    assert metric_variants[0]["required"] == ["kind", "record_id_field_id"]
    assert metric_variants[1]["required"] == ["kind", "function", "field_id"]


def test_anthropic_adapter_preserves_scalar_and_array_validation_constraints():
    schema = anthropic_loop._anthropic_strict_schema(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "uniqueItems": True,
                    "items": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                }
            },
            "required": ["items"],
        }
    )

    items_schema = schema["properties"]["items"]
    assert {
        "min_items": items_schema["minItems"],
        "max_items": items_schema["maxItems"],
        "unique_items": items_schema["uniqueItems"],
        "item_minimum": items_schema["items"]["minimum"],
        "item_maximum": items_schema["items"]["maximum"],
    } == {
        "min_items": 1,
        "max_items": 2,
        "unique_items": True,
        "item_minimum": 1,
        "item_maximum": 10,
    }


def test_openai_adapter_keeps_canonical_source_binding_grammar():
    spec = source_binding_tool_spec()
    canonical_schema_text = json.dumps(spec.input_schema)
    runtime = OpenAICompatibleLoopRuntime(config=_openai_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="openai",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=(spec,),
        )
    )

    schema = openai_compatible_loop._completion_kwargs(payload)["tools"][0]["function"][
        "parameters"
    ]
    source_bindings_variant = schema["properties"]["outcome"]["anyOf"][0]
    fact_binding = source_bindings_variant["properties"]["bindings_for_fact_1"]
    invocation_item = fact_binding["properties"]["metric"]
    canonical_invocation_item = spec.input_schema["properties"]["outcome"]["oneOf"][0][
        "properties"
    ]["bindings_for_fact_1"]["properties"]["metric"]
    invocation_variants = _schema_variants_by_binding_target(
        invocation_item,
        variant_key="anyOf",
    )
    canonical_invocation_variants = _schema_variants_by_binding_target(
        canonical_invocation_item,
        variant_key="oneOf",
    )

    invocation_item_text = json.dumps(invocation_item)
    assert {
        "canonical_schema_unchanged": json.dumps(spec.input_schema)
        == canonical_schema_text,
        "compact_invocation_item": tuple(invocation_item),
        "canonical_compact_invocation_item": tuple(canonical_invocation_item),
        "binding_targets": tuple(invocation_variants),
        "canonical_binding_targets": tuple(canonical_invocation_variants),
        "source_1_param_decisions": tuple(
            invocation_variants["target.source_1"]["properties"]["param_decisions"][
                "properties"
            ]
        ),
        "source_1_finite_choice_reviews": tuple(
            invocation_variants["target.source_1"]["properties"][
                "finite_choice_param_reviews"
            ]["properties"]
        ),
        "source_2_param_decisions": tuple(
            invocation_variants["target.source_2"]["properties"]["param_decisions"][
                "properties"
            ]
        ),
        "source_2_finite_choice_reviews": tuple(
            invocation_variants["target.source_2"]["properties"][
                "finite_choice_param_reviews"
            ]["properties"]
        ),
        "fulfillment_requires_real_answer_choice": "anyOf"
        in invocation_variants["target.source_1"]["properties"][
            "fulfillment_decisions"
        ],
        "has_finite_choice_reviews": (
            "finite_choice_param_reviews"
            in invocation_variants["target.source_1"]["properties"]
        ),
        "forbidden_terms_present": [
            term
            for term in (
                "optional_param_applicability",
                "choice_param_memberships",
                "safe_to_omit",
                "prefixItems",
                '"items": false',
                "allOf",
                "contains",
            )
            if term in invocation_item_text
        ],
    } == {
        "canonical_schema_unchanged": True,
        "compact_invocation_item": ("anyOf",),
        "canonical_compact_invocation_item": ("oneOf",),
        "binding_targets": ("target.source_1", "target.source_2"),
        "canonical_binding_targets": ("target.source_1", "target.source_2"),
        "source_1_param_decisions": ("start_date",),
        "source_1_finite_choice_reviews": ("status",),
        "source_2_param_decisions": (),
        "source_2_finite_choice_reviews": (),
        "fulfillment_requires_real_answer_choice": False,
        "has_finite_choice_reviews": True,
        "forbidden_terms_present": [],
    }


def test_openai_accepts_source_binding_schema_nesting_depth():
    spec = source_binding_tool_spec()
    projected_schema = openai_compatible_loop._openai_strict_schema(
        spec.input_schema
    )

    assert _maximum_container_nesting(projected_schema) <= 10


def _maximum_container_nesting(schema: object, *, depth: int = 0) -> int:
    if not isinstance(schema, dict):
        return depth
    current_depth = depth + (schema.get("type") in {"object", "array"})
    child_depths = [current_depth]
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        child_depths.extend(
            _maximum_container_nesting(child, depth=current_depth)
            for child in properties.values()
        )
    for variant_key in ("anyOf", "oneOf"):
        variants = schema.get(variant_key, ())
        if isinstance(variants, list):
            child_depths.extend(
                _maximum_container_nesting(child, depth=current_depth)
                for child in variants
            )
    if "items" in schema:
        child_depths.append(
            _maximum_container_nesting(schema["items"], depth=current_depth)
        )
    return max(child_depths)


def _schema_variants_by_binding_target(
    schema: dict[str, Any],
    *,
    variant_key: str,
) -> dict[str, dict[str, Any]]:
    variants = schema[variant_key]
    return {
        variant["properties"]["binding_target_id"]["enum"][0]: variant
        for variant in variants
    }


def test_anthropic_adapter_receives_canonical_source_binding_tool():
    runtime = AnthropicLoopRuntime(config=_anthropic_test_config())
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="anthropic",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=(source_binding_tool_spec(),),
        )
    )

    tools = anthropic_loop._message_kwargs(payload)["tools"]
    tool_names = [tool["name"] for tool in tools]
    tool_payload = json.dumps(tools)

    assert (
        tool_names,
        [
            term
            for term in (
                "submit_source_invocation_selection",
                "submit_source_param_value_bindings",
                "choice_param_memberships",
            )
            if term in tool_payload
        ],
    ) == (["submit_source_binding"], [])


def test_provider_budget_uses_provider_registered_projection_hook(
    fervis_foundation_reset,
):
    reset_provider_backbone_for_tests()
    register_provider(build_anthropic_registration())
    spec = source_binding_tool_spec()
    projected = {"provider": "stub", "schema": "projected"}
    loop = StubLoopRuntime()
    register_provider(
        ProviderRegistration(
            name="stub",
            model_adapter=StubModelAdapter(loop),
            loop_runtime=loop,
            stream_runtime=StubStreamRuntime(),
            session_runtime=StubSessionRuntime(),
            hitl_runtime=StubHitlRuntime(),
            hooks_runtime=StubHooksRuntime(),
            trace_runtime=StubTraceRuntime(),
            budget_tool_specs=lambda tool_specs: (projected,),
        )
    )

    anthropic_specs = provider_budget_tool_specs(
        provider="anthropic",
        tool_specs=(spec,),
    )
    stub_specs = provider_budget_tool_specs(
        provider="stub",
        tool_specs=(spec,),
    )

    assert (
        stub_specs,
        isinstance(anthropic_specs[0], dict),
        json.dumps(anthropic_specs[0]).count('"oneOf"')
        < json.dumps(spec.input_schema).count('"oneOf"'),
    ) == ((projected,), True, True)


def test_provider_tool_arguments_are_validated_against_original_schema():
    class ModelPort:
        def generate(self, **kwargs):
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_pattern_fact_plan",
                        "arguments": {
                            "outcome": {
                                "kind": "answer",
                                "values": [
                                    {
                                        "id": "since_jan_13",
                                        "kind": "time",
                                        "payload": {
                                            "expression": "since Jan 13th",
                                            "anchor_date_ref": "ANCHOR_DATE",
                                            "timezone_ref": "ANCHOR_TIMEZONE",
                                            "intent": {
                                                "kind": "open_range",
                                                "start": "2026-01-13",
                                            },
                                        },
                                    }
                                ],
                                "operations": [],
                                "render_spec": {"relation_outputs": []},
                            }
                        },
                    }
                ),
                "usage": {},
            }

    with pytest.raises(RequiredToolOutputError, match="arguments do not match schema"):
        generate_one_of_tool_output(
            model_port=ModelPort(),
            provider="anthropic",
            system_prompt="system",
            prompt="{}",
            max_thinking_tokens=64,
            tool_specs=_fact_plan_tool_specs(),
        )


def test_openai_tool_argument_json_decode_error_preserves_raw_context():
    raw_arguments = '{"outcome":{"kind":"source_bindings","note":"unfinished}'
    choice = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="submit_source_bindings",
                        arguments=raw_arguments,
                    )
                )
            ]
        ),
    )

    with pytest.raises(ProviderExecutionError) as exc_info:
        openai_compatible_loop._answer_from_choice(
            choice,
            output_mode=ProviderOutputMode.TOOL_CALL,
            json_object_arguments={},
        )

    exc = exc_info.value
    assert {
        "error_class": exc.error_class,
        "tool_name": exc.context["tool_name"],
        "finish_reason": exc.context["finish_reason"],
        "raw_tool_arguments": exc.context["raw_tool_arguments"],
        "raw_tool_arguments_len": exc.context["raw_tool_arguments_len"],
        "json_error_pos_positive": exc.context["json_error_pos"] > 0,
    } == {
        "error_class": "JSONDecodeError",
        "tool_name": "submit_source_bindings",
        "finish_reason": "tool_calls",
        "raw_tool_arguments": raw_arguments,
        "raw_tool_arguments_len": len(raw_arguments),
        "json_error_pos_positive": True,
    }


def test_provider_execution_error_context_survives_as_provider_metadata():
    payload = provider_error_payload(
        ProviderExecutionError(
            error_class="JSONDecodeError",
            reason="Unterminated string",
            context={
                "finish_reason": "tool_calls",
                "raw_tool_arguments": '{"unfinished": "value}',
                "raw_tool_arguments_len": 22,
            },
        )
    )

    context = _provider_error_context(payload)

    assert context["provider_metadata"] == {
        "finish_reason": "tool_calls",
        "raw_tool_arguments": '{"unfinished": "value}',
        "raw_tool_arguments_len": "22",
    }


def test_provider_tool_output_strips_nullable_projection_nulls_before_validation():
    raw_answer = json.dumps(
        {
            "tool": "submit_payload",
            "arguments": {
                "requiredValue": "kept",
                "optionalValue": None,
                "nested": {"optionalChild": None},
            },
        }
    )

    class ModelPort:
        def generate(self, **kwargs):
            return {
                "answer": raw_answer,
                "usage": {},
            }

    output = generate_one_of_tool_output(
        model_port=ModelPort(),
        provider="openai",
        system_prompt="system",
        prompt="{}",
        max_thinking_tokens=64,
        tool_specs=(
            ToolSpec(
                name="submit_payload",
                description="Submit payload.",
                strict=True,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["requiredValue"],
                    "properties": {
                        "requiredValue": {"type": "string"},
                        "optionalValue": {"type": "string"},
                        "nested": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"optionalChild": {"type": "string"}},
                        },
                    },
                },
            ),
        ),
    )

    assert output.arguments == {"requiredValue": "kept", "nested": {}}
    assert json.loads(output.raw_output) == {"answer": raw_answer, "usage": {}}


def test_provider_tool_output_preserves_required_nullable_fields():
    class ModelPort:
        def generate(self, **kwargs):
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_payload",
                        "arguments": {
                            "requiredValue": "kept",
                            "requiredNullable": None,
                            "optionalValue": None,
                        },
                    }
                ),
                "usage": {},
            }

    output = generate_one_of_tool_output(
        model_port=ModelPort(),
        provider="openai",
        system_prompt="system",
        prompt="{}",
        max_thinking_tokens=64,
        tool_specs=(
            ToolSpec(
                name="submit_payload",
                description="Submit payload.",
                strict=True,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["requiredValue", "requiredNullable"],
                    "properties": {
                        "requiredValue": {"type": "string"},
                        "requiredNullable": {"type": ["string", "null"]},
                        "optionalValue": {"type": "string"},
                    },
                },
            ),
        ),
    )

    assert output.arguments == {
        "requiredValue": "kept",
        "requiredNullable": None,
    }


def test_provider_tool_output_raw_output_preserves_normalized_provider_envelope():
    raw_answer = json.dumps({"tool": "submit_payload", "arguments": {}})

    class ModelPort:
        def generate(self, **kwargs):
            return {
                "provider": "openai",
                "answer": raw_answer,
                "usage": {"inputTokens": 1},
                "raw": {"provider_request_id": "req_1", "finish_reason": "tool_calls"},
            }

    output = generate_one_of_tool_output(
        model_port=ModelPort(),
        provider="openai",
        system_prompt="system",
        prompt="{}",
        max_thinking_tokens=64,
        tool_specs=(
            ToolSpec(
                name="submit_payload",
                description="Submit payload.",
                strict=True,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            ),
        ),
    )

    assert json.loads(output.raw_output) == {
        "answer": raw_answer,
        "provider": "openai",
        "raw": {"finish_reason": "tool_calls", "provider_request_id": "req_1"},
        "usage": {"inputTokens": 1},
    }


def test_provider_tool_output_preserves_unknown_nulls_for_strict_validation():
    class ModelPort:
        def generate(self, **kwargs):
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_payload",
                        "arguments": {
                            "requiredValue": "kept",
                            "unexpected": None,
                        },
                    }
                ),
                "usage": {},
            }

    with pytest.raises(
        RequiredToolOutputError, match="arguments do not match schema"
    ) as exc_info:
        generate_one_of_tool_output(
            model_port=ModelPort(),
            provider="openai",
            system_prompt="system",
            prompt="{}",
            max_thinking_tokens=64,
            tool_specs=(
                ToolSpec(
                    name="submit_payload",
                    description="Submit payload.",
                    strict=True,
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["requiredValue"],
                        "properties": {
                            "requiredValue": {"type": "string"},
                        },
                    },
                ),
            ),
        )
    assert json.loads(exc_info.value.raw_output) == {
        "answer": json.dumps(
            {
                "tool": "submit_payload",
                "arguments": {
                    "requiredValue": "kept",
                    "unexpected": None,
                },
            }
        ),
        "usage": {},
    }


def test_provider_tool_output_preserves_required_nullable_fields_inside_union_branch():
    class ModelPort:
        def generate(self, **kwargs):
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_payload",
                        "arguments": {
                            "outcome": {
                                "kind": "needs_value",
                                "requiredNullable": None,
                                "optionalValue": None,
                            }
                        },
                    }
                ),
                "usage": {},
            }

    output = generate_one_of_tool_output(
        model_port=ModelPort(),
        provider="openai",
        system_prompt="system",
        prompt="{}",
        max_thinking_tokens=64,
        tool_specs=(
            ToolSpec(
                name="submit_payload",
                description="Submit payload.",
                strict=True,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["outcome"],
                    "properties": {
                        "outcome": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["kind", "requiredNullable"],
                                    "properties": {
                                        "kind": {"const": "needs_value"},
                                        "requiredNullable": {
                                            "type": ["string", "null"]
                                        },
                                        "optionalValue": {"type": "string"},
                                    },
                                },
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["kind", "value"],
                                    "properties": {
                                        "kind": {"const": "has_value"},
                                        "value": {"type": "string"},
                                    },
                                },
                            ]
                        }
                    },
                },
            ),
        ),
    )

    assert output.arguments == {
        "outcome": {
            "kind": "needs_value",
            "requiredNullable": None,
        }
    }


def test_openai_compatible_adapter_sends_fact_plan_tool_contract():
    runtime = OpenAICompatibleLoopRuntime(
        config=ChatProviderConfig(
            provider_name="openai",
            model_name="gpt-5.4-mini",
            api_key_env_var="OPENAI_API_KEY",
            sdk_name="openai-compatible-chat-completions",
            default_base_url="https://api.openai.com/v1",
            max_output_tokens_parameter="max_completion_tokens",
        )
    )
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="openai",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=_fact_plan_tool_specs(),
        )
    )

    completion_kwargs = openai_compatible_loop._completion_kwargs(payload)
    tool = completion_kwargs["tools"][0]["function"]
    schema = tool["parameters"]

    assert {
        "tool_name": tool["name"],
        "required": schema["required"],
        "has_one_of": "oneOf" in json.dumps(schema),
        "outcome_has_any_of": "anyOf" in schema["properties"]["outcome"],
    } == {
        "tool_name": "submit_pattern_fact_plan",
        "required": ["outcome"],
        "has_one_of": False,
        "outcome_has_any_of": True,
    }


def test_openai_compatible_adapter_sends_question_contract_tools_as_root_object_schemas():
    runtime = OpenAICompatibleLoopRuntime(
        config=ChatProviderConfig(
            provider_name="openai",
            model_name="gpt-5.4-mini",
            api_key_env_var="OPENAI_API_KEY",
            sdk_name="openai-compatible-chat-completions",
            default_base_url="https://api.openai.com/v1",
            max_output_tokens_parameter="max_completion_tokens",
        )
    )
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="openai",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=_question_contract_tool_specs(),
        )
    )

    completion_kwargs = openai_compatible_loop._completion_kwargs(payload)
    schemas = [tool["function"]["parameters"] for tool in completion_kwargs["tools"]]

    assert {
        "tool_names": [tool["function"]["name"] for tool in completion_kwargs["tools"]],
        "tool_choice": completion_kwargs["tool_choice"],
        "parallel_tool_calls": completion_kwargs["parallel_tool_calls"],
        "strict": [tool["function"]["strict"] for tool in completion_kwargs["tools"]],
        "all_object": all(schema["type"] == "object" for schema in schemas),
        "forbidden_keywords_present": [
            keyword
            for keyword in ("oneOf", "anyOf", "allOf", "not")
            if any(keyword in schema for schema in schemas)
        ],
    } == {
        "tool_names": [
            "submit_answer_request_contract",
            "submit_missing_input_clarification",
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "strict": [True, True],
        "all_object": True,
        "forbidden_keywords_present": [],
    }


def test_opencode_zen_uses_openai_compatible_tool_call_contract():
    runtime = OpenAICompatibleLoopRuntime(
        config=ChatProviderConfig(
            provider_name="opencode",
            model_name="deepseek-v4-pro",
            api_key_env_var="OPENCODE_API_KEY",
            sdk_name="openai_chat_completions",
            default_base_url="https://opencode.ai/zen/v1",
        )
    )
    payload = runtime.request_payload(
        ProviderRunRequest(
            provider="opencode",
            prompt="{}",
            max_thinking_tokens=64,
            system_prompt="system",
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=_question_contract_tool_specs(),
        )
    )

    completion_kwargs = openai_compatible_loop._completion_kwargs(payload)

    assert {
        "base_url": payload.base_url,
        "model": completion_kwargs["model"],
        "has_format": "format" in completion_kwargs,
        "has_tools": "tools" in completion_kwargs,
        "tool_choice": completion_kwargs["tool_choice"],
        "parallel_tool_calls": completion_kwargs["parallel_tool_calls"],
        "tool_names": [tool["function"]["name"] for tool in completion_kwargs["tools"]],
    } == {
        "base_url": "https://opencode.ai/zen/v1",
        "model": "deepseek-v4-pro",
        "has_format": False,
        "has_tools": True,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "tool_names": [
            "submit_answer_request_contract",
            "submit_missing_input_clarification",
        ],
    }


def test_provider_error_metadata_is_nested_under_llm_context_schema(
    fervis_foundation_reset,
    monkeypatch,
):
    def failing_worker(payload, result_queue):
        result_queue.put(
            {
                "ok": False,
                "errorClass": "BadRequestError",
                "error": "invalid model",
                "statusCode": "400",
                "errorType": "invalid_request_error",
            }
        )

    monkeypatch.setattr(anthropic_loop, "_anthropic_request_worker", failing_worker)

    class Runtime(AnthropicLoopRuntime):
        def _sdk_status(self) -> str:
            return "enabled"

    with pytest.raises(api_errors.Unavailable) as exc:
        Runtime(config=_anthropic_test_config()).run(
            ProviderRunRequest(
                provider="anthropic",
                prompt="hello",
                max_thinking_tokens=64,
                system_prompt="system",
            )
        )

    assert exc.value.context["provider_metadata"] == {
        "statusCode": "400",
        "errorType": "invalid_request_error",
    }


def test_stub_adapter_passes_contract_smoke_test(fervis_foundation_reset):
    reset_provider_backbone_for_tests()
    _register_stub_provider()

    backbone = build_provider_backbone("stub")
    payload = backbone.model_router.generate(
        provider="stub",
        system_prompt="system",
        prompt="hello",
        max_thinking_tokens=8,
    )

    assert (payload["provider"], payload["answer"], backbone.provider_name) == (
        "stub",
        "stub:hello",
        "stub",
    )
