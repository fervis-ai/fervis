from types import SimpleNamespace

from fervis.model_io.backbone.dto import (
    ProviderOutputMode,
    ProviderRunRequest,
    ToolSpec,
)
from fervis.model_io.providers.openai_responses_adapter import (
    OPENAI_RESPONSES_PROVIDER_CONFIG,
)
from fervis.model_io.providers.openai_responses_adapter import (
    loop_adapter as responses_loop,
)
from fervis.model_io.providers.openai_responses_adapter.loop_adapter import (
    OpenAIResponsesLoopRuntime,
)


def _request() -> ProviderRunRequest:
    return ProviderRunRequest(
        provider="openai",
        prompt="Classify this input.",
        max_thinking_tokens=64,
        system_prompt="Return the contract.",
        output_mode=ProviderOutputMode.TOOL_CALL,
        tool_specs=(
            ToolSpec(
                name="submit_contract",
                description="Submit the contract.",
                input_schema={
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                    "additionalProperties": False,
                },
            ),
        ),
    )


def test_openai_responses_uses_medium_reasoning_with_one_required_strict_tool():
    payload = OpenAIResponsesLoopRuntime(
        config=OPENAI_RESPONSES_PROVIDER_CONFIG
    ).request_payload(_request())

    kwargs = responses_loop._response_kwargs(payload)

    assert kwargs["reasoning"] == {"effort": "medium"}
    assert kwargs["tool_choice"] == {
        "type": "function",
        "name": "submit_contract",
    }
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "submit_contract",
            "description": "Submit the contract.",
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    assert "temperature" not in kwargs


def test_openai_responses_decodes_one_function_call():
    response = SimpleNamespace(
        output=(
            SimpleNamespace(
                type="function_call",
                name="submit_contract",
                arguments='{"result":"ok"}',
            ),
        )
    )

    answer = responses_loop._answer_from_response(
        response,
        output_mode=ProviderOutputMode.TOOL_CALL,
        json_object_arguments={},
    )

    assert '"tool":"submit_contract"' in answer
    assert '"result":"ok"' in answer


def test_openai_responses_projects_tuple_one_of_as_supported_any_of():
    request = _request()
    spec = request.tool_specs[0]
    nested_union = {
        "type": "object",
        "properties": {
            "value_type": {
                "oneOf": (
                    {"type": "object", "properties": {}, "required": []},
                    {"type": "null"},
                )
            }
        },
        "required": ["value_type"],
        "additionalProperties": False,
    }
    request = ProviderRunRequest(
        provider=request.provider,
        prompt=request.prompt,
        max_thinking_tokens=request.max_thinking_tokens,
        system_prompt=request.system_prompt,
        output_mode=request.output_mode,
        tool_specs=(
            ToolSpec(
                name=spec.name,
                description=spec.description,
                input_schema=nested_union,
            ),
        ),
    )

    payload = OpenAIResponsesLoopRuntime(
        config=OPENAI_RESPONSES_PROVIDER_CONFIG
    ).request_payload(request)
    value_type = payload.tool_specs[0]["parameters"]["properties"]["value_type"]

    assert "oneOf" not in value_type
    assert value_type["anyOf"] == [
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        {"type": "null"},
    ]


def test_openai_responses_adds_the_type_required_for_bounded_string_enums():
    request = _request()
    spec = request.tool_specs[0]
    request = ProviderRunRequest(
        provider=request.provider,
        prompt=request.prompt,
        max_thinking_tokens=request.max_thinking_tokens,
        system_prompt=request.system_prompt,
        output_mode=request.output_mode,
        tool_specs=(
            ToolSpec(
                name=spec.name,
                description=spec.description,
                input_schema={
                    "type": "object",
                    "properties": {"input_use_ref": {"enum": ["use_1"]}},
                    "required": ["input_use_ref"],
                    "additionalProperties": False,
                },
            ),
        ),
    )

    payload = OpenAIResponsesLoopRuntime(
        config=OPENAI_RESPONSES_PROVIDER_CONFIG
    ).request_payload(request)

    assert payload.tool_specs[0]["parameters"]["properties"]["input_use_ref"] == {
        "enum": ["use_1"],
        "type": "string",
    }


def test_openai_responses_types_an_empty_enum_without_making_it_satisfiable():
    request = _request()
    spec = request.tool_specs[0]
    request = ProviderRunRequest(
        provider=request.provider,
        prompt=request.prompt,
        max_thinking_tokens=request.max_thinking_tokens,
        system_prompt=request.system_prompt,
        output_mode=request.output_mode,
        tool_specs=(
            ToolSpec(
                name=spec.name,
                description=spec.description,
                input_schema={
                    "type": "object",
                    "properties": {"unavailable_ref": {"enum": []}},
                    "required": ["unavailable_ref"],
                    "additionalProperties": False,
                },
            ),
        ),
    )

    payload = OpenAIResponsesLoopRuntime(
        config=OPENAI_RESPONSES_PROVIDER_CONFIG
    ).request_payload(request)

    assert payload.tool_specs[0]["parameters"]["properties"]["unavailable_ref"] == {
        "enum": [],
        "type": "string",
    }


def test_malformed_response_retains_usage_through_structured_output_failure(
    monkeypatch,
):
    import pytest
    from fervis.model_io.providers.chat_runtime import ConfiguredChatModelAdapter
    from fervis.model_io.structured_output.generation import generate_one_of_tool_output
    from fervis.model_io.structured_output.errors import RequiredToolOutputError

    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=40,
            output_tokens_details=SimpleNamespace(reasoning_tokens=10),
        ),
        output=(
            SimpleNamespace(
                type="function_call", name="submit_contract", arguments='{"result":'
            ),
        ),
    )
    monkeypatch.setattr(
        responses_loop,
        "openai_compatible_client",
        lambda **kwargs: SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kw: response)
        ),
    )
    runtime = OpenAIResponsesLoopRuntime(config=OPENAI_RESPONSES_PROVIDER_CONFIG)
    monkeypatch.setattr(runtime, "_sdk_status", lambda: "enabled")
    adapter = ConfiguredChatModelAdapter(
        loop_runtime=runtime, config=OPENAI_RESPONSES_PROVIDER_CONFIG
    )

    class Port:
        def generate(self, *, provider, **kwargs):
            return adapter.generate(**kwargs)

    request = _request()
    with pytest.raises(RequiredToolOutputError) as caught:
        generate_one_of_tool_output(
            model_port=Port(),
            provider="openai",
            system_prompt=request.system_prompt,
            prompt=request.prompt,
            max_thinking_tokens=request.max_thinking_tokens,
            tool_specs=request.tool_specs,
        )
    usage = caught.value.output.get("usage", {})
    assert usage.get("inputTokens") == 100
    assert usage.get("outputTokens") == 30
    assert usage.get("thinkingTokens") == 10
    assert usage.get("costUsd", 0) > 0


def test_incomplete_response_reports_exhausted_output_budget_before_tool_parsing():
    import pytest
    from fervis.model_io.providers.chat_runtime import ProviderExecutionError

    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=(),
    )
    with pytest.raises(ProviderExecutionError) as failure:
        responses_loop._answer_from_response(
            response, output_mode=ProviderOutputMode.TOOL_CALL, json_object_arguments={}
        )
    assert failure.value.error_class == "ProviderOutputLimitExceeded"
    assert failure.value.context == {
        "status": "incomplete",
        "reason": "max_output_tokens",
    }


def test_reasoning_output_budget_can_be_configured_above_8192(monkeypatch):
    monkeypatch.setenv("FERVIS_PROVIDER_MAX_OUTPUT_TOKENS", "16384")
    payload = OpenAIResponsesLoopRuntime(
        config=OPENAI_RESPONSES_PROVIDER_CONFIG
    ).request_payload(_request())
    assert payload.max_output_tokens == 16384


def test_flex_tier_is_explicit_and_does_not_change_the_default(monkeypatch):
    monkeypatch.delenv("FERVIS_OPENAI_SERVICE_TIER", raising=False)
    runtime = OpenAIResponsesLoopRuntime(config=OPENAI_RESPONSES_PROVIDER_CONFIG)
    assert "service_tier" not in responses_loop._response_kwargs(
        runtime.request_payload(_request())
    )
    monkeypatch.setenv("FERVIS_OPENAI_SERVICE_TIER", "flex")
    assert (
        responses_loop._response_kwargs(runtime.request_payload(_request()))[
            "service_tier"
        ]
        == "flex"
    )


def test_reported_flex_usage_receives_batch_rates(monkeypatch):
    from fervis.model_io.providers.chat_runtime import (
        ChatProviderConfig,
        build_provider_run_result,
    )
    from fervis.model_io.providers import chat_runtime
    from fervis.model_io.pricing import ModelPricing
    from fervis.observability.usage_types import CostSource

    monkeypatch.setattr(
        chat_runtime,
        "resolve_model_pricing",
        lambda **kwargs: ModelPricing(
            0.75, 4.5, 4.5, "test-standard", CostSource.MODELS_DEV
        ),
    )
    config = ChatProviderConfig(
        provider_name="openai",
        sdk_name="openai",
        model_name="gpt-5.4-mini",
        api_key_env_var="UNUSED",
    )
    standard = build_provider_run_result(
        config, answer="ok", input_tokens=1000, output_tokens=100, thinking_tokens=100
    )
    flex = build_provider_run_result(
        config,
        answer="ok",
        input_tokens=1000,
        output_tokens=100,
        thinking_tokens=100,
        usage_details={"serviceTier": "flex"},
    )
    assert flex.usage["costUsd"] == standard.usage["costUsd"] / 2
    assert flex.usage["serviceTier"] == "flex"
    assert flex.usage["pricingVersion"].endswith(":flex")


def test_configured_effective_prices_are_not_discounted_again():
    from fervis.model_io.providers.chat_runtime import (
        ChatProviderConfig,
        build_provider_run_result,
    )

    config = ChatProviderConfig(
        provider_name="openai",
        sdk_name="openai",
        model_name="custom-model",
        api_key_env_var="UNUSED",
        input_cost_per_million_tokens=1,
        output_cost_per_million_tokens=1,
        pricing_version="effective-rates",
    )
    result = build_provider_run_result(
        config,
        answer="ok",
        input_tokens=1000,
        output_tokens=1000,
        usage_details={"serviceTier": "flex"},
    )
    assert result.usage["costUsd"] == 0.002
