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


def test_openai_responses_uses_low_reasoning_with_one_required_strict_tool():
    payload = OpenAIResponsesLoopRuntime(
        config=OPENAI_RESPONSES_PROVIDER_CONFIG
    ).request_payload(_request())

    kwargs = responses_loop._response_kwargs(payload)

    assert kwargs["reasoning"] == {"effort": "low"}
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
