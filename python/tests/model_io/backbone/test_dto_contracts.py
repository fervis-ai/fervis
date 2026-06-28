from fervis.model_io.backbone.dto import (
    ProviderRunChunk,
    ProviderRunRequest,
    ProviderRunResult,
    SessionRef,
    ToolDecision,
    TraceEvent,
)


def test_provider_dtos_expose_no_sdk_specific_types(fervis_foundation_reset):
    modules = [type(item).__module__.lower() for item in _provider_dtos()]

    assert [
        module
        for module in modules
        if "agents" in module or "langchain" in module
    ] == []


def _provider_dtos():
    return (
        ProviderRunRequest(
            provider="openai",
            prompt="hello",
            max_thinking_tokens=16,
            system_prompt="system",
        ),
        ProviderRunResult(
            provider="openai",
            answer="ok",
            usage={
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 1,
                "costUsd": 0.0,
            },
        ),
        ProviderRunChunk(
            event_id="evt-1", event_type="run.completed", payload={"ok": True}
        ),
        SessionRef(session_id="s1", provider_session_id="openai:s1"),
        ToolDecision(decision="approve"),
        TraceEvent(event_type="run.start", payload={"runId": "r1"}),
    )
