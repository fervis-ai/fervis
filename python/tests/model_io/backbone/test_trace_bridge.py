from fervis.model_io.backbone.factory import build_provider_backbone


def test_tracing_sensitive_data_disabled_by_default(fervis_foundation_reset):
    backbone = build_provider_backbone()
    trace_runtime = backbone.registration.trace_runtime
    trace_runtime.clear()

    backbone.trace(
        event_type="trace.test",
        payload={"token": "[REDACTED_SECRET]"},
        correlation_id="corr-1",
    )

    assert len(trace_runtime.events) == 1
    assert trace_runtime.events[0].payload["token"] == "[REDACTED_SECRET]"


def test_custom_trace_processor_receives_run_and_span_data(fervis_foundation_reset):
    backbone = build_provider_backbone()
    trace_runtime = backbone.registration.trace_runtime
    trace_runtime.clear()

    backbone.trace(
        event_type="run.start", payload={"runId": "run-1"}, correlation_id="run-1"
    )
    backbone.trace(
        event_type="span.tool", payload={"tool": "search"}, correlation_id="run-1"
    )

    assert [event.event_type for event in trace_runtime.events] == [
        "run.start",
        "span.tool",
    ]
