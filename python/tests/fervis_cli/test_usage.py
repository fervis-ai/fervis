from __future__ import annotations

from ._support import *  # noqa: F401,F403

def test_fervis_usage_answer_view_uses_observability_service() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "usage",
            "answer_1",
            "--step",
            "source_binding",
            "--verbose",
            "--format",
            "text",
        ),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Usage answer answer_1" in rendered
    assert "cost total: USD 0.001500" in rendered
    assert "input_tokens: 20 tokens" in rendered
    assert "output_tokens: 5 tokens" in rendered
    assert "source_binding#1: openai/gpt-test succeeded" in rendered
    assert "chars: prompt=100, schema=50, tool_spec=75" in rendered


def test_fervis_usage_supports_additive_detail_modes() -> None:
    compact = StringIO()
    verbose = StringIO()
    debug = StringIO()

    assert (
        run_fervis(
            ("usage", "answer_1", "--format", "text"),
            ports=_ports(),
            stdout=compact,
            stderr=StringIO(),
        )
        == 0
    )
    assert (
        run_fervis(
            ("usage", "answer_1", "--verbose", "--format", "text"),
            ports=_ports(),
            stdout=verbose,
            stderr=StringIO(),
        )
        == 0
    )
    assert (
        run_fervis(
            ("usage", "answer_1", "--debug", "--format", "text"),
            ports=_ports(),
            stdout=debug,
            stderr=StringIO(),
        )
        == 0
    )

    assert "Usage answer answer_1" in compact.getvalue()
    assert "calls:" not in compact.getvalue()
    assert "calls:" in verbose.getvalue()
    assert "usage rows:" not in verbose.getvalue()
    assert "usage rows:" in debug.getvalue()


def test_fervis_usage_supports_provider_model_and_usage_filters() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "usage",
            "answer_1",
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
            "--usage-kind",
            "output_tokens",
            "--verbose",
            "--format",
            "text",
        ),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "cost total: USD 0.000080" in rendered
    assert "output_tokens: 2 tokens" in rendered
    assert "fact_planning#1: anthropic/claude-test succeeded" in rendered
    assert "source_binding#1: openai/gpt-test succeeded" not in rendered


def test_fervis_usage_default_agent_returns_command_envelope() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("usage", "answer_1"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    envelope = _command_envelope(stdout.getvalue(), command="usage")
    assert envelope["payload_schema"] == "fervis-usage-result.v0.1"
    assert envelope["payload"]["scope"] == "answer"
    assert envelope["payload"]["scope_id"] == "answer_1"
    assert envelope["payload"]["usage_totals"] == [
        {"quantity": 20, "unit": "tokens", "usage_kind": "input_tokens"},
        {"quantity": 7, "unit": "tokens", "usage_kind": "output_tokens"},
    ]


def test_fervis_usage_rejects_removed_json_flag() -> None:
    with pytest.raises(SystemExit) as error:
        run_fervis(
            ("usage", "answer_1", "--json"),
            ports=_ports(),
            stdout=StringIO(),
            stderr=StringIO(),
        )

    assert error.value.code == 2
