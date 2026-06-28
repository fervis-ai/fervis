from __future__ import annotations

from ._support import *  # noqa: F401,F403

def test_fervis_project_inspect_module_entrypoint_bypasses_full_parser(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cli_main = importlib.import_module("fervis.interfaces.cli.main")
    cli_parsers = importlib.import_module("fervis.interfaces.cli.parsers")
    inspected_commands = []

    def fail_if_full_parser_is_built():
        raise AssertionError("project inspect should not build the full CLI parser")

    def record_project_command(args, *, project):
        inspected_commands.append((args, project))
        return 2

    monkeypatch.setenv("FERVIS_INVOCATION_CWD", str(tmp_path))
    monkeypatch.setattr(cli_parsers, "parser", fail_if_full_parser_is_built)
    monkeypatch.setattr(cli_main, "run_project_command", record_project_command)

    exit_code = cli_main.main(("project", "inspect"))

    assert exit_code == 2
    assert inspected_commands[0][0] == ("project", "inspect")
    assert inspected_commands[0][1].root_path == tmp_path

def test_fervis_project_inspect_returns_agent_envelope() -> None:
    stdout = StringIO()
    project = ProjectInspection(
        framework="django",
        root_path=API_DIR,
        config_path=Path("config") / "fervis.json",
        expected_config_path=Path("config") / "fervis.json",
        confidence="high",
    )

    exit_code = run_fervis(
        ("project", "inspect"),
        ports=_ports(project=project),
        stdout=stdout,
        stderr=StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload == {
        "schema": "fervis-command-result.v0.1",
        "command": "project.inspect",
        "status": "succeeded",
        "exit_code": 0,
        "project": {
            "framework": "django",
            "config_path": "config/fervis.json",
        },
        "next_actions": [],
        "payload_schema": "fervis-project-inspection.v0.1",
        "payload": {
            "framework": "django",
            "root_path": str(API_DIR),
            "config_path": "config/fervis.json",
            "expected_config_path": "config/fervis.json",
            "confidence": "high",
            "blocked_reason": None,
        },
    }

def test_fervis_project_inspect_unknown_project_is_blocked() -> None:
    stdout = StringIO()
    project = ProjectInspection(
        framework="unknown",
        root_path=Path("/tmp/no-fervis-project"),
        config_path=None,
        expected_config_path=None,
        confidence="low",
        blocked_reason="No Django or FastAPI project marker was found.",
    )

    exit_code = run_fervis(
        ("project", "inspect"),
        ports=_ports(project=project),
        stdout=stdout,
        stderr=StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["project"] == {
        "framework": "unknown",
        "config_path": None,
    }
    assert payload["payload"]["blocked_reason"] == (
        "No Django or FastAPI project marker was found."
    )

def test_fervis_inspect_prompts_is_the_prompt_viewer_execution_surface(
    monkeypatch,
) -> None:
    import fervis.observability.prompt_viewer.render_prompts as prompt_viewer

    calls = []
    monkeypatch.setattr(
        prompt_viewer,
        "render_prompt_viewer",
        lambda request, *, prompt_capture_query: (
            calls.append((request, prompt_capture_query))
            or prompt_viewer.PromptViewerResult(
                run_count=1,
                index_path=Path(".goldset-runs/prompt-viewer/latest/index.html"),
            )
        ),
    )
    stdout = StringIO()

    exit_code = run_fervis(
        ("inspect", "prompts", "--run-id", "run_1"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    envelope = _command_envelope(stdout.getvalue(), command="inspect.prompts")
    assert envelope["payload_schema"] == "fervis-prompt-inspection-result.v0.1"
    assert envelope["payload"]["output_format"] == "raw"
    assert [call[0].run_id for call in calls] == ["run_1"]
    assert [call[0].output_format for call in calls] == [
        prompt_viewer.PromptInspectionFormat.RAW
    ]
    assert all(isinstance(call[1], _PromptCaptureQuery) for call in calls)

def test_fervis_inspect_prompts_supports_explicit_text_format(monkeypatch) -> None:
    import fervis.observability.prompt_viewer.render_prompts as prompt_viewer

    monkeypatch.setattr(
        prompt_viewer,
        "render_prompt_viewer",
        lambda request, *, prompt_capture_query: prompt_viewer.PromptViewerResult(
            run_count=1,
            index_path=Path(".goldset-runs/prompt-viewer/latest/index.html"),
            output_format=request.output_format,
        ),
    )
    stdout = StringIO()

    exit_code = run_fervis(
        ("inspect", "prompts", "--run-id", "run_1", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert "Wrote 1 run(s) to .goldset-runs/prompt-viewer/latest/index.html" in (
        stdout.getvalue()
    )

def test_fervis_inspect_prompts_supports_explicit_html_format(monkeypatch) -> None:
    import fervis.observability.prompt_viewer.render_prompts as prompt_viewer

    calls = []
    monkeypatch.setattr(
        prompt_viewer,
        "render_prompt_viewer",
        lambda request, *, prompt_capture_query: (
            calls.append((request, prompt_capture_query))
            or prompt_viewer.PromptViewerResult(
                run_count=1,
                index_path=Path(".goldset-runs/prompt-viewer/latest/index.html"),
                output_format=request.output_format,
            )
        ),
    )

    exit_code = run_fervis(
        ("inspect", "prompts", "--run-id", "run_1", "--viewer-format", "html"),
        ports=_ports(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert [call[0].output_format for call in calls] == [
        prompt_viewer.PromptInspectionFormat.HTML
    ]

def test_fervis_inspect_prompts_open_requires_html_format() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("inspect", "prompts", "--run-id", "run_1", "--open"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    envelope = _blocked_envelope(stdout.getvalue(), command="inspect.prompts")
    assert (
        "--open requires --viewer-format html"
        in (envelope["payload"]["error"]["message"])
    )

def test_fervis_inspect_artifact_is_the_full_artifact_surface() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("inspect", "artifact", "artifact_parsed"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    envelope = _command_envelope(rendered, command="inspect.artifact")
    assert envelope["payload_schema"] == "fervis-artifact-content-result.v0.1"
    assert envelope["payload"]["artifact_id"] == "artifact_parsed"

    text = StringIO()
    assert (
        run_fervis(
            ("inspect", "artifact", "artifact_parsed", "--format", "text"),
            ports=_ports(),
            stdout=text,
            stderr=StringIO(),
        )
        == 0
    )
    rendered = text.getvalue()
    assert (
        "Artifact artifact_parsed (parsed_payload, application/json, 17 bytes)"
        in rendered
    )
    assert '{"answer": "parsed"}' in rendered

def test_fervis_inspect_artifact_rejects_removed_json_flag() -> None:
    with pytest.raises(SystemExit) as error:
        run_fervis(
            ("inspect", "artifact", "artifact_parsed", "--json"),
            ports=_ports(),
            stdout=StringIO(),
            stderr=StringIO(),
        )

    assert error.value.code == 2

def test_fervis_command_result_is_structured_before_text_rendering() -> None:
    result = evaluate_fervis(
        ("explain", "answer_1", "--verbose", "--format", "text"), ports=_ports()
    )

    assert result.kind is FervisCommandKind.EXPLAIN
    assert result.render_options.detail is LineageRenderDetail.VERBOSE
    assert result.payload.lineage.root_id == "answer_1"
    assert result.payload.model_calls
    assert "Question question_1" in render_fervis_result(result)
