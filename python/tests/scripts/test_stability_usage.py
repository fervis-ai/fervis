"""Paid failures must retain usage just like successful experiment calls."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", [None, "assertion", "schema"])
def test_experiment_ledger_retains_usage_on_every_generated_call(tmp_path, monkeypatch, failure):
    path = Path(__file__).resolve().parents[3] / "scripts/run-model-step-stability.py"
    spec = importlib.util.spec_from_file_location("stability_usage_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    destination = tmp_path / "calls.jsonl"
    args = SimpleNamespace(
        runs=1, workers=1, patch_file=[], assertion_context=None,
        boundary_file=tmp_path / "boundary.json", provider=None, model_key=None,
        assertion_file=None, label="usage", output_jsonl=destination,
        max_thinking_tokens=4096, enforce_structured_stability=False,
    )
    boundary = module.ExperimentBoundary("", 1, "question_frame", "openai",
        "openai:gpt-5.4-mini", "system", "question", (), {})
    usage = {"inputTokens": 100, "outputTokens": 20, "thinkingTokens": 0, "costUsd": "0.001"}
    monkeypatch.setattr(module, "_arguments", lambda: args)
    monkeypatch.setattr(module, "_load_standalone_boundary", lambda *a, **k: boundary)
    monkeypatch.setattr(module, "_ModelPort", lambda **k: object())
    monkeypatch.setattr(module, "_load_assertion", lambda path: lambda a, c: ["wrong meaning"] if failure == "assertion" else [])

    def generate(**kwargs):
        if failure == "schema":
            error = ValueError("schema rejected generated output")
            error.output = {"usage": usage}
            error.arguments = {"value": "invalid"}
            raise error
        return SimpleNamespace(arguments={"value": "valid"}, output={"usage": usage})

    monkeypatch.setattr(module, "generate_one_of_tool_output", generate)
    assert module.main() == (1 if failure else 0)
    saved = json.loads(destination.read_text())
    assert saved["usage"] == usage
