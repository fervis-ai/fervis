from __future__ import annotations

from pathlib import Path


def test_no_deleted_source_binding_escape_hatches_in_runtime_contracts():
    root = Path(__file__).resolve().parents[3]
    production_files = [
        root / "src" / "fervis" / "lookup" / "source_binding" / "prompt.py",
        root / "src" / "fervis" / "lookup" / "source_binding" / "schema.py",
        root / "src" / "fervis" / "lookup" / "source_binding" / "parser.py",
        root / "src" / "fervis" / "lookup" / "source_binding" / "turn.py",
        root
        / "src"
        / "fervis"
        / "model_io"
        / "providers"
        / "anthropic_adapter"
        / "loop_adapter.py",
        root
        / "src"
        / "fervis"
        / "model_io"
        / "providers"
        / "anthropic_adapter"
        / "source_binding_transport"
    ]
    deleted_terms = (
        "SourceInvocationSelectionTurnPrompt",
        "SourceParamValueBindingTurnPrompt",
        "build_source_invocation_selection_schema",
        "build_source_param_value_binding_schema",
        "parse_source_invocation_selection",
        "parse_source_param_value_binding",
        "assemble_split_source_binding",
        "submit_source_invocation_selection",
        "submit_source_param_value_bindings",
        "optional_param_applicability",
        "leave_unbound",
        "omit_param",
        "use_param",
        "choice_param_memberships",
        "membership_mode",
    )

    scanned_files = [
        file_path
        for path in production_files
        for file_path in ([path] if path.is_file() else path.rglob("*.py"))
    ]
    offenders = {
        str(path.relative_to(root)): [
            term for term in deleted_terms if term in path.read_text(encoding="utf-8")
        ]
        for path in scanned_files
    }
    offenders = {path: terms for path, terms in offenders.items() if terms}

    assert offenders == {}
