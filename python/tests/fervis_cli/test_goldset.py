from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from fervis.interfaces.cli.dispatch import run_fervis
from fervis.host_api.contracts.authority import ReadContextRef
from fervis.questions import AskResult

from ._support import _ports


class _Follower:
    def follow(self, result, *, event_sink=None, wait_seconds=0.0):
        return result


def test_fervis_goldset_run_loads_path_suite_and_writes_ledger(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    ledger_path = tmp_path / "goldset.jsonl"
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "goldset",
            "run",
            "--suite-path",
            str(suite_path),
            "--case-ids",
            "sales_count",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "principal_1",
            "--ledger-file",
            str(ledger_path),
        ),
        ports=_ports(question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    ledger_rows = [
        json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()
    ]
    assert exit_code == 0
    assert envelope["schema"] == "fervis-command-result.v0.1"
    assert envelope["command"] == "goldset.run"
    assert envelope["status"] == "succeeded"
    assert envelope["payload_schema"] == "fervis-goldset-run.v0.1"
    assert envelope["payload"]["passed_count"] == 1
    assert envelope["payload"]["failed_count"] == 0
    assert envelope["payload"]["cases"][0]["case_id"] == "sales_count"
    assert envelope["payload"]["cases"][0]["status"] == "passed"
    assert ledger_rows == envelope["payload"]["cases"]


def test_fervis_goldset_run_returns_failed_when_suite_oracle_fails(
    tmp_path: Path,
) -> None:
    suite_path = _write_suite(tmp_path)
    questions = _AnsweringQuestions(
        AskResult(
            status="COMPLETED",
            conversation_id="conversation_1",
            question_id="question_1",
            run_id="run_1",
            answer="not forty two",
            result_data={},
        )
    )
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "goldset",
            "run",
            "--suite-path",
            str(suite_path),
            "--case-ids",
            "sales_count",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "principal_1",
        ),
        ports=_ports(questions=questions, question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 1
    assert envelope["command"] == "goldset.run"
    assert envelope["status"] == "failed"
    assert envelope["payload"]["passed_count"] == 0
    assert envelope["payload"]["failed_count"] == 1
    assert envelope["payload"]["cases"][0]["status"] == "failed"
    assert envelope["payload"]["cases"][0]["message"] == "expected answer 42"


def test_fervis_goldset_run_accepts_comma_separated_case_ids_and_model_override(
    tmp_path: Path,
) -> None:
    suite_path = _write_suite(tmp_path)
    questions = _AnsweringQuestions(
        AskResult(
            status="COMPLETED",
            conversation_id="conversation_1",
            question_id="question_1",
            run_id="run_1",
            answer="42",
            result_data={},
        )
    )
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "goldset",
            "run",
            "--suite-path",
            str(suite_path),
            "--case-ids",
            "sales_count",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "principal_1",
            "--model",
            "anthropic:claude-haiku-4-5-20251001",
        ),
        ports=_ports(questions=questions, question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["payload"]["case_count"] == 1
    assert questions.requests[0].model_key == "anthropic:claude-haiku-4-5-20251001"
    assert questions.requests[0].provider == "anthropic"
    assert questions.requests[0].principal.read_context_ref == ReadContextRef(
        scheme="django_principal",
        key="principal_1",
    )


def test_fervis_goldset_run_submits_setup_questions_in_same_conversation(
    tmp_path: Path,
) -> None:
    suite_path = _write_suite(tmp_path, setup_questions=("Who is Nadia?",))
    questions = _ConversationQuestions()
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "goldset",
            "run",
            "--suite-path",
            str(suite_path),
            "--case-ids",
            "sales_count",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "principal_1",
        ),
        ports=_ports(questions=questions, question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["payload"]["passed_count"] == 1
    assert [request.question for request in questions.requests] == [
        "Who is Nadia?",
        "How many sales happened this month?",
    ]
    assert questions.requests[0].conversation_id == ""
    assert questions.requests[1].conversation_id == "conversation_shared"


def _write_suite(
    tmp_path: Path,
    *,
    setup_questions: tuple[str, ...] = (),
) -> Path:
    suite_path = tmp_path / "suite"
    suite_path.mkdir()
    setup_repr = repr(setup_questions)
    (suite_path / "fervis_goldset.py").write_text(
        f"""
from fervis.evaluation.goldsets import GoldsetCase, GoldsetMatch, GoldsetSuite


def load_suite():
    return GoldsetSuite(
        name="demo",
        cases=(
            GoldsetCase(
                case_id="sales_count",
                question="How many sales happened this month?",
                setup_questions={setup_repr},
            ),
        ),
        match_answer=match_answer,
    )


def match_answer(case, result):
    if result.answer == "42":
        return GoldsetMatch(passed=True, message="matched answer")
    return GoldsetMatch(passed=False, message="expected answer 42")
""".strip()
        + "\n"
    )
    return suite_path


class _AnsweringQuestions:
    def __init__(self, result: AskResult) -> None:
        self.result = result
        self.requests = []

    def ask(self, request, *, event_sink=None):
        self.requests.append(request)
        return self.result

    def continue_question(self, request, *, event_sink=None):
        raise AssertionError("goldset run should not continue questions directly")


class _ConversationQuestions:
    def __init__(self) -> None:
        self.requests = []

    def ask(self, request, *, event_sink=None):
        self.requests.append(request)
        return AskResult(
            status="COMPLETED",
            conversation_id=request.conversation_id or "conversation_shared",
            question_id=f"question_{len(self.requests)}",
            run_id=f"run_{len(self.requests)}",
            answer="42",
            result_data={},
        )

    def continue_question(self, request, *, event_sink=None):
        raise AssertionError("goldset run should not continue questions directly")
