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
            "--determinism-runs",
            "2",
        ),
        ports=_ports(questions=questions, question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["payload"]["case_count"] == 1
    assert len(questions.requests) == 2
    assert questions.requests[0].model_key == "anthropic:claude-haiku-4-5-20251001"
    assert questions.requests[0].provider == "anthropic"
    assert questions.requests[0].principal.read_context_ref == ReadContextRef(
        scheme="django_principal",
        key="principal_1",
    )


def test_fervis_goldset_run_loads_import_entrypoint_suite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "demo_goldsets"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "suite.py").write_text(
        """
from fervis.evaluation.goldsets import GoldsetCase, GoldsetMatch, GoldsetSuite


def load_suite():
    return GoldsetSuite(
        name="imported",
        cases=(GoldsetCase(case_id="sales_count", question="Question?"),),
        match_answer=lambda case, result: GoldsetMatch(passed=True),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "goldset",
            "run",
            "--suite",
            "demo_goldsets.suite:load_suite",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "principal_1",
        ),
        ports=_ports(question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["payload"]["suite_name"] == "imported"
    assert envelope["payload"]["passed_count"] == 1


def test_fervis_goldset_run_uses_env_suite_case_ids_and_tenant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_path = _write_suite(tmp_path)
    monkeypatch.setenv("FERVIS_GOLDSET_SUITE", str(suite_path))
    monkeypatch.setenv("FERVIS_GOLDSET_CASE_IDS", "sales_count")
    monkeypatch.setenv("FERVIS_GOLDSET_TENANT_ID", "tenant_from_env")
    monkeypatch.setenv("FERVIS_GOLDSET_PRINCIPAL_ID", "principal_from_env")
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
        ("goldset", "run"),
        ports=_ports(questions=questions, question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert questions.requests[0].principal.tenant_id == "tenant_from_env"
    assert questions.requests[0].principal.principal_id == "principal_from_env"


def test_fervis_goldset_run_preflights_before_model_calls(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite"
    suite_path.mkdir()
    (suite_path / "fervis_goldset.py").write_text(
        """
from fervis.evaluation.goldsets import GoldsetCase, GoldsetMatch, GoldsetSuite


def load_suite():
    return GoldsetSuite(
        name="demo",
        cases=(GoldsetCase(case_id="sales_count", question="Question?"),),
        match_answer=lambda case, result: GoldsetMatch(passed=True),
        preflight=lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )
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
    assert exit_code == 2
    assert questions.requests == []
    assert envelope["status"] == "blocked"
    assert (
        envelope["payload"]["error"]["message"]
        == "goldset preflight failed: database unavailable"
    )


def test_fervis_goldset_run_uses_env_principal_id(
    tmp_path: Path,
    monkeypatch,
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
    monkeypatch.setenv("FERVIS_GOLDSET_PRINCIPAL_ID", "principal_from_env")
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
        ),
        ports=_ports(questions=questions, question_run_follower=_Follower()),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert questions.requests[0].principal.principal_id == "principal_from_env"
    assert questions.requests[0].principal.read_context_ref == ReadContextRef(
        scheme="django_principal",
        key="principal_from_env",
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


def test_fervis_goldset_run_enforces_duration_ratio_to_setup(
    tmp_path: Path,
) -> None:
    suite_path = _write_suite(
        tmp_path,
        setup_questions=("How many sales happened today?",),
        metadata={"max_duration_ratio_to_setup": 0.5},
    )
    questions = _TimedConversationQuestions(
        setup_duration_ms=1000,
        target_duration_ms=500,
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
    case = envelope["payload"]["cases"][0]
    assert exit_code == 1
    assert case["status"] == "failed"
    assert case["duration_ms"] == 500
    assert case["details"]["failure_class"] == "runtime_duration_too_slow"
    assert "Expected run duration 500ms" in case["message"]


def test_fervis_goldset_run_answers_declared_clarifications() -> None:
    questions = _ClarifyingQuestions()
    suite = _suite(
        clarification_answers=("BBS Mall",),
    )

    result = _run_suite(suite, questions=questions)

    assert result.passed_count == 1
    assert [request.question for request in questions.requests] == [
        "How many sales happened this month?",
        "BBS Mall",
    ]
    continuation = questions.requests[1]
    assert continuation.base_run_id == "run_clarification"
    assert continuation.trigger_clarification_response_id == "clarification_1"
    assert continuation.runtime_context["caseId"] == "sales_count"


def test_fervis_goldset_run_repeats_independent_determinism_runs() -> None:
    questions = _DeterminismQuestions()

    result = _run_suite(
        _suite(),
        questions=questions,
        determinism_runs=2,
    )

    assert result.passed_count == 1
    assert len(questions.requests) == 2
    contexts = [request.runtime_context for request in questions.requests]
    assert [context["determinismRun"] for context in contexts] == ["1", "2"]
    assert {context["determinismRuns"] for context in contexts} == {"2"}
    assert len({context["goldsetRunId"] for context in contexts}) == 1
    assert len(result.cases[0].details["determinism_runs"]) == 2


def test_fervis_goldset_run_fails_mixed_determinism_outcomes() -> None:
    questions = _DeterminismQuestions(answers=("42", "41"))

    result = _run_suite(
        _suite(),
        questions=questions,
        determinism_runs=2,
    )

    assert result.failed_count == 1
    assert result.cases[0].details["failure_class"] == "nondeterminism"
    assert result.cases[0].message == "1/2 determinism runs passed."


def test_fervis_goldset_run_can_require_identical_structured_results() -> None:
    questions = _DeterminismQuestions(result_values=("42", "42.0"))

    result = _run_suite(
        _suite(),
        questions=questions,
        determinism_runs=2,
        enforce_structured_determinism=True,
    )

    assert result.failed_count == 1
    assert result.cases[0].details["failure_class"] == "nondeterminism"
    assert "structured results differed" in result.cases[0].message


def test_fervis_goldset_run_retries_provider_failures() -> None:
    questions = _RetryingQuestions()

    result = _run_suite(
        _suite(),
        questions=questions,
        attempts=2,
        retry_provider_failures=True,
        retry_sleep_seconds=0,
    )

    assert result.passed_count == 1
    assert len(questions.requests) == 2


def _write_suite(
    tmp_path: Path,
    *,
    setup_questions: tuple[str, ...] = (),
    metadata: dict[str, object] | None = None,
) -> Path:
    suite_path = tmp_path / "suite"
    suite_path.mkdir()
    setup_repr = repr(setup_questions)
    metadata_repr = repr(metadata or {})
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
                metadata={metadata_repr},
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


def _suite(*, clarification_answers: tuple[str, ...] = ()):
    from fervis.evaluation.goldsets import GoldsetCase, GoldsetMatch, GoldsetSuite

    return GoldsetSuite(
        name="demo",
        cases=(
            GoldsetCase(
                case_id="sales_count",
                question="How many sales happened this month?",
                clarification_answers=clarification_answers,
            ),
        ),
        match_answer=lambda case, result: GoldsetMatch(
            passed=result.answer == "42",
            message="matched" if result.answer == "42" else "expected answer 42",
        ),
    )


def _run_suite(suite, *, questions, **kwargs):
    from fervis.evaluation.goldsets.runner import run_goldset_suite
    from fervis.interfaces.common.admission import ConfiguredModelPolicy
    from fervis.questions import AskRequestLimits, QuestionPrincipal

    return run_goldset_suite(
        suite,
        questions=questions,
        question_run_follower=_Follower(),
        question_run_limits=AskRequestLimits(),
        model_policy=ConfiguredModelPolicy(
            default_provider="anthropic",
            default_model_key="claude-haiku-4-5-20251001",
            allowed_model_keys_by_provider={
                "anthropic": frozenset({"claude-haiku-4-5-20251001"})
            },
        ),
        principal=QuestionPrincipal(
            principal_id="principal_1",
            tenant_id="tenant_1",
        ),
        **kwargs,
    )


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


class _TimedConversationQuestions:
    def __init__(
        self,
        *,
        setup_duration_ms: int,
        target_duration_ms: int,
    ) -> None:
        self.requests = []
        self.durations = (setup_duration_ms, target_duration_ms)

    def ask(self, request, *, event_sink=None):
        self.requests.append(request)
        index = len(self.requests)
        return AskResult(
            status="COMPLETED",
            conversation_id=request.conversation_id or "conversation_shared",
            question_id=f"question_{index}",
            run_id=f"run_{index}",
            answer="42",
            result_data={},
            duration_ms=self.durations[index - 1],
        )

    def continue_question(self, request, *, event_sink=None):
        raise AssertionError("goldset run should not continue questions directly")


class _ClarifyingQuestions:
    def __init__(self) -> None:
        self.requests = []

    def ask(self, request, *, event_sink=None):
        self.requests.append(request)
        return AskResult(
            status="NEEDS_CLARIFICATION",
            conversation_id="conversation_1",
            question_id="question_1",
            run_id="run_clarification",
            answer="Which store?",
            result_data={
                "details": {"clarifications": [{"id": "clarification_1"}]}
            },
        )

    def continue_question(self, request, *, event_sink=None):
        self.requests.append(request)
        return AskResult(
            status="COMPLETED",
            conversation_id="conversation_1",
            question_id="question_1",
            run_id="run_answer",
            answer="42",
            result_data={"kind": "answer", "scalars": {"value": 42}},
        )


class _DeterminismQuestions:
    def __init__(
        self,
        *,
        answers: tuple[str, ...] = ("42", "42"),
        result_values: tuple[str, ...] = ("42", "42"),
    ) -> None:
        self.requests = []
        self.answers = answers
        self.result_values = result_values

    def ask(self, request, *, event_sink=None):
        self.requests.append(request)
        index = len(self.requests)
        answer = self.answers[index - 1]
        result_value = self.result_values[index - 1]
        return AskResult(
            status="COMPLETED",
            conversation_id=f"conversation_{index}",
            question_id=f"question_{index}",
            run_id=f"run_{index}",
            answer=answer,
            result_data={"kind": "answer", "scalars": {"value": result_value}},
        )

    def continue_question(self, request, *, event_sink=None):
        raise AssertionError("determinism case should not continue")


class _RetryingQuestions(_DeterminismQuestions):
    def ask(self, request, *, event_sink=None):
        self.requests.append(request)
        if len(self.requests) == 1:
            return AskResult(
                status="FAILED",
                conversation_id="conversation_failed",
                question_id="question_failed",
                run_id="run_failed",
                error="provider_timeout",
            )
        return AskResult(
            status="COMPLETED",
            conversation_id="conversation_2",
            question_id="question_2",
            run_id="run_2",
            answer="42",
            result_data={"kind": "answer", "scalars": {"value": 42}},
        )
