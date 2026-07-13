from __future__ import annotations

from ._support import *  # noqa: F401,F403


def test_fervis_runtime_ask_calls_question_run_service() -> None:
    ports = _ports(
        questions=_QuestionService(
            AskResult(
                status="COMPLETED",
                conversation_id="conversation_1",
                question_id="question_1",
                run_id="run_1",
                answer="42",
                result_data={"value": 42},
            ),
            progress_events=(
                {
                    "event": "run.progress",
                    "message": "normalizing requested fact",
                    "run_id": "run_1",
                    "stage": "question_contract",
                },
                {
                    "event": "run.progress",
                    "message": "selecting source read",
                    "run_id": "run_1",
                    "stage": "source_binding",
                },
                {
                    "event": "run.progress",
                    "message": "reading source",
                    "run_id": "run_1",
                    "stage": "execute",
                },
            ),
        )
    )
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conversation_1",
            "--model",
            "anthropic:claude-haiku-4-5-20251001",
            "--idempotency-key",
            "same-key",
            "--max-budget-usd",
            "0.25",
            "--max-thinking-tokens",
            "128",
        ),
        ports=ports,
        stdout=stdout,
        stderr=StringIO(),
    )

    events = _jsonl_events(stdout.getvalue())
    request = ports.questions.requests[0]
    assert exit_code == 0
    assert events == [
        {
            "event": "run.accepted",
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "run_id": "run_1",
            "status": "RUNNING",
        },
        {
            "event": "run.progress",
            "message": "normalizing requested fact",
            "run_id": "run_1",
            "stage": "question_contract",
        },
        {
            "event": "run.progress",
            "message": "selecting source read",
            "run_id": "run_1",
            "stage": "source_binding",
        },
        {
            "event": "run.progress",
            "message": "reading source",
            "run_id": "run_1",
            "stage": "execute",
        },
        {
            "event": "run.completed",
            "answer": "42",
            "next_actions": [inspect_question_action("question_1")],
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "result_data": {"value": 42},
            "run_id": "run_1",
            "status": "COMPLETED",
        },
    ]
    assert {
        "question": request.question,
        "execution_mode": request.execution_mode.value,
        "conversation_id": request.conversation_id,
        "principal_id": request.principal.principal_id,
        "tenant_id": request.principal.tenant_id,
        "provider": request.provider,
        "model_key": request.model_key,
        "idempotency_key": request.idempotency_key,
        "max_budget_usd": str(request.max_budget_usd),
        "max_thinking_tokens": request.max_thinking_tokens,
    } == {
        "question": "How many orders came in today?",
        "execution_mode": "queued",
        "conversation_id": "conversation_1",
        "principal_id": "user_1",
        "tenant_id": "tenant_1",
        "provider": "anthropic",
        "model_key": "anthropic:claude-haiku-4-5-20251001",
        "idempotency_key": "same-key",
        "max_budget_usd": "0.25",
        "max_thinking_tokens": 128,
    }


def test_fervis_runtime_ask_rejects_model_outside_configured_policy() -> None:
    service = _QuestionService()
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--model",
            "opencode:deepseek-v4-pro",
        ),
        ports=_ports(questions=service),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    assert service.requests == []
    assert _jsonl_events(stdout.getvalue()) == [
        {
            "event": "run.invalid_request",
            "status": "INVALID_REQUEST",
            "error": {
                "code": "invalid_request",
                "message": (
                    "modelKey is not allowed by the configured Fervis model policy."
                ),
                "retryable": False,
            },
        }
    ]


def test_fervis_runtime_ask_writes_accepted_before_lookup_finishes() -> None:
    service = _BlockingEventedQuestionService()
    stdout = StringIO()
    exit_codes: list[int] = []

    thread = threading.Thread(
        target=lambda: exit_codes.append(
            run_fervis(
                (
                    "runtime",
                    "ask",
                    "How many orders came in today?",
                    "--tenant-id",
                    "tenant_1",
                    "--principal-id",
                    "user_1",
                ),
                ports=_ports(questions=service),
                stdout=stdout,
                stderr=StringIO(),
            )
        )
    )
    thread.start()
    assert service.started.wait(timeout=1)

    try:
        deadline = time.monotonic() + 1
        accepted = []
        while time.monotonic() < deadline:
            accepted = [
                event
                for event in _jsonl_events(stdout.getvalue())
                if event["event"] == "run.accepted"
            ]
            if accepted:
                break
            time.sleep(0.01)
        assert accepted == [
            {
                "event": "run.accepted",
                "conversation_id": "conversation_stream",
                "question_id": "question_stream",
                "run_id": "run_stream",
                "status": "RUNNING",
            }
        ]
    finally:
        service.release.set()
        thread.join(timeout=1)

    assert exit_codes == [0]


def test_fervis_runtime_ask_result_is_event_stream() -> None:
    result = evaluate_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
        ),
        ports=_ports(),
    )

    assert result.kind is FervisCommandKind.RUNTIME_ASK
    assert isinstance(result.payload, RuntimeAskEventStream)


def test_fervis_runtime_ask_wait_follows_queued_event_stream() -> None:
    stdout = StringIO()
    follower = _QuestionRunFollower(
        AskResult(
            status="COMPLETED",
            conversation_id="conversation_1",
            question_id="question_1",
            run_id="run_1",
            answer="42",
            result_data={"value": 42},
        ),
        progress_events=(
            {
                "event": "run.progress",
                "message": "reading source",
                "run_id": "run_1",
                "stage": "execution",
            },
        ),
    )

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conversation_1",
            "--wait",
            "60",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="QUEUED",
                    conversation_id="conversation_1",
                    question_id="question_1",
                    run_id="run_1",
                )
            ),
            question_run_follower=follower,
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert follower.calls == [("run_1", 60.0)]
    assert _jsonl_events(stdout.getvalue()) == [
        {
            "event": "run.accepted",
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "run_id": "run_1",
            "status": "QUEUED",
        },
        {
            "event": "run.progress",
            "message": "reading source",
            "run_id": "run_1",
            "stage": "execution",
        },
        {
            "event": "run.completed",
            "answer": "42",
            "next_actions": [inspect_question_action("question_1")],
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "result_data": {"value": 42},
            "run_id": "run_1",
            "status": "COMPLETED",
        },
    ]


def test_fervis_runtime_ask_wait_drops_queued_handoff_after_terminal_event() -> None:
    stdout = StringIO()
    follower = _QuestionRunFollower()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conversation_1",
            "--wait",
            "60",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="COMPLETED",
                    conversation_id="conversation_1",
                    question_id="question_1",
                    run_id="run_1",
                    answer="42",
                    result_data={"value": 42},
                ),
                progress_events=(
                    {
                        "event": "run.queued",
                        "run_id": "run_1",
                        "status": "QUEUED",
                    },
                ),
            ),
            question_run_follower=follower,
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert follower.calls == []
    assert _jsonl_events(stdout.getvalue()) == [
        {
            "event": "run.accepted",
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "run_id": "run_1",
            "status": "RUNNING",
        },
        {
            "event": "run.completed",
            "answer": "42",
            "next_actions": [inspect_question_action("question_1")],
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "result_data": {"value": 42},
            "run_id": "run_1",
            "status": "COMPLETED",
        },
    ]


def test_fervis_runtime_ask_does_not_follow_queued_run_without_wait() -> None:
    stdout = StringIO()
    follower = _QuestionRunFollower()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conversation_1",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="QUEUED",
                    conversation_id="conversation_1",
                    question_id="question_1",
                    run_id="run_1",
                )
            ),
            question_run_follower=follower,
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert follower.calls == []
    assert _jsonl_events(stdout.getvalue())[-1] == {
        "event": "run.queued",
        "next_actions": [inspect_question_action("question_1")],
        "conversation_id": "conversation_1",
        "question_id": "question_1",
        "run_id": "run_1",
        "status": "QUEUED",
    }


def test_fervis_runtime_ask_wait_timeout_keeps_queued_handoff() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conversation_1",
            "--wait",
            "0.01",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="QUEUED",
                    conversation_id="conversation_1",
                    question_id="question_1",
                    run_id="run_1",
                )
            ),
            question_run_follower=_TimedOutQuestionRunFollower(),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert _jsonl_events(stdout.getvalue()) == [
        {
            "event": "run.accepted",
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "run_id": "run_1",
            "status": "QUEUED",
        },
        {
            "event": "run.queued",
            "next_actions": [inspect_question_action("question_1")],
            "conversation_id": "conversation_1",
            "question_id": "question_1",
            "run_id": "run_1",
            "status": "QUEUED",
        },
    ]


def test_fervis_runtime_ask_wait_without_follower_is_explicit() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conversation_1",
            "--wait",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="QUEUED",
                    conversation_id="conversation_1",
                    question_id="question_1",
                    run_id="run_1",
                )
            )
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert _jsonl_events(stdout.getvalue())[-1] == {
        "event": "run.wait_unavailable",
        "message": "local wait execution is not configured",
        "next_actions": [inspect_question_action("question_1")],
        "conversation_id": "conversation_1",
        "question_id": "question_1",
        "run_id": "run_1",
        "status": "QUEUED",
    }


def test_fervis_runtime_ask_wait_failure_is_an_event() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conversation_1",
            "--wait",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="QUEUED",
                    conversation_id="conversation_1",
                    question_id="question_1",
                    run_id="run_1",
                )
            ),
            question_run_follower=_FailingQuestionRunFollower(),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 1
    assert _jsonl_events(stdout.getvalue())[-1] == {
        "event": "run.failed",
        "status": "RUNTIME_ERROR",
        "error": {
            "code": "runtime_ask_failed",
            "message": "follower unavailable",
            "retryable": False,
        },
    }


def test_fervis_runtime_ask_uses_one_public_execution_path() -> None:
    ports = _ports()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
        ),
        ports=ports,
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert ports.questions.requests[0].execution_mode.value == "queued"
    assert str(ports.questions.requests[0].max_budget_usd) == "0.5"
    assert ports.questions.requests[0].max_thinking_tokens == 64


def test_fervis_runtime_ask_rejects_invalid_limits() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--max-budget-usd",
            "0",
        ),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    events = _jsonl_events(stdout.getvalue())
    assert exit_code == 2
    assert events == [
        {
            "event": "run.invalid_request",
            "status": "INVALID_REQUEST",
            "error": {
                "code": "invalid_request",
                "message": "max_budget_usd must be greater than 0",
                "retryable": False,
            },
        }
    ]


def test_fervis_runtime_ask_rejects_non_finite_budget() -> None:
    for value in ("NaN", "Infinity"):
        stdout = StringIO()

        exit_code = run_fervis(
            (
                "runtime",
                "ask",
                "How many orders came in today?",
                "--tenant-id",
                "tenant_1",
                "--principal-id",
                "user_1",
                "--max-budget-usd",
                value,
            ),
            ports=_ports(),
            stdout=stdout,
            stderr=StringIO(),
        )

        events = _jsonl_events(stdout.getvalue())
        assert exit_code == 2
        assert events[0]["error"]["message"] == (
            "max_budget_usd must be greater than 0"
        )


def test_fervis_runtime_ask_rejects_budget_over_shared_limit() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--max-budget-usd",
            "10.01",
        ),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    events = _jsonl_events(stdout.getvalue())
    assert exit_code == 2
    assert events[0]["error"]["message"] == ("max_budget_usd must be at most 10.0")


def test_fervis_runtime_ask_uses_injected_request_limits() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--max-budget-usd",
            "0.5",
        ),
        ports=_ports(
            question_run_limits=AskRequestLimits(max_budget_usd="0.25"),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    events = _jsonl_events(stdout.getvalue())
    assert exit_code == 2
    assert events[0]["error"]["message"] == ("max_budget_usd must be at most 0.25")


def test_fervis_runtime_ask_service_failure_uses_agent_result() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
        ),
        ports=_ports(questions=_FailingQuestionService()),
        stdout=stdout,
        stderr=StringIO(),
    )

    events = _jsonl_events(stdout.getvalue())
    assert exit_code == 1
    assert events == [
        {
            "event": "run.failed",
            "status": "RUNTIME_ERROR",
            "error": {
                "code": "runtime_ask_failed",
                "message": "service unavailable",
                "retryable": False,
            },
        }
    ]


def test_fervis_runtime_ask_service_validation_failure_uses_agent_result() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
        ),
        ports=_ports(questions=_ValidationFailingQuestionService()),
        stdout=stdout,
        stderr=StringIO(),
    )

    events = _jsonl_events(stdout.getvalue())
    assert exit_code == 2
    assert events == [
        {
            "event": "run.invalid_request",
            "status": "INVALID_REQUEST",
            "error": {
                "code": "invalid_request",
                "message": "ask request question must not be empty",
                "retryable": False,
            },
        }
    ]


def test_fervis_runtime_ask_active_conflict_is_nonzero_agent_result() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="ACTIVE_RUN_CONFLICT",
                    conversation_id="conversation_cli",
                    question_id="question_active",
                    run_id="run_active",
                    active_run_id="run_active",
                    error="active_run_conflict",
                )
            )
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    events = _jsonl_events(stdout.getvalue())
    assert exit_code == 3
    assert events == [
        {
            "event": "run.active_conflict",
            "conversation_id": "conversation_cli",
            "question_id": "question_active",
            "run_id": "run_active",
            "active_run_id": "run_active",
            "status": "ACTIVE_RUN_CONFLICT",
            "error": {
                "code": "active_run_conflict",
                "message": "active_run_conflict",
                "retryable": True,
            },
            "next_actions": [inspect_question_action("question_active")],
        }
    ]


def test_fervis_runtime_ask_streams_actionable_clarification() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "What was gross profit last month?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conv_123",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="WAITING_FOR_CLARIFICATION",
                    conversation_id="conv_123",
                    question_id="q_1",
                    run_id="r_1",
                    result_data={
                        "clarifications": [
                            {
                                "id": "clar_1",
                                "question": "Which store do you mean?",
                                "options": [
                                    {"id": "store_1", "label": "ABC Mall"},
                                    {"id": "store_2", "label": "ABC Outlet"},
                                ],
                            }
                        ]
                    },
                )
            ),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert _jsonl_events(stdout.getvalue()) == [
        {
            "event": "run.accepted",
            "conversation_id": "conv_123",
            "question_id": "q_1",
            "run_id": "r_1",
            "status": "RUNNING",
        },
        {
            "event": "run.waiting_for_clarification",
            "clarifications": [
                {
                    "id": "clar_1",
                    "question": "Which store do you mean?",
                    "options": [
                        {"id": "store_1", "label": "ABC Mall"},
                        {"id": "store_2", "label": "ABC Outlet"},
                    ],
                }
            ],
            "conversation_id": "conv_123",
            "question_id": "q_1",
            "next_actions": [
                provide_clarification_action(
                    "conv_123",
                    question_id="q_1",
                    run_id="r_1",
                    clarification_id="clar_1",
                    tenant_id="tenant_1",
                    principal_id="user_1",
                )
            ],
            "run_id": "r_1",
            "status": "WAITING_FOR_CLARIFICATION",
        },
    ]


def test_fervis_runtime_ask_clarification_does_not_depend_on_explain_lineage() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "What was gross profit last month?",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conv_123",
        ),
        ports=_ports(
            questions=_QuestionService(
                AskResult(
                    status="WAITING_FOR_CLARIFICATION",
                    conversation_id="conv_123",
                    question_id="q_1",
                    run_id="r_1",
                    result_data={
                        "clarifications": [
                            {
                                "id": "clar_1",
                                "question": "Which store do you mean?",
                                "options": [
                                    {"id": "store_1", "label": "ABC Mall"},
                                    {"id": "store_2", "label": "ABC Outlet"},
                                ],
                            }
                        ]
                    },
                )
            ),
            lineage_query=_ExplodingLineageQuery(),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert _jsonl_events(stdout.getvalue()) == [
        {
            "event": "run.accepted",
            "conversation_id": "conv_123",
            "question_id": "q_1",
            "run_id": "r_1",
            "status": "RUNNING",
        },
        {
            "event": "run.waiting_for_clarification",
            "clarifications": [
                {
                    "id": "clar_1",
                    "question": "Which store do you mean?",
                    "options": [
                        {"id": "store_1", "label": "ABC Mall"},
                        {"id": "store_2", "label": "ABC Outlet"},
                    ],
                }
            ],
            "conversation_id": "conv_123",
            "question_id": "q_1",
            "next_actions": [
                provide_clarification_action(
                    "conv_123",
                    question_id="q_1",
                    run_id="r_1",
                    clarification_id="clar_1",
                    tenant_id="tenant_1",
                    principal_id="user_1",
                )
            ],
            "run_id": "r_1",
            "status": "WAITING_FOR_CLARIFICATION",
        },
    ]


def test_fervis_runtime_ask_streams_clarification_response_followup() -> None:
    stdout = StringIO()
    questions = _QuestionService(
        AskResult(
            status="COMPLETED",
            conversation_id="conv_123",
            question_id="q_1",
            run_id="r_2",
            answer="42",
            result_data={"value": 42},
        ),
        progress_events=(
            {
                "event": "run.progress",
                "message": "reading source",
                "run_id": "r_2",
                "stage": "execute",
            },
        ),
        accepted_trigger={
            "kind": "clarification_response",
            "run_id": "r_1",
            "clarification_id": "clar_1",
        },
    )

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "ABC Mall",
            "--tenant-id",
            "tenant_1",
            "--principal-id",
            "user_1",
            "--conversation-id",
            "conv_123",
            "--question-id",
            "q_1",
            "--run-id",
            "r_1",
            "--clarification-id",
            "clar_1",
        ),
        ports=_ports(
            questions=questions,
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert questions.requests == []
    assert len(questions.continue_requests) == 1
    continued = questions.continue_requests[0]
    assert continued.question_id == "q_1"
    assert continued.run_id == "r_1"
    assert continued.clarification_id == "clar_1"
    assert _jsonl_events(stdout.getvalue()) == [
        {
            "event": "run.accepted",
            "conversation_id": "conv_123",
            "question_id": "q_1",
            "run_id": "r_2",
            "status": "RUNNING",
            "trigger": {
                "kind": "clarification_response",
                "run_id": "r_1",
                "clarification_id": "clar_1",
            },
        },
        {
            "event": "run.progress",
            "message": "reading source",
            "run_id": "r_2",
            "stage": "execute",
        },
        {
            "event": "run.completed",
            "answer": "42",
            "next_actions": [inspect_question_action("q_1")],
            "conversation_id": "conv_123",
            "question_id": "q_1",
            "result_data": {"value": 42},
            "run_id": "r_2",
            "status": "COMPLETED",
        },
    ]
