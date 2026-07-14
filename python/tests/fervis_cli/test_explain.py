from __future__ import annotations

from ._support import *  # noqa: F401,F403


def test_fervis_explain_default_agent_view_is_structured_and_compact() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    envelope = _command_envelope(stdout.getvalue(), command="explain")
    assert envelope["payload_schema"] == "fervis-explain-result.v0.1"
    payload = envelope["payload"]
    assert payload["view_kind"] == "lineage"
    assert "kind" not in payload
    assert "payload" not in payload
    assert payload["detail"] == "compact"
    assert payload["root"] == {"kind": "answer", "id": "answer_1"}
    assert payload["filters"] == {
        "answer_output": None,
        "errors_only": False,
        "fact": None,
        "step": None,
    }
    assert payload["summary"] == {
        "answer_output_count": 1,
        "model_call_count": 1,
        "question_count": 1,
        "run_count": 1,
        "runtime_error_count": 0,
        "source_read_count": 1,
        "step_count": 4,
    }
    assert payload["index"]["answer_outputs"][0]["value"] == "staff:staff_id=staff_9393"
    assert payload["index"]["answer_outputs"][0]["step_key"] == "source_binding"
    assert payload["index"]["source_reads"][0]["endpoint"] == (
        "retail_ops/list_shift_compensation_list"
    )
    assert payload["index"]["source_reads"][0]["row_count"] == 3
    assert "model_calls" not in payload["index"]
    question = payload["questions"][0]
    assert question["question_id"] == "question_1"
    assert question["text"] == "Which staff earned the most this month?"
    run = question["runs"][0]
    assert run["run_id"] == "run_1"
    assert run["result_kind"] == "answered"
    source_binding = _agent_step(run, "source_binding")
    assert source_binding["step_key"] == "source_binding"
    assert [decision["detail"] for decision in source_binding["decisions"]] == [
        "compact"
    ]
    execute = _agent_step(run, "execute")
    assert execute["source_reads"][0]["endpoint"] == (
        "retail_ops/list_shift_compensation_list"
    )
    assert "catalog_endpoint" not in execute["source_reads"][0]
    assert "model_calls" not in source_binding


def test_fervis_explain_answer_compact_view_is_signal_first() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Question question_1: Which staff earned the most this month?" in rendered
    assert "Answer output answer_1: entity staff:staff_id=staff_9393" in rendered
    assert (
        "Answer presentation (default/text): "
        "Staff staff_9393 earned the most compensation."
    ) in rendered
    assert "Explicit inputs: June 2026" in rendered
    assert "Derived inputs: month=2026-06" in rendered
    assert (
        "      Source read: retail_ops/list_shift_compensation_list rows=3" in rendered
    )
    assert "hash=sha256:source" in rendered
    assert "    Step 1: question_contract" in rendered
    assert "    Step 2: source_binding" in rendered
    assert "    Step 3: render" in rendered
    assert "    Step 9000: execute" in rendered
    assert "payload_json" not in rendered
    assert "{" not in rendered


def test_fervis_explain_verbose_view_shows_step_decision_basis() -> None:
    compact = StringIO()
    verbose = StringIO()

    assert (
        run_fervis(
            ("explain", "answer_1", "--step", "source_binding", "--format", "text"),
            ports=_ports(),
            stdout=compact,
            stderr=StringIO(),
        )
        == 0
    )
    assert (
        run_fervis(
            (
                "explain",
                "answer_1",
                "--step",
                "source_binding",
                "--verbose",
                "--format",
                "text",
            ),
            ports=_ports(),
            stdout=verbose,
            stderr=StringIO(),
        )
        == 0
    )

    assert "Population basis:" not in compact.getvalue()
    rendered = verbose.getvalue()
    assert "Source binding source_6: USE_SOURCE for fact_1" in rendered
    assert (
        "Population basis: Shift compensation rows match the staff population "
        "for this month."
    ) in rendered
    assert (
        "Fulfillment basis answer_1/choice_staff_id: staff_id is the canonical "
        "returned staff identity."
    ) in rendered
    assert (
        "calculated_pay: row-level payroll amount -> fits_requested_answer" in rendered
    )


def test_fervis_explain_supports_run_id_root() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--run-id", "run_1", "--verbose", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Question question_1: Which staff earned the most this month?" in rendered
    assert "Run run_1 (#1): answered" in rendered
    assert "Run run_2 (#2): runtime_error" not in rendered


def test_fervis_explain_inputs_view_is_user_readable() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--inputs", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Inputs used for fact_1" in rendered
    assert "Fact: staff member who earned the most compensation" in rendered
    assert "Explicit\n- June 2026" in rendered
    assert "Derived\n- month=2026-06" in rendered
    assert (
        "Applied in execution\n- month=2026-06 was used as an endpoint argument."
        in rendered
    )
    assert "payload_json" not in rendered
    assert "{" not in rendered


def test_fervis_explain_inputs_default_agent_returns_structured_view() -> None:
    result = evaluate_fervis(("explain", "answer_1", "--inputs"), ports=_ports())

    payload = _command_payload(render_fervis_result(result), command="explain")

    assert payload["view_kind"] == "input_lineage"
    assert payload["detail"] == "compact"
    assert payload["input_lineage"]["root"] == {"kind": "answer", "id": "answer_1"}
    assert payload["input_lineage"]["results"][0]["explicit"] == ["June 2026"]
    assert payload["input_lineage"]["results"][0]["derived"] == ["month=2026-06"]
    assert payload["input_lineage"]["results"][0]["applied"] == [
        "month=2026-06 was used as an endpoint argument."
    ]
    assert "evidence_refs" not in payload["input_lineage"]["results"][0]
    assert "proof_handles" not in payload["input_lineage"]["results"][0]


def test_fervis_explain_inputs_detail_modes_are_additive() -> None:
    compact = StringIO()
    verbose = StringIO()
    debug = StringIO()

    assert (
        run_fervis(
            ("explain", "answer_1", "--inputs", "--format", "text"),
            ports=_ports(),
            stdout=compact,
            stderr=StringIO(),
        )
        == 0
    )
    assert (
        run_fervis(
            ("explain", "answer_1", "--inputs", "--verbose", "--format", "text"),
            ports=_ports(),
            stdout=verbose,
            stderr=StringIO(),
        )
        == 0
    )
    assert (
        run_fervis(
            ("explain", "answer_1", "--inputs", "--debug", "--format", "text"),
            ports=_ports(),
            stdout=debug,
            stderr=StringIO(),
        )
        == 0
    )

    assert "Evidence" not in compact.getvalue()
    assert "Evidence\n- known_input:month_1" in verbose.getvalue()
    assert "Proof handles" not in verbose.getvalue()
    assert "Proof handles\n- answer_output:fact_1:answer_1" in debug.getvalue()


def test_fervis_explain_agent_inputs_detail_modes_are_additive() -> None:
    compact = _command_payload(
        render_fervis_result(
            evaluate_fervis(("explain", "answer_1", "--inputs"), ports=_ports())
        ),
        command="explain",
    )
    verbose = _command_payload(
        render_fervis_result(
            evaluate_fervis(
                ("explain", "answer_1", "--inputs", "--verbose"), ports=_ports()
            )
        ),
        command="explain",
    )
    debug = _command_payload(
        render_fervis_result(
            evaluate_fervis(
                ("explain", "answer_1", "--inputs", "--debug"), ports=_ports()
            )
        ),
        command="explain",
    )

    compact_result = compact["input_lineage"]["results"][0]
    verbose_result = verbose["input_lineage"]["results"][0]
    debug_result = debug["input_lineage"]["results"][0]
    assert "evidence_refs" not in compact_result
    assert "proof_handles" not in compact_result
    assert verbose_result["evidence_refs"] == ["known_input:month_1"]
    assert "proof_handles" not in verbose_result
    assert "answer_output:fact_1:answer_1" in debug_result["proof_handles"]


def test_fervis_explain_agent_detail_modes_are_additive() -> None:
    compact = _command_payload(
        render_fervis_result(evaluate_fervis(("explain", "answer_1"), ports=_ports())),
        command="explain",
    )
    verbose = _command_payload(
        render_fervis_result(
            evaluate_fervis(("explain", "answer_1", "--verbose"), ports=_ports())
        ),
        command="explain",
    )
    debug = _command_payload(
        render_fervis_result(
            evaluate_fervis(("explain", "answer_1", "--debug"), ports=_ports())
        ),
        command="explain",
    )

    compact_step = compact["questions"][0]["runs"][0]["steps"][1]
    verbose_step = verbose["questions"][0]["runs"][0]["steps"][1]
    debug_output = _agent_step(debug["questions"][0]["runs"][0], "source_binding")[
        "answer_outputs"
    ][0]

    assert "model_calls" not in compact_step
    assert verbose_step["model_call_ids"] == ["call_1"]
    assert verbose["index"]["model_calls"][0]["model_call_id"] == "call_1"
    assert verbose["index"]["model_calls"][0]["artifacts"][0]["artifact_id"] == (
        "artifact_prompt"
    )
    assert [decision["detail"] for decision in verbose_step["decisions"]] == [
        "compact",
        "verbose",
    ]
    assert "endpoint_args" in debug_output["proof"]
    verbose_output = _agent_step(verbose["questions"][0]["runs"][0], "source_binding")[
        "answer_outputs"
    ][0]
    assert "handle" not in verbose_output["proof"]["endpoint_args"][0]
    assert "handle" in debug_output["proof"]["endpoint_args"][0]
    assert "debug_evidence_handles" in debug_output["proof"]
    assert "relation:source_1" in debug_output["proof"]["debug_evidence_handles"]


def test_fervis_explain_verbose_view_shows_proof_graph_without_raw_payload_dump() -> (
    None
):
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--verbose", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Proof record: proof_1" in rendered
    assert "Evidence used:" in rendered
    assert "applied month" in rendered
    assert "Computation:" in rendered
    assert "source rows were used as computation input" in rendered
    assert "derived rows produced the answer output" in rendered
    assert "Model calls:" in rendered
    assert "source_binding#1: openai/gpt-test succeeded" in rendered
    assert "prompt: artifact_prompt size=11" in rendered
    assert "parsed_payload: artifact_parsed size=17" in rendered
    assert "Nodes:" not in rendered
    assert "Edges:" not in rendered
    assert "proof nodes" not in rendered
    assert "operation:op_1" not in rendered
    assert "relation:source_1" not in rendered
    assert "payload_json" not in rendered


def test_fervis_explain_debug_view_shows_complete_formatted_audit_details() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--debug", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Step 1: question_contract" in rendered
    assert "Step 2: source_binding" in rendered
    assert "Step 9000: execute" in rendered
    assert "Proof details for answer_1: proof_1" in rendered
    assert "Evidence handles:" in rendered
    assert "- relation:source_1" in rendered
    assert "Computation links:" in rendered
    assert "operation:op_1 -> answer_output:fact_1:answer_1 (produces)" in rendered
    assert "payload_json" not in rendered


def test_fervis_explain_rejects_ambiguous_roots() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--question-id", "question_1"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    envelope = _blocked_envelope(stdout.getvalue(), command="explain")
    assert "provide exactly one root" in envelope["payload"]["error"]["message"]


def test_fervis_explain_clarifications_include_followup_next_actions() -> None:
    stdout = StringIO()
    dataset = _lineage_dataset()
    dataset["clarification_requests"] = [
        {
            "clarification_id": "clar_1",
            "run_id": "run_1",
            "need": "target_reference",
            "reason": "multiple_matching_entities",
            "payload_json": {
                "id": "clar_1",
                "need": "target_reference",
                "reason": "multiple_matching_entities",
                "requestedFactId": "fact_1",
                "question": "Which store do you mean?",
                "subjects": [
                    {
                        "kind": "question_input",
                        "id": "store",
                        "label": "store",
                        "sourceText": "store",
                        "options": [
                            {"id": "store_1", "label": "ABC Mall"},
                            {"id": "store_2", "label": "ABC Outlet"},
                        ],
                    }
                ],
                "evidence": [
                    {"kind": "resolver_read", "id": "source_read:store_lookup"}
                ],
            },
            "fact_result_id": "fact_result_1",
            "step_id": "step_source_binding",
        }
    ]

    exit_code = run_fervis(
        ("explain", "--run-id", "run_1"),
        ports=_ports(lineage_query=fixture_lineage_query(dataset)),
        stdout=stdout,
        stderr=StringIO(),
    )

    payload = _command_payload(stdout.getvalue(), command="explain")
    clarification = payload["questions"][0]["runs"][0]["steps"][1]["clarifications"][0]
    assert exit_code == 0
    assert clarification["next_actions"] == [
        provide_clarification_action(
            "conversation_1",
            question_id="question_1",
            run_id="run_1",
            clarification_id="clar_1",
        )
    ]


def test_fervis_explain_question_errors_filters_to_runtime_error_runs() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--question-id", "question_1", "--errors", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Run run_2 (#2): runtime_error" in rendered
    assert "Run run_1 (#1): answered" not in rendered


def test_fervis_explain_errors_view_preserves_answered_only_root() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--question-id", "question_2", "--errors", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert "Question question_2: How many stores are open?" in stdout.getvalue()
    assert "No runtime errors." in stdout.getvalue()
    assert "errors_filter_no_runtime_error_runs" in stdout.getvalue()


def test_fervis_explain_errors_notice_names_filtered_terminal_runs() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--question-id", "question_2", "--errors"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    payload = _command_payload(stdout.getvalue(), command="explain")
    notices = payload["observability_notices"]
    assert notices == [
        {
            "details": {"result_kinds_by_run_id": {"run_3": "factual_terminal"}},
            "kind": "errors_filter_no_runtime_error_runs",
            "message": (
                "--errors shows only runs whose result_kind is runtime_error. "
                "The selected lineage has no runtime_error runs; run without --errors "
                "to inspect terminal facts, clarifications, source reads, and model calls."
            ),
            "run_ids": ["run_3"],
            "severity": "info",
        }
    ]


def test_fervis_explain_notice_names_missing_model_call_audit_rows() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--question-id", "question_2"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    payload = _command_payload(stdout.getvalue(), command="explain")
    notices = payload["observability_notices"]
    assert [notice["kind"] for notice in notices] == ["missing_model_call_audits"]
    assert notices[0]["run_ids"] == ["run_3"]
    assert "prompts, schemas, raw outputs, and parsed payloads" in notices[0]["message"]


def test_fervis_explain_rejects_scope_flags_that_do_not_apply_to_root() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--run", "run_1"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    envelope = _blocked_envelope(stdout.getvalue(), command="explain")
    assert (
        "--run can only scope --question-id or --conversation-id"
        in (envelope["payload"]["error"]["message"])
    )


def test_fervis_explain_rejects_conflicting_conversation_scopes() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "explain",
            "--conversation-id",
            "conversation_1",
            "--question",
            "question_1",
            "--run",
            "run_1",
        ),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    envelope = _blocked_envelope(stdout.getvalue(), command="explain")
    assert (
        "use either --question or --run, not both"
        in (envelope["payload"]["error"]["message"])
    )


def test_fervis_explain_default_agent_returns_structured_lineage_view() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--verbose"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    payload = _command_payload(stdout.getvalue(), command="explain")
    assert payload["view_kind"] == "lineage"
    assert payload["root"] == {"kind": "answer", "id": "answer_1"}
    assert payload["questions"][0]["runs"][0]["run_id"] == "run_1"
    step = payload["questions"][0]["runs"][0]["steps"][1]
    assert step["model_call_ids"] == ["call_1"]
    assert payload["index"]["model_calls"][0]["model_call_id"] == "call_1"
    assert "content" not in payload["index"]["model_calls"][0]["artifacts"][0]
    assert payload["detail"] == "verbose"


def test_fervis_explain_conversation_can_scope_to_question() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "explain",
            "--conversation-id",
            "conversation_1",
            "--question",
            "question_1",
            "--format",
            "text",
        ),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert (
        "Question question_1: Which staff earned the most this month?"
        in stdout.getvalue()
    )


def test_fervis_explain_rejects_child_question_outside_conversation_scope() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--conversation-id", "conversation_1", "--question", "question_2"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    envelope = _blocked_envelope(stdout.getvalue(), command="explain")
    assert (
        "question 'question_2' is not in conversation 'conversation_1'"
        in (envelope["payload"]["error"]["message"])
    )


def test_fervis_explain_rejects_child_run_outside_question_scope() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--question-id", "question_1", "--run", "run_3"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    envelope = _blocked_envelope(stdout.getvalue(), command="explain")
    assert (
        "run 'run_3' is not in question 'question_1'"
        in (envelope["payload"]["error"]["message"])
    )


def test_fervis_explain_agent_uses_same_step_slice_as_text() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--step", "source_binding"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    payload = _command_payload(stdout.getvalue(), command="explain")
    steps = payload["questions"][0]["runs"][0]["steps"]
    assert [step["step_key"] for step in steps] == ["source_binding"]


def test_fervis_explain_agent_recursively_applies_step_slice() -> None:
    dataset = _lineage_dataset()
    dataset["steps"] = [
        *dataset["steps"],
        {
            "step_id": "step_compile",
            "run_id": "run_1",
            "sequence": 3,
            "step_key": "compile",
            "kind": "deterministic",
        },
        {
            "step_id": "step_execute",
            "run_id": "run_1",
            "sequence": 4,
            "step_key": "execute",
            "kind": "deterministic",
        },
    ]
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--step", "source_binding"),
        ports=FervisCliPorts(
            lineage_query=fixture_lineage_query(dataset),
            observability_query=_ObservabilityQuery(),
            prompt_capture_query=_PromptCaptureQuery(),
            questions=_QuestionService(),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    payload = _command_payload(stdout.getvalue(), command="explain")
    step = payload["questions"][0]["runs"][0]["steps"][0]
    assert step["step_key"] == "source_binding"
    assert [fact["requested_fact_id"] for fact in step["requested_facts"]] == ["fact_1"]
    assert [result["fact_result_id"] for result in step["fact_results"]] == [
        "fact_result_1"
    ]


def test_fervis_explain_agent_errors_slice_returns_only_error_runs() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--question-id", "question_1", "--errors"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    payload = _command_payload(stdout.getvalue(), command="explain")
    runs = payload["questions"][0]["runs"]
    assert [(run["run_id"], run["result_kind"]) for run in runs] == [
        ("run_2", "runtime_error")
    ]


def test_fervis_explain_rejects_missing_answer_output_slice() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "answer_1", "--answer-output", "missing_output"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 2
    envelope = _blocked_envelope(stdout.getvalue(), command="explain")
    assert (
        "answer output 'missing_output' is not in lineage view"
        in (envelope["payload"]["error"]["message"])
    )


def test_fervis_explain_error_view_includes_runtime_error_detail() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("explain", "--question-id", "question_1", "--errors", "--format", "text"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    rendered = stdout.getvalue()
    assert "Run run_2 (#2): runtime_error" in rendered
    assert "provider_runtime_failed: provider timed out" in rendered
    assert "failed step: source_binding" in rendered


def test_lineage_proof_inputs_use_persisted_proof_values_not_source_read_suffixes() -> (
    None
):
    dataset = _lineage_dataset()
    dataset["source_reads"] = [
        {
            "source_read_id": "source_read_unrelated",
            "run_id": "run_1",
            "step_id": "step_execute",
            "catalog_endpoint_id": "55555555-5555-4555-8555-555555555555",
            "args_json": {"month": "2026-05"},
            "status": "succeeded",
            "row_count": 1,
            "response_hash": "sha256:unrelated",
        },
        *dataset["source_reads"],
    ]
    result = render_fervis_result(
        FervisCommandResult(
            kind=FervisCommandKind.EXPLAIN,
            payload=evaluate_fervis(
                ("explain", "answer_1", "--format", "text"),
                ports=FervisCliPorts(
                    lineage_query=fixture_lineage_query(dataset),
                    observability_query=_ObservabilityQuery(),
                    prompt_capture_query=_PromptCaptureQuery(),
                    questions=_QuestionService(),
                ),
            ).payload,
            view_kind=FervisViewKind.LINEAGE,
        )
    )

    derived_input_lines = [
        line.strip()
        for line in result.splitlines()
        if line.strip().startswith("Derived inputs:")
    ]
    assert derived_input_lines == ["Derived inputs: month=2026-06"]
