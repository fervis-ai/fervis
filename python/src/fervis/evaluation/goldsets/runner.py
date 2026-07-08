"""Generic goldset runner for host-owned suite cases."""

from __future__ import annotations

import json
from pathlib import Path

from fervis.interfaces.cli.runtime_ask import RuntimeAskFollower, RuntimeAskQuestions
from fervis.interfaces.common.admission import ConfiguredModelPolicy
from fervis.questions import (
    AskRequest,
    AskRequestLimits,
    ExecutionMode,
    QuestionPrincipal,
)

from .contracts import (
    GoldsetCase,
    GoldsetCaseResult,
    GoldsetRunResult,
    GoldsetSuite,
)


class GoldsetPreflightError(ValueError):
    pass


def run_goldset_suite(
    suite: GoldsetSuite,
    *,
    questions: RuntimeAskQuestions,
    question_run_follower: RuntimeAskFollower | None,
    question_run_limits: AskRequestLimits,
    model_policy: ConfiguredModelPolicy,
    principal: QuestionPrincipal,
    case_ids: tuple[str, ...] = (),
    limit: int | None = None,
    model_key: str | None = None,
    wait_seconds: float = 60.0,
    ledger_file: Path | None = None,
) -> GoldsetRunResult:
    cases = _selected_cases(
        suite.cases,
        case_ids=case_ids,
        limit=limit,
    )
    _run_suite_preflight(suite)
    results = tuple(
        _run_case(
            case,
            suite=suite,
            questions=questions,
            question_run_follower=question_run_follower,
            question_run_limits=question_run_limits,
            model_policy=model_policy,
            model_key=model_key,
            principal=principal,
            wait_seconds=wait_seconds,
        )
        for case in cases
    )
    if ledger_file is not None:
        _write_ledger(ledger_file, results)
    passed_count = sum(1 for result in results if result.status == "passed")
    return GoldsetRunResult(
        suite_name=suite.name,
        case_count=len(results),
        passed_count=passed_count,
        failed_count=len(results) - passed_count,
        cases=results,
    )


def _selected_cases(
    cases: tuple[GoldsetCase, ...],
    *,
    case_ids: tuple[str, ...],
    limit: int | None,
) -> tuple[GoldsetCase, ...]:
    requested = tuple(case_id for case_id in case_ids if case_id)
    if requested:
        by_id = {case.case_id: case for case in cases}
        missing = tuple(case_id for case_id in requested if case_id not in by_id)
        if missing:
            raise ValueError(f"goldset case not found: {', '.join(missing)}")
        selected = tuple(by_id[case_id] for case_id in requested)
    else:
        selected = cases
    if limit is not None:
        return selected[: max(0, int(limit))]
    return selected


def _run_case(
    case: GoldsetCase,
    *,
    suite: GoldsetSuite,
    questions: RuntimeAskQuestions,
    question_run_follower: RuntimeAskFollower | None,
    question_run_limits: AskRequestLimits,
    model_policy: ConfiguredModelPolicy,
    model_key: str | None,
    principal: QuestionPrincipal,
    wait_seconds: float,
) -> GoldsetCaseResult:
    if suite.prepare_case is not None:
        _run_case_setup(suite, case)
    admitted_model = model_policy.admit(
        requested_provider="",
        requested_model_key=model_key,
    )
    conversation_id = ""
    for setup_question in case.setup_questions:
        setup_result = _ask_and_follow(
            setup_question,
            conversation_id=conversation_id,
            questions=questions,
            question_run_follower=question_run_follower,
            question_run_limits=question_run_limits,
            provider=admitted_model.provider,
            model_key=admitted_model.model_key,
            principal=principal,
            wait_seconds=wait_seconds,
        )
        if setup_result.status != "COMPLETED":
            return GoldsetCaseResult(
                case_id=case.case_id,
                status="failed",
                question=case.question,
                conversation_id=setup_result.conversation_id,
                question_id=setup_result.question_id,
                run_id=setup_result.run_id,
                answer=setup_result.answer,
                message=f"setup question failed: {setup_question}",
                details={"setup_status": setup_result.status},
            )
        conversation_id = setup_result.conversation_id
    result = _ask_and_follow(
        case.question,
        conversation_id=conversation_id,
        questions=questions,
        question_run_follower=question_run_follower,
        question_run_limits=question_run_limits,
        provider=admitted_model.provider,
        model_key=admitted_model.model_key,
        principal=principal,
        wait_seconds=wait_seconds,
    )
    try:
        match = suite.match_answer(case, result)
    except Exception as exc:
        return GoldsetCaseResult(
            case_id=case.case_id,
            status="failed",
            question=case.question,
            conversation_id=result.conversation_id,
            question_id=result.question_id,
            run_id=result.run_id,
            answer=result.answer,
            message=f"goldset oracle failed: {exc}",
            details={"failure_class": "goldset_oracle_failed"},
        )
    return GoldsetCaseResult(
        case_id=case.case_id,
        status="passed" if match.passed else "failed",
        question=case.question,
        conversation_id=result.conversation_id,
        question_id=result.question_id,
        run_id=result.run_id,
        answer=result.answer,
        message=match.message,
        details=dict(match.details),
    )


def _run_suite_preflight(suite: GoldsetSuite) -> None:
    if suite.preflight is None:
        return
    try:
        suite.preflight()
    except Exception as exc:
        raise GoldsetPreflightError(f"goldset preflight failed: {exc}") from exc


def _run_case_setup(suite: GoldsetSuite, case: GoldsetCase) -> None:
    if suite.prepare_case is None:
        return
    try:
        suite.prepare_case(case)
    except Exception as exc:
        raise GoldsetPreflightError(
            f"goldset case setup failed for {case.case_id}: {exc}"
        ) from exc


def _ask_and_follow(
    question: str,
    *,
    conversation_id: str,
    questions: RuntimeAskQuestions,
    question_run_follower: RuntimeAskFollower | None,
    question_run_limits: AskRequestLimits,
    provider: str,
    model_key: str,
    principal: QuestionPrincipal,
    wait_seconds: float,
):
    result = questions.ask(
        AskRequest(
            question=question,
            principal=principal,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id=conversation_id,
            provider=provider,
            model_key=model_key,
            limits=question_run_limits,
        ),
        event_sink=_NullEventSink(),
    )
    if (
        question_run_follower is not None
        and wait_seconds > 0
        and result.status in {"QUEUED", "RUNNING"}
    ):
        return question_run_follower.follow(
            result,
            event_sink=_NullEventSink(),
            wait_seconds=wait_seconds,
        )
    return result


def _write_ledger(path: Path, results: tuple[GoldsetCaseResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_payload(), sort_keys=True))
            handle.write("\n")


class _NullEventSink:
    def emit(self, event) -> None:
        del event
