"""Generic goldset runner for host-owned suite cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
import uuid

from fervis.interfaces.common.admission import ConfiguredModelPolicy
from fervis.questions import (
    AskRequest,
    AskRequestLimits,
    AskResult,
    ClarificationResponseRequest,
    ExecutionMode,
    QuestionPrincipal,
)
from fervis.questions.result_data import result_data_clarifications

from .contracts import (
    GoldsetCase,
    GoldsetCaseResult,
    GoldsetMatch,
    GoldsetRunResult,
    GoldsetSuite,
)
from .ports import GoldsetQuestions, GoldsetRunFollower


class GoldsetPreflightError(ValueError):
    pass


_RETRYABLE_PROVIDER_ERRORS = {
    "provider_runtime_failed",
    "provider_connection_failed",
    "provider_timeout",
    "provider_rate_limited",
    "provider_internal_error",
}


@dataclass(frozen=True)
class _CaseExecution:
    result: AskResult
    setup_results: tuple[AskResult, ...]
    failure_class: str = ""
    failure_message: str = ""


@dataclass(frozen=True)
class _EvaluatedCaseRun:
    result: AskResult
    evaluation: GoldsetCaseResult


GoldsetCaseRunObserver = Callable[[int, int, GoldsetCaseResult], None]


def run_goldset_suite(
    suite: GoldsetSuite,
    *,
    questions: GoldsetQuestions,
    question_run_follower: GoldsetRunFollower | None,
    question_run_limits: AskRequestLimits,
    model_policy: ConfiguredModelPolicy,
    principal: QuestionPrincipal,
    case_ids: tuple[str, ...] = (),
    limit: int | None = None,
    model_key: str | None = None,
    wait_seconds: float = 60.0,
    ledger_file: Path | None = None,
    stable_runs: int = 1,
    enforce_structured_determinism: bool = False,
    attempts: int = 1,
    retry_provider_failures: bool = False,
    retry_sleep_seconds: float = 300.0,
    case_run_observer: GoldsetCaseRunObserver | None = None,
) -> GoldsetRunResult:
    run_count = _positive_int(stable_runs, field_name="stable_runs")
    attempt_count = _positive_int(attempts, field_name="attempts")
    goldset_run_id = f"goldset:{uuid.uuid4()}"
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
            goldset_run_id=goldset_run_id,
            stable_runs=run_count,
            enforce_structured_determinism=enforce_structured_determinism,
            attempts=attempt_count,
            retry_provider_failures=retry_provider_failures,
            retry_sleep_seconds=retry_sleep_seconds,
            case_run_observer=case_run_observer,
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
    questions: GoldsetQuestions,
    question_run_follower: GoldsetRunFollower | None,
    question_run_limits: AskRequestLimits,
    model_policy: ConfiguredModelPolicy,
    model_key: str | None,
    principal: QuestionPrincipal,
    wait_seconds: float,
    goldset_run_id: str,
    stable_runs: int,
    enforce_structured_determinism: bool,
    attempts: int,
    retry_provider_failures: bool,
    retry_sleep_seconds: float,
    case_run_observer: GoldsetCaseRunObserver | None,
) -> GoldsetCaseResult:
    admitted_model = model_policy.admit(
        requested_provider="",
        requested_model_key=model_key or "",
    )
    runs: list[_EvaluatedCaseRun] = []
    for stability_run in range(1, stable_runs + 1):
        run = _run_case_once(
            case,
            suite=suite,
            questions=questions,
            question_run_follower=question_run_follower,
            question_run_limits=question_run_limits,
            provider=admitted_model.provider,
            model_key=admitted_model.model_key,
            principal=principal,
            wait_seconds=wait_seconds,
            runtime_context=_evaluation_context(
                case_id=case.case_id,
                goldset_run_id=goldset_run_id,
                stability_run=stability_run,
                stable_runs=stable_runs,
            ),
            attempts=attempts,
            retry_provider_failures=retry_provider_failures,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        runs.append(run)
        if case_run_observer is not None:
            case_run_observer(
                stability_run,
                stable_runs,
                run.evaluation,
            )
    return _aggregate_case_runs(
        tuple(runs),
        enforce_structured_determinism=enforce_structured_determinism,
    )


def _run_case_once(
    case: GoldsetCase,
    *,
    suite: GoldsetSuite,
    questions: GoldsetQuestions,
    question_run_follower: GoldsetRunFollower | None,
    question_run_limits: AskRequestLimits,
    provider: str,
    model_key: str,
    principal: QuestionPrincipal,
    wait_seconds: float,
    runtime_context: dict[str, str],
    attempts: int,
    retry_provider_failures: bool,
    retry_sleep_seconds: float,
) -> _EvaluatedCaseRun:
    if suite.prepare_case is not None:
        _run_case_setup(suite, case)
    execution = _execute_case(
        case,
        questions=questions,
        question_run_follower=question_run_follower,
        question_run_limits=question_run_limits,
        provider=provider,
        model_key=model_key,
        principal=principal,
        wait_seconds=wait_seconds,
        runtime_context=runtime_context,
    )
    attempt = 1
    while (
        retry_provider_failures
        and attempt < attempts
        and _is_retryable_provider_failure(execution.result)
    ):
        time.sleep(max(0.0, retry_sleep_seconds))
        attempt += 1
        execution = _execute_case(
            case,
            questions=questions,
            question_run_follower=question_run_follower,
            question_run_limits=question_run_limits,
            provider=provider,
            model_key=model_key,
            principal=principal,
            wait_seconds=wait_seconds,
            runtime_context=runtime_context,
        )
    return _evaluate_execution(case, suite=suite, execution=execution)


def _execute_case(
    case: GoldsetCase,
    *,
    questions: GoldsetQuestions,
    question_run_follower: GoldsetRunFollower | None,
    question_run_limits: AskRequestLimits,
    provider: str,
    model_key: str,
    principal: QuestionPrincipal,
    wait_seconds: float,
    runtime_context: dict[str, str],
) -> _CaseExecution:
    conversation_id = ""
    setup_results: list[AskResult] = []
    for setup_question in case.setup_questions:
        setup_started_at = time.monotonic()
        setup_result = _ask_and_follow(
            setup_question,
            conversation_id=conversation_id,
            questions=questions,
            question_run_follower=question_run_follower,
            question_run_limits=question_run_limits,
            provider=provider,
            model_key=model_key,
            principal=principal,
            wait_seconds=wait_seconds,
            runtime_context=runtime_context,
        )
        setup_result = _with_wall_clock_duration(
            setup_result,
            started_at=setup_started_at,
        )
        setup_failure = _setup_failure_message(
            setup_result,
            question=setup_question,
        )
        if setup_failure:
            return _CaseExecution(
                result=setup_result,
                setup_results=tuple(setup_results),
                failure_class="setup_question_failed",
                failure_message=setup_failure,
            )
        setup_results.append(setup_result)
        conversation_id = setup_result.conversation_id
    target_started_at = time.monotonic()
    result = _ask_and_follow(
        case.question,
        conversation_id=conversation_id,
        questions=questions,
        question_run_follower=question_run_follower,
        question_run_limits=question_run_limits,
        provider=provider,
        model_key=model_key,
        principal=principal,
        wait_seconds=wait_seconds,
        runtime_context=runtime_context,
    )
    for answer in case.clarification_answers:
        if result.status != "WAITING_FOR_CLARIFICATION":
            result = _with_wall_clock_duration(result, started_at=target_started_at)
            return _CaseExecution(
                result=result,
                setup_results=tuple(setup_results),
                failure_class="clarification_not_requested",
                failure_message=(
                    "Expected the target turn to request clarification before "
                    f"accepting: {answer}"
                ),
            )
        continued = _continue_and_follow(
            answer,
            previous=result,
            questions=questions,
            question_run_follower=question_run_follower,
            principal=principal,
            wait_seconds=wait_seconds,
        )
        if continued is None:
            result = _with_wall_clock_duration(result, started_at=target_started_at)
            return _CaseExecution(
                result=result,
                setup_results=tuple(setup_results),
                failure_class="clarification_context_missing",
                failure_message="Clarification result has no actionable clarification id.",
            )
        result = continued
    for answer in case.optional_clarification_answers:
        if result.status != "WAITING_FOR_CLARIFICATION":
            break
        continued = _continue_and_follow(
            answer,
            previous=result,
            questions=questions,
            question_run_follower=question_run_follower,
            principal=principal,
            wait_seconds=wait_seconds,
        )
        if continued is None:
            result = _with_wall_clock_duration(result, started_at=target_started_at)
            return _CaseExecution(
                result=result,
                setup_results=tuple(setup_results),
                failure_class="clarification_context_missing",
                failure_message="Clarification result has no actionable clarification id.",
            )
        result = continued
    result = _with_wall_clock_duration(result, started_at=target_started_at)
    return _CaseExecution(result=result, setup_results=tuple(setup_results))


def _setup_failure_message(result: AskResult, *, question: str) -> str:
    if result.status != "COMPLETED":
        return f"setup question failed: {question}"
    result_data = result.result_data
    if not isinstance(result_data, Mapping) or result_data.get("kind") != "answer":
        return f"setup question did not produce an answer: {question}"
    return ""


def _with_wall_clock_duration(
    result: AskResult,
    *,
    started_at: float,
) -> AskResult:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    return replace(result, duration_ms=max(0, elapsed_ms))


def _evaluate_execution(
    case: GoldsetCase,
    *,
    suite: GoldsetSuite,
    execution: _CaseExecution,
) -> _EvaluatedCaseRun:
    result = execution.result
    if execution.failure_class:
        return _EvaluatedCaseRun(
            result=result,
            evaluation=_case_result(
                case,
                result=result,
                status="failed",
                message=execution.failure_message,
                details={"failure_class": execution.failure_class},
            ),
        )
    try:
        match = suite.match_answer(case, result)
    except Exception as exc:
        return _EvaluatedCaseRun(
            result=result,
            evaluation=_case_result(
                case,
                result=result,
                status="failed",
                message=f"goldset oracle failed: {exc}",
                details={"failure_class": "goldset_oracle_failed"},
            ),
        )
    match = _with_duration_assertion(
        case=case,
        result=result,
        setup_results=execution.setup_results,
        match=match,
    )
    return _EvaluatedCaseRun(
        result=result,
        evaluation=_case_result(
            case,
            result=result,
            status="passed" if match.passed else "failed",
            message=match.message,
            details=dict(match.details),
        ),
    )


def _case_result(
    case: GoldsetCase,
    *,
    result: AskResult,
    status: str,
    message: str,
    details: dict[str, object],
) -> GoldsetCaseResult:
    return GoldsetCaseResult(
        case_id=case.case_id,
        status=status,
        question=case.question,
        conversation_id=result.conversation_id,
        question_id=result.question_id,
        run_id=result.run_id,
        answer=result.answer,
        message=message,
        duration_ms=result.duration_ms,
        details=details,
    )


def _aggregate_case_runs(
    runs: tuple[_EvaluatedCaseRun, ...],
    *,
    enforce_structured_determinism: bool,
) -> GoldsetCaseResult:
    if len(runs) == 1:
        return runs[0].evaluation
    passed_count = sum(run.evaluation.status == "passed" for run in runs)
    records = [
        {
            "run": index,
            "status": run.evaluation.status,
            "run_id": run.evaluation.run_id,
            "message": run.evaluation.message,
            "duration_ms": run.evaluation.duration_ms,
            "result_fingerprint": _result_fingerprint(run.result),
        }
        for index, run in enumerate(runs, start=1)
    ]
    fingerprints = {record["result_fingerprint"] for record in records}
    all_passed = passed_count == len(runs)
    structured_match = len(fingerprints) <= 1
    if all_passed and (structured_match or not enforce_structured_determinism):
        subject = runs[-1].evaluation
        return replace(
            subject,
            details={
                **dict(subject.details),
                "stability_runs": records,
            },
        )
    subject = runs[-1].evaluation if passed_count else runs[0].evaluation
    if all_passed:
        message = "All runs passed, but their structured results differed."
    elif passed_count:
        message = f"{passed_count}/{len(runs)} stability runs passed."
    else:
        message = subject.message
    return replace(
        subject,
        status="failed",
        message=message,
        details={
            **dict(subject.details),
            "failure_class": (
                "nondeterminism"
                if passed_count or all_passed
                else subject.details.get("failure_class")
            ),
            "stability_runs": records,
        },
    )


def _result_fingerprint(result: AskResult) -> str:
    payload = {
        "status": result.status,
        "result_data": result.result_data,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def _evaluation_context(
    *,
    case_id: str,
    goldset_run_id: str,
    stability_run: int,
    stable_runs: int,
) -> dict[str, str]:
    return {
        "caseId": case_id,
        "goldsetRunId": goldset_run_id,
        "certificationRunId": goldset_run_id,
        "stabilityRun": str(stability_run),
        "stableRuns": str(stable_runs),
    }


def _positive_int(value: int, *, field_name: str) -> int:
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return normalized


def _is_retryable_provider_failure(result: AskResult) -> bool:
    return result.status == "FAILED" and result.error in _RETRYABLE_PROVIDER_ERRORS


def _with_duration_assertion(
    *,
    case: GoldsetCase,
    result: AskResult,
    setup_results: tuple[AskResult, ...],
    match: GoldsetMatch,
) -> GoldsetMatch:
    if not match.passed:
        return match
    max_ratio = case.metadata.get("max_duration_ratio_to_setup")
    if max_ratio is None:
        return match
    try:
        ratio = float(max_ratio)
    except (TypeError, ValueError):
        return GoldsetMatch(
            passed=False,
            message=f"invalid max_duration_ratio_to_setup: {max_ratio}",
            details={
                **dict(match.details),
                "failure_class": "goldset_runtime_assertion_invalid",
            },
        )
    baseline = _first_completed_setup_duration_ms(setup_results)
    actual = result.duration_ms
    if baseline is None or actual is None:
        return GoldsetMatch(
            passed=False,
            message="Missing run duration for deterministic timing assertion.",
            details={
                **dict(match.details),
                "failure_class": "runtime_duration_missing",
                "duration_ms": actual,
                "setup_durations_ms": _setup_durations_ms(setup_results),
            },
        )
    threshold = baseline * ratio
    if actual < threshold:
        return GoldsetMatch(
            passed=True,
            message=match.message,
            details={
                **dict(match.details),
                "duration_ms": actual,
                "setup_duration_ms": baseline,
                "max_duration_ratio_to_setup": ratio,
            },
        )
    return GoldsetMatch(
        passed=False,
        message=(
            f"Expected run duration {actual}ms to be < "
            f"{ratio:g}x setup duration {baseline}ms."
        ),
        details={
            **dict(match.details),
            "failure_class": "runtime_duration_too_slow",
            "duration_ms": actual,
            "setup_duration_ms": baseline,
            "max_duration_ratio_to_setup": ratio,
        },
    )


def _first_completed_setup_duration_ms(
    setup_results: tuple[AskResult, ...],
) -> int | None:
    for setup in setup_results:
        if setup.status == "COMPLETED" and setup.duration_ms is not None:
            return setup.duration_ms
    return None


def _setup_durations_ms(setup_results: tuple[AskResult, ...]) -> list[int | None]:
    return [setup.duration_ms for setup in setup_results]


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
    questions: GoldsetQuestions,
    question_run_follower: GoldsetRunFollower | None,
    question_run_limits: AskRequestLimits,
    provider: str,
    model_key: str,
    principal: QuestionPrincipal,
    wait_seconds: float,
    runtime_context: dict[str, str],
) -> AskResult:
    result = questions.ask(
        AskRequest(
            question=question,
            principal=principal,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id=conversation_id,
            provider=provider,
            model_key=model_key,
            runtime_context=runtime_context,
            limits=question_run_limits,
        ),
        event_sink=_NullEventSink(),
    )
    return _follow_result(
        result,
        question_run_follower=question_run_follower,
        wait_seconds=wait_seconds,
    )


def _continue_and_follow(
    answer: str,
    *,
    previous: AskResult,
    questions: GoldsetQuestions,
    question_run_follower: GoldsetRunFollower | None,
    principal: QuestionPrincipal,
    wait_seconds: float,
) -> AskResult | None:
    clarification_id = _first_clarification_id(previous)
    if not clarification_id:
        return None
    result = questions.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=previous.question_id,
            run_id=previous.run_id,
            clarification_id=clarification_id,
            response_text=answer,
            principal=principal,
            execution_mode=ExecutionMode.QUEUED,
        ),
        event_sink=_NullEventSink(),
    )
    return _follow_result(
        result,
        question_run_follower=question_run_follower,
        wait_seconds=wait_seconds,
    )


def _follow_result(
    result: AskResult,
    *,
    question_run_follower: GoldsetRunFollower | None,
    wait_seconds: float,
) -> AskResult:
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


def _first_clarification_id(result: AskResult) -> str:
    for item in result_data_clarifications(result.result_data):
        if isinstance(item, Mapping):
            clarification_id = str(item.get("id") or "").strip()
            if clarification_id:
                return clarification_id
    return ""


def _write_ledger(path: Path, results: tuple[GoldsetCaseResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_payload(), sort_keys=True))
            handle.write("\n")


class _NullEventSink:
    def emit(self, event) -> None:
        del event
