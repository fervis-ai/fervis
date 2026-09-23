import type { ClarificationOption, ClarificationRequest, RunPayload, RunStatus, TerminalResultData } from "../fervis-api/contracts";
import { formatTriggerKind } from "./textFormat";

export function failMissingOption(): ClarificationOption {
  throw new Error("Choice clarification requires at least one option");
}

export function firstClarification(run: RunPayload): ClarificationRequest | null {
  if (
    run.status !== "WAITING_FOR_CLARIFICATION" ||
    run.resultData?.kind !== "needs_clarification"
  ) {
    return null;
  }
  return run.resultData.details.clarifications[0] ?? null;
}

export function clarificationOptions(
  clarification: ClarificationRequest
): readonly ClarificationOption[] {
  return clarification.subjects.flatMap((subject) => subject.options);
}

export function runSummary(run: RunPayload): string {
  const clarification = firstClarification(run);
  if (clarification !== null) {
    return `clarification: "${clarification.question}" · ${run.steps.length} steps`;
  }
  if (run.status === "FAILED") {
    return `failed · ${run.steps.length} steps`;
  }
  const executionLabel =
    run.executionKind === "continue_prior_request"
      ? formatTriggerKind(run.executionKind)
      : formatTriggerKind(run.triggerKind);
  return `${run.steps.length} steps · ${executionLabel}`;
}

export function statusClassName(status: RunStatus): string {
  if (status === "RUNNING" || status === "QUEUED") {
    return "running";
  }
  if (status === "FAILED") {
    return "failed";
  }
  if (status === "WAITING_FOR_CLARIFICATION") {
    return "clarification";
  }
  return "completed";
}

export function pollableStatus(status: RunStatus): boolean {
  return status === "RUNNING" || status === "QUEUED";
}

export function pollableRun(run: RunPayload): boolean {
  if (pollableStatus(run.status)) {
    return true;
  }
  return run.status === "COMPLETED" && !completedRunRenderable(run);
}

function completedRunRenderable(run: RunPayload): boolean {
  return completedAnswerText(run) !== null;
}

export function completedAnswerText(run: RunPayload): string | null {
  if (run.answer !== null && run.answer.trim() !== "") {
    return run.answer;
  }
  const result = run.resultData;
  if (result?.kind === "answer" || result?.kind === "partial") {
    const answer = result.outputs.map((output) => output.displayValue).join(", ");
    const values = answer ? [answer] : [];
    if (result.kind === "partial") values.push(...result.facts.map(terminalResultText));
    return values.length > 0 ? values.join("\n") : null;
  }
  if (result?.kind === "impossible" || result?.kind === "no_data" || result?.kind === "undefined") {
    return terminalResultText(result);
  }
  return null;
}

function terminalResultText(result: TerminalResultData): string {
  return result.message;
}

export function askPlaceholder(status: RunStatus): string {
  if (status === "FAILED") {
    return "Re-ask, or ask a different question…";
  }
  if (status === "WAITING_FOR_CLARIFICATION") {
    return "Or ask a different question instead…";
  }
  return "Ask a follow-up question…";
}

export function askHint(status: RunStatus): string {
  if (status === "FAILED") {
    return "This run failed; sending a new question queues a fresh run.";
  }
  if (status === "WAITING_FOR_CLARIFICATION") {
    return "A clarification is pending above; answering it continues this question.";
  }
  if (status === "RUNNING" || status === "QUEUED") {
    return "A run is in progress; asking queues a new question in this conversation.";
  }
  return "Enter sends · runs are queued and polled under the conversation.";
}
