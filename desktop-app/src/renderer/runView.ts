import type { ClarificationOption, ClarificationRequest, RunPayload, RunStatus } from "../fervis-api/contracts";
import { formatTriggerKind } from "./textFormat";

export function failMissingOption(): ClarificationOption {
  throw new Error("Choice clarification requires at least one option");
}

export function firstClarification(run: RunPayload): ClarificationRequest | null {
  if (run.resultData?.kind !== "needs_clarification") {
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
  return `${run.steps.length} steps · ${formatTriggerKind(run.triggerKind)}`;
}

export function statusClassName(status: RunStatus): string {
  if (status === "RUNNING" || status === "QUEUED") {
    return "running";
  }
  if (status === "FAILED") {
    return "failed";
  }
  if (status === "NEEDS_CLARIFICATION") {
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
  if (run.answer !== null) {
    return run.answer;
  }
  if (run.resultData?.kind !== "answer" || run.resultData.outputs.length === 0) {
    return null;
  }
  return run.resultData.outputs.map((output) => output.value).join(", ");
}

export function askPlaceholder(status: RunStatus): string {
  if (status === "FAILED") {
    return "Re-ask, or ask a different question…";
  }
  if (status === "NEEDS_CLARIFICATION") {
    return "Or ask a different question instead…";
  }
  return "Ask a follow-up question…";
}

export function askHint(status: RunStatus): string {
  if (status === "FAILED") {
    return "This run failed; sending a new question queues a fresh run.";
  }
  if (status === "NEEDS_CLARIFICATION") {
    return "A clarification is pending above; answering it continues this question.";
  }
  if (status === "RUNNING" || status === "QUEUED") {
    return "A run is in progress; asking queues a new question in this conversation.";
  }
  return "Enter sends · runs are queued and polled under the conversation.";
}
