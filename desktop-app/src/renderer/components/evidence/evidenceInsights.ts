import type {
  LineageRun,
  LineageStep,
  RunPayload,
  SourceRead
} from "../../../fervis-api/contracts";
import { semanticStepSignalsFor } from "../../stepDisplay";
import { formatRoutePath, formatStepKey, titleWords } from "../../textFormat";
import { inputSummary } from "./inputSummary";
import type { EvidenceInsight } from "./types";

export function findLineageRun(run: RunPayload, compact: boolean): LineageRun | null {
  const timeline = compact
    ? run.explanation.lineage.compact
    : run.explanation.lineage.verbose;
  const lineageQuestion = timeline.questions.find((question) =>
    question.runs.some((candidate) => candidate.runId === run.runId)
  );
  return (
    lineageQuestion?.runs.find((candidate) => candidate.runId === run.runId) ??
    null
  );
}

export function compactProofSteps(steps: readonly LineageStep[]): readonly LineageStep[] {
  const signalSteps = steps.filter(
    (step) =>
      step.sourceReads.length > 0 ||
      step.runtimeErrors.length > 0 ||
      step.decisions.length > 0 ||
      semanticStepSignalsFor(step.stepKey, step.semantic).length > 0
  );
  if (signalSteps.length > 0) {
    return signalSteps;
  }
  return steps.slice(-1);
}

export function answerEvidenceInsights(
  run: RunPayload,
  lineageRun: LineageRun
): readonly EvidenceInsight[] {
  const insights: EvidenceInsight[] = [];
  if (run.resultData?.kind === "answer") {
    insights.push(
      ...run.resultData.outputs.slice(0, 3).map((output) => ({
        label: titleWords(output.key),
        value: output.displayValue,
        detail: formatStepKey(output.valueKind)
      }))
    );
  }
  insights.push(
    ...run.explanation.inputs.results.slice(0, 2).map((input) => ({
      label: "Fact used",
      value: input.factDescription,
      detail: inputSummary(input)
    }))
  );
  const reads = sourceReadsFor(lineageRun);
  if (reads.length > 0) {
    const firstRead = reads[0] ?? failMissingSourceRead();
    insights.push({
      label: reads.length === 1 ? "Source read" : "Source reads",
      value: reads.length === 1 ? formatRoutePath(firstRead.path) : `${reads.length} API reads`,
      detail:
        reads.length === 1
          ? `${firstRead.rowCount} rows returned`
          : `${reads.reduce((total, read) => total + read.rowCount, 0)} rows returned`
    });
  }
  return insights;
}

export function sourceReadsFor(lineageRun: LineageRun): readonly SourceRead[] {
  return lineageRun.steps.flatMap((step) => step.sourceReads);
}

export function compactProofSummary(run: RunPayload, lineageRun: LineageRun): string {
  const reads = sourceReadsFor(lineageRun);
  if (run.status === "COMPLETED") {
    if (reads.length > 0) {
      return `${reads.length} source read${
        reads.length === 1 ? "" : "s"
      } used to derive the answer`;
    }
    return "answer derivation summary";
  }
  if (run.status === "FAILED") {
    return "failure summary";
  }
  return "pending evidence";
}

export function compactStepLabel(step: LineageStep): string {
  if (semanticStepSignalsFor(step.stepKey, step.semantic).length > 0) {
    return formatStepKey(step.stepKey);
  }
  if (step.sourceReads.length > 0) {
    return "Read source data";
  }
  if (step.runtimeErrors.length > 0) {
    return "Runtime failure";
  }
  return formatStepKey(step.stepKey);
}

export function compactSourceReadSummary(read: SourceRead): string {
  return `${read.method} ${formatRoutePath(read.path)} returned ${read.rowCount} row${
    read.rowCount === 1 ? "" : "s"
  } · `;
}

function failMissingSourceRead(): SourceRead {
  throw new Error("Source read summary requires at least one source read");
}
