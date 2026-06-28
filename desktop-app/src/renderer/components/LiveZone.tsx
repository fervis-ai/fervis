import type { RunPayload, RunStep } from "../../fervis-api/contracts";
import { liveStepHighlightsFor, sourceReferenceLabelsFor } from "../stepDisplay";
import { formatStepKey } from "../textFormat";

export function LiveZone({ run }: { readonly run: RunPayload }) {
  const currentStep = run.steps[run.steps.length - 1];

  if (currentStep === undefined) {
    return (
      <div className="live-zone">
        <div className="step-head">0 / 0 · starting</div>
        <div className="decision-lines">
          <div className="decision-line current">
            Preparing the run before the first reasoning step starts.
            <span className="cursor" />
          </div>
        </div>
      </div>
    );
  }

  const lines = liveStepLines(currentStep, run);

  return (
    <div className="live-zone">
      <div className="step-head">
        <span className="step-index">{run.steps.length}</span>
        <span className="step-total">/ {run.steps.length}</span>
        <span className="step-name">{formatStepKey(currentStep.stepKey)}</span>
      </div>
      <div className="decision-lines">
        {lines.map((line, index) => (
          <div className="decision-line current" key={`${currentStep.stepId}-${index}`}>
            {line}
            {index === lines.length - 1 ? <span className="cursor" /> : null}
          </div>
        ))}
      </div>
      <div className="trail">
        {run.steps.map((step, index) => (
          <span key={step.stepId}>
            {index < run.steps.length - 1 ? "✓ " : ""}
            {index + 1} {formatStepKey(step.stepKey)}
          </span>
        ))}
      </div>
    </div>
  );
}

function liveStepLines(step: RunStep, run: RunPayload): readonly string[] {
  const decisionLines = step.decisions.flatMap((decision) => decision.lines);
  const sourceLabels = sourceReferenceLabelsFor(run, decisionLines);
  return liveStepHighlightsFor(
    step.stepKey,
    decisionLines,
    step.semantic,
    sourceLabels
  ).map((signal) => `${signal.label}: ${signal.text}`);
}
