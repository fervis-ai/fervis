import type { ReactNode } from "react";

import {
  stepInputItemText,
  type StepInputItem,
  type StepSignal
} from "../stepDisplay";

export function StepSignalContent({
  signal,
  tail = null
}: {
  readonly signal: StepSignal;
  readonly tail?: ReactNode;
}) {
  if (signal.kind === "inputs") {
    return <StepInputSummary inputs={signal.inputs} tail={tail} />;
  }
  return (
    <>
      {signal.label}: {signal.text}
      {tail}
    </>
  );
}

export function StepInputSummary({
  inputs,
  tail = null
}: {
  readonly inputs: readonly StepInputItem[];
  readonly tail?: ReactNode;
}) {
  return (
    <div className="step-input-summary">
      <div className="step-input-summary-label">Inputs:</div>
      <ul>
        {inputs.map((input, index) => (
          <li key={`${input.sourceText}:${input.summary}`}>
            {stepInputItemText(input)}
            {index === inputs.length - 1 ? tail : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
