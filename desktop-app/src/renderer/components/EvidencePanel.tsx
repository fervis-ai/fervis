import { useEffect, useState } from "react";

import type { RunPayload } from "../../fervis-api/contracts";
import type { FervisApiClient } from "../../fervis-api/client";
import { AskAboutAnswerControl } from "./AskAboutAnswerControl";
import { ExplanationProof } from "./evidence/ExplanationProof";
import { InputsMatrix } from "./evidence/InputsMatrix";
import type { ProofMode } from "./evidence/types";

export function EvidencePanel({
  run,
  defaultOpen,
  apiClient
}: {
  readonly run: RunPayload;
  readonly defaultOpen: boolean;
  readonly apiClient: FervisApiClient | null;
}) {
  const [proofMode, setProofMode] = useState<ProofMode>("compact");
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [isPlayingAnswer, setIsPlayingAnswer] = useState(false);

  useEffect(() => {
    setIsOpen(defaultOpen);
  }, [defaultOpen, run.runId]);

  useEffect(() => {
    if (isPlayingAnswer) {
      setIsOpen(true);
    }
  }, [isPlayingAnswer]);

  return (
    <details
      className={
        isPlayingAnswer ? "evidence-panel answering" : "evidence-panel"
      }
      open={isOpen}
      onToggle={(event) => setIsOpen(event.currentTarget.open)}
    >
      <summary>
        <span className="evidence-summary-label">Evidence</span>
        {run.status === "COMPLETED" && apiClient !== null ? (
          <AskAboutAnswerControl
            apiClient={apiClient}
            questionId={run.questionId}
            runId={run.runId}
            onPlaybackChange={setIsPlayingAnswer}
          />
        ) : null}
        <span className="evidence-summary-meta">{evidenceSummary(run)}</span>
      </summary>
      <div className="evidence-body">
        <section className="evidence-zone">
          <div className="evidence-label">
            <span>§3</span> Inputs
          </div>
          {run.status === "COMPLETED" ? (
            <InputsMatrix inputs={run.explanation.inputs.results} />
          ) : (
            <p className="quiet">
              {run.status === "FAILED"
                ? "no inputs computed — run failed before an answer was produced"
                : "pending run completion"}
            </p>
          )}
        </section>
        <section className="evidence-zone">
          <div className="explanation-head">
            <div className="evidence-label">
              <span>§4</span> Explanation
            </div>
            <ProofModeToggle mode={proofMode} onChange={setProofMode} />
          </div>
          <ExplanationProof mode={proofMode} run={run} />
        </section>
      </div>
    </details>
  );
}

function ProofModeToggle({
  mode,
  onChange
}: {
  readonly mode: ProofMode;
  readonly onChange: (mode: ProofMode) => void;
}) {
  return (
    <div className="toggle">
      <button
        className={mode === "compact" ? "active" : ""}
        type="button"
        onClick={() => onChange("compact")}
      >
        Compact
      </button>
      <button
        className={mode === "verbose" ? "active" : ""}
        type="button"
        onClick={() => onChange("verbose")}
      >
        More
      </button>
    </div>
  );
}

function evidenceSummary(run: RunPayload): string {
  const inputCount = run.explanation.inputs.results.length;
  const stepCount = run.steps.length;
  if (run.status === "COMPLETED") {
    return `${inputCount} input${inputCount === 1 ? "" : "s"} · ${stepCount} step${
      stepCount === 1 ? "" : "s"
    }`;
  }
  if (run.status === "FAILED") {
    return `failure trace · ${stepCount} step${stepCount === 1 ? "" : "s"}`;
  }
  return `pending · ${stepCount} step${stepCount === 1 ? "" : "s"}`;
}
