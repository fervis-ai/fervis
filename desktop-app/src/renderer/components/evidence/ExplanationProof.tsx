import { useState } from "react";

import type { LineageStep, RunPayload, SourceRead } from "../../../fervis-api/contracts";
import { DetailToggle } from "../DetailToggle";
import { formatStepKey } from "../../textFormat";
import {
  answerEvidenceInsights,
  compactProofSteps,
  compactProofSummary,
  compactSourceReadSummary,
  compactStepLabel,
  findLineageRun
} from "./evidenceInsights";
import { allVerboseProofNotes, proofNotes, ProofNoteView } from "./proofNotes";
import type { ProofMode, ProofNote } from "./types";

export function ExplanationProof({
  run,
  mode
}: {
  readonly run: RunPayload;
  readonly mode: ProofMode;
}) {
  const lineageRun = findLineageRun(run, mode === "compact");

  if (lineageRun === null || lineageRun.steps.length === 0) {
    return <p className="quiet">steps will render here as they persist</p>;
  }
  const evidenceSteps =
    mode === "compact" ? compactProofSteps(lineageRun.steps) : lineageRun.steps;
  const insights = answerEvidenceInsights(run, lineageRun);

  return (
    <div className="proof">
      <div className="quiet">
        {mode === "compact"
          ? compactProofSummary(run, lineageRun)
          : `${lineageRun.steps.length} step${
              lineageRun.steps.length === 1 ? "" : "s"
            } · verbose trace`}
      </div>
      {mode === "compact" && insights.length > 0 ? (
        <div className="proof-insights">
          {insights.map((insight) => (
            <div className="proof-insight" key={`${insight.label}-${insight.value}`}>
              <span>{insight.label}</span>
              <strong>{insight.value}</strong>
              {insight.detail !== null ? <small>{insight.detail}</small> : null}
            </div>
          ))}
        </div>
      ) : null}
      {evidenceSteps.map((step, index) => (
        <ProofStep
          index={index + 1}
          key={step.stepId}
          mode={mode}
          run={run}
          step={step}
        />
      ))}
    </div>
  );
}

function ProofStep({
  step,
  index,
  mode,
  run
}: {
  readonly step: LineageStep;
  readonly index: number;
  readonly mode: ProofMode;
  readonly run: RunPayload;
}) {
  const failed = step.runtimeErrors.length > 0;
  const notes = proofNotes(step, mode, run);
  const allNotes = mode === "verbose" ? allVerboseProofNotes(step, run) : notes;
  return (
    <div className={failed ? "proof-step failed" : "proof-step"}>
      <div className="proof-index">{index}</div>
      <div className="proof-body">
        <div className={failed ? "proof-key failed" : "proof-key"}>
          {mode === "compact"
            ? compactStepLabel(step)
            : formatStepKey(step.stepKey)}
          {failed ? " · failed" : ""}
        </div>
        {step.sourceReads.map((read) => (
          <SourceReadLine key={read.sourceReadId} mode={mode} read={read} />
        ))}
        {notes.length > 0 ? <ProofNotes notes={notes} allNotes={allNotes} /> : null}
      </div>
    </div>
  );
}

function ProofNotes({
  notes,
  allNotes
}: {
  readonly notes: readonly ProofNote[];
  readonly allNotes: readonly ProofNote[];
}) {
  const [showHidden, setShowHidden] = useState(false);
  const hiddenNotes = allNotes.slice(notes.length);
  const renderedNotes = showHidden ? allNotes : notes;

  return (
    <div className="proof-notes">
      {renderedNotes.map((note) => (
        <ProofNoteView note={note} key={`${note.label}-${note.text}`} />
      ))}
      {hiddenNotes.length > 0 ? (
        <DetailToggle
          collapsedAriaLabel="Show lower-signal details"
          collapsedLabel={`${hiddenNotes.length} lower-signal detail${
            hiddenNotes.length === 1 ? "" : "s"
          } hidden`}
          expanded={showHidden}
          expandedAriaLabel="Hide lower-signal details"
          expandedLabel="Hide lower-signal details"
          onToggle={() => setShowHidden((current) => !current)}
        />
      ) : null}
    </div>
  );
}

function SourceReadLine({
  read,
  mode
}: {
  readonly read: SourceRead;
  readonly mode: ProofMode;
}) {
  return (
    <div className="source-read">
      {mode === "compact"
        ? compactSourceReadSummary(read)
        : `${read.method} ${read.path} · ${read.rowCount} rows · `}
      <span className={read.status === "succeeded" ? "" : "bad"}>
        {mode === "compact" ? formatStepKey(read.status) : read.status}
      </span>
    </div>
  );
}
