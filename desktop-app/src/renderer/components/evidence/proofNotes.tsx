import { useState } from "react";

import type { LineageStep, RunPayload } from "../../../fervis-api/contracts";
import {
  formatEvidenceTextWithSourceReferences,
  labelSourceReferences,
  sourceReferenceLabelFor,
  sourceReferenceLabelsFor,
  semanticStepSignalsFor,
  type SourceReferenceLabel
} from "../../stepDisplay";
import {
  formatEvidenceText,
  formatStepKey,
  sentenceWithPeriod,
  titleWords
} from "../../textFormat";
import type { ProofMode, ProofNote } from "./types";

const PROOF_NOTE_PREVIEW_CHARS = 150;

export function ProofNoteView({ note }: { readonly note: ProofNote }) {
  const [expanded, setExpanded] = useState(false);
  const preview = proofNotePreview(note.text);
  const expandable = preview !== note.text;
  return (
    <div className="proof-note">
      <span>{note.label}</span>
      <p>{expanded ? note.text : preview}</p>
      {expandable ? (
        <button
          aria-label={expanded ? "Collapse rationale" : "Read full rationale"}
          className="proof-note-toggle"
          type="button"
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? "−" : "+"}
        </button>
      ) : null}
    </div>
  );
}

export function proofNotes(
  step: LineageStep,
  mode: ProofMode,
  run: RunPayload
): readonly ProofNote[] {
  if (mode === "compact") {
    if (step.runtimeErrors.length > 0) {
      return [
        {
          label: "Failure",
          text: step.runtimeErrors[0]?.message ?? "Step failed."
        }
      ];
    }
    const semanticNotes = stepSignalNotes(step);
    if (step.sourceReads.length > 0 && semanticNotes.length === 0) {
      return [];
    }
    if (semanticNotes.length > 0) {
      return semanticNotes;
    }
    return verboseProofNotes(step, run).slice(0, 2);
  }
  return verboseProofNotes(step, run).slice(0, 4);
}

export function proofNoteCount(
  step: LineageStep,
  mode: ProofMode,
  run: RunPayload
): number {
  if (mode === "compact") {
    return proofNotes(step, mode, run).length;
  }
  return verboseProofNotes(step, run).length;
}

export function allVerboseProofNotes(
  step: LineageStep,
  run: RunPayload
): readonly ProofNote[] {
  return verboseProofNotes(step, run);
}

function verboseProofNotes(
  step: LineageStep,
  run: RunPayload
): readonly ProofNote[] {
  const lines = step.decisions.flatMap((decision) => decision.lines);
  const sourceLabels = sourceReferenceLabelsFor(run, lines);
  const notes = lines
    .map((line) => proofNoteFromDecisionLine(line, sourceLabels))
    .filter((note): note is ProofNote => note !== null);
  if (notes.length > 0) {
    return prioritizeProofNotes(notes);
  }
  if (step.runtimeErrors.length > 0) {
    return step.runtimeErrors.map((error) => ({
      label: formatStepKey(error.code),
      text: error.message
    }));
  }
  return stepSignalNotes(step);
}

function stepSignalNotes(step: LineageStep): readonly ProofNote[] {
  return semanticStepSignalsFor(step.stepKey, step.semantic).map((signal) => ({
    label: signal.label,
    text: signal.text
  }));
}

function proofNoteFromDecisionLine(
  line: string,
  sourceLabels: readonly SourceReferenceLabel[]
): ProofNote | null {
  const normalized = normalizeTraceText(line);
  if (normalized === "") {
    return null;
  }

  const eligibility = normalized.match(/retained (\d+) source candidates?, dropped (\d+)/i);
  if (eligibility !== null) {
    return {
      label: "Read eligibility",
      text: `${eligibility[1]} source candidate${
        eligibility[1] === "1" ? "" : "s"
      } retained; ${eligibility[2]} dropped.`
    };
  }

  const sourceDecision = parseSourceDecision(normalized, sourceLabels);
  if (sourceDecision !== null) {
    return {
      label: `${sourceDecision.label} · ${sourceActionLabel(sourceDecision.action)}`,
      text: sentenceWithPeriod(
        cleanSourceRationale(sourceDecision.rationale, sourceLabels)
      )
    };
  }

  const reviewed = normalized.match(/reviewed source candidates?: (.+)$/i);
  if (reviewed !== null) {
    return {
      label: "Source candidates",
      text: sentenceWithPeriod(
        `Reviewed ${formatDecisionEvidenceText(reviewed[1] ?? "", sourceLabels)}`
      )
    };
  }

  return {
    label: formatDecisionLabel(normalized, sourceLabels),
    text: sentenceWithPeriod(formatDecisionEvidenceText(normalized, sourceLabels))
  };
}

function formatDecisionLabel(
  value: string,
  sourceLabels: readonly SourceReferenceLabel[]
): string {
  const label = labelSourceReferences(stepKeyFromDecision(value), sourceLabels);
  if (/^source_\d+\b/i.test(label)) {
    return label;
  }
  return formatStepKey(label);
}

function parseSourceDecision(
  value: string,
  sourceLabels: readonly SourceReferenceLabel[]
): { readonly label: string; readonly action: string; readonly rationale: string } | null {
  const actionMatch = value.match(
    /^(.*?)\b(RETAIN|DROP|DIRECT|REFERENCE|SKIP)\b\s*[-–—−]\s*(.+)$/i
  );
  if (actionMatch === null) {
    return null;
  }
  const prefix = (actionMatch[1] ?? "").trim().replace(/[:：]$/, "").trim();
  const sourceMatch = prefix.match(/^source[\s_-]*(\d+)(?:\s+(.+))?$/i);
  if (sourceMatch === null) {
    return null;
  }
  const sourceRef = `source_${sourceMatch[1] ?? ""}`;
  const sourceName = (sourceMatch[2] ?? "").trim();
  return {
    label:
      sourceName === ""
        ? sourceReferenceLabelFor(sourceRef, sourceLabels) ?? sourceRef
        : sourceReferenceLabelFor(sourceRef, sourceLabels) ?? `${sourceRef} (${formatStepKey(sourceName)})`,
    action: actionMatch[2] ?? "reviewed",
    rationale: actionMatch[3] ?? ""
  };
}

function cleanSourceRationale(
  value: string,
  sourceLabels: readonly SourceReferenceLabel[]
): string {
  return formatDecisionEvidenceText(
    value
      .replace(/^Rows:\s*\d+\s*-\s*Fields:\s*\d+\s*-\s*/i, "")
      .replace(/^Rows:\s*\d+\s*-\s*/i, "")
      .replace(/^Fields:\s*\d+\s*-\s*/i, "")
      .replace(/^rows=\d+\s*-\s*fields=\d+\s*-\s*/i, "")
      .replace(/^rows=\d+\s*-\s*/i, "")
      .replace(/^fields=\d+\s*-\s*/i, ""),
    sourceLabels
  );
}

function formatDecisionEvidenceText(
  value: string,
  sourceLabels: readonly SourceReferenceLabel[]
): string {
  return formatEvidenceTextWithSourceReferences(value, sourceLabels);
}

function sourceActionLabel(value: string): string {
  const normalized = value.toUpperCase();
  if (normalized === "RETAIN" || normalized === "DIRECT" || normalized === "REFERENCE") {
    return "used";
  }
  if (normalized === "DROP" || normalized === "SKIP") {
    return "not used";
  }
  return titleWords(value);
}

function prioritizeProofNotes(notes: readonly ProofNote[]): readonly ProofNote[] {
  return [
    ...notes.filter((note) => importantProofLabel(note.label)),
    ...notes.filter((note) => !importantProofLabel(note.label))
  ];
}

function importantProofLabel(label: string): boolean {
  const lowerLabel = label.toLowerCase();
  return (
    lowerLabel.includes("retain") ||
    lowerLabel.includes("direct") ||
    lowerLabel.includes("eligibility") ||
    lowerLabel.includes("source candidates")
  );
}

function normalizeTraceText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function stepKeyFromDecision(value: string): string {
  const beforeColon = value.split(":")[0] ?? "decision";
  if (beforeColon.length < 32) {
    return beforeColon;
  }
  return "decision";
}

function proofNotePreview(value: string): string {
  if (value.length <= PROOF_NOTE_PREVIEW_CHARS) {
    return value;
  }
  const sliced = value.slice(0, PROOF_NOTE_PREVIEW_CHARS);
  const wordBoundary = sliced.lastIndexOf(" ");
  const preview = wordBoundary > 90 ? sliced.slice(0, wordBoundary) : sliced;
  return `${preview.trim()}...`;
}
