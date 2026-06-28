import type {
  LineageTimeline,
  RunPayload,
  StepSemantic
} from "../fervis-api/contracts";
import { formatEvidenceText, titleWords } from "./textFormat";

export interface StepSignal {
  readonly label: string;
  readonly text: string;
}

export interface SourceReferenceLabel {
  readonly ref: string;
  readonly resourceName: string;
  readonly label: string;
}

export function semanticStepSignalsFor(
  stepKey: string,
  semantic: StepSemantic
): readonly StepSignal[] {
  if (stepKey === "question_contract") {
    return compactSignals([
      signalFromValues(
        semantic.requestedFacts.length === 1 ? "Requested fact" : "Requested facts",
        semantic.requestedFacts.map((fact) => fact.description),
        3
      ),
      signalFromEnumeratedValues(
        "Known inputs",
        semantic.knownInputs.map((input) => input.text),
        3
      )
    ]);
  }
  if (stepKey === "conversation_resolution") {
    return compactSignals(
      semantic.conversationClauses.flatMap(conversationClauseSignals)
    );
  }
  if (stepKey === "query_enrichment") {
    return compactSignals([
      ...semantic.resolverCandidates.map((candidate) =>
        semanticSignal("Resolver", resolverCandidateText(candidate))
      )
    ]);
  }
  if (stepKey === "grounding") {
    return compactSignals([
      ...semantic.interpretedInputs.map((input) =>
        semanticSignal("Interpreted input", interpretedInputText(input))
      ),
      ...semantic.groundingResults.map((result) =>
        semanticSignal("Matched entity", groundingResultText(result))
      ),
      ...semantic.resolverCandidates.map((candidate) =>
        semanticSignal("Resolver", resolverCandidateText(candidate))
      )
    ]);
  }
  return [];
}

export function liveStepHighlightsFor(
  stepKey: string,
  decisionLines: readonly string[],
  semantic: StepSemantic,
  sourceLabels: readonly SourceReferenceLabel[] = []
): readonly StepSignal[] {
  const semanticSignals = semanticStepSignalsFor(stepKey, semantic);
  if (semanticSignals.length > 0) {
    if (stepKey === "conversation_resolution") {
      return semanticSignals;
    }
    return semanticSignals.slice(0, 2);
  }
  if (decisionLines.length === 0) {
    return [];
  }
  const decisionSignals = compactDecisionSignals(decisionLines, sourceLabels);
  if (decisionSignals.length > 0) {
    return decisionSignals.slice(0, 2);
  }
  return decisionLines.slice(0, 2).map((line) => ({
    label: "Decision",
    text: formatEvidenceTextWithSourceReferences(line, sourceLabels)
  }));
}

export function sourceReferenceLabelsFor(
  run: RunPayload,
  additionalDecisionLines: readonly string[] = []
): readonly SourceReferenceLabel[] {
  return uniqueSourceLabels(
    [
      ...decisionLinesFromTimeline(run.explanation.lineage.compact),
      ...decisionLinesFromTimeline(run.explanation.lineage.verbose),
      ...additionalDecisionLines
    ]
      .map(sourceReferenceFromDecision)
      .filter((label): label is SourceReferenceLabel => label !== null)
  );
}

export function labelSourceReferences(
  value: string,
  labels: readonly SourceReferenceLabel[]
): string {
  let labeled = value;
  for (const source of labels) {
    const sourceNumber = source.ref.replace("source_", "");
    labeled = labeled
      .replace(new RegExp(`\\b${source.ref}\\b`, "g"), source.label)
      .replace(new RegExp(`\\bsource\\s+${sourceNumber}\\b`, "gi"), source.label);
  }
  return labeled;
}

export function sourceReferenceLabelFor(
  ref: string,
  labels: readonly SourceReferenceLabel[]
): string | null {
  return labels.find(
    (source) => source.ref.toLowerCase() === ref.toLowerCase()
  )?.label ?? null;
}

export function formatEvidenceTextWithSourceReferences(
  value: string,
  labels: readonly SourceReferenceLabel[]
): string {
  const replacements = new Map<string, string>();
  const protectedValue = labelSourceReferences(value, labels).replace(
    /\bsource_\d+\s+\([^)]+\)/g,
    (match) => {
      const key = `FERVISSOURCELABELTOKEN${replacements.size}`;
      replacements.set(key, match);
      return key;
    }
  );
  let formatted = formatEvidenceText(protectedValue);
  for (const [key, label] of replacements) {
    formatted = formatted.replace(key, label);
  }
  return formatted;
}

function compactSignals(
  candidates: readonly (StepSignal | null)[]
): readonly StepSignal[] {
  return candidates.filter((signal): signal is StepSignal => signal !== null);
}

function signalFromValues(
  label: string,
  values: readonly string[],
  limit: number
): StepSignal | null {
  const uniqueValues = unique(values.map(formatSignalValue).filter((value) => value !== ""));
  if (uniqueValues.length === 0) {
    return null;
  }
  return {
    label,
    text: compactList(uniqueValues, limit)
  };
}

function signalFromEnumeratedValues(
  label: string,
  values: readonly string[],
  limit: number
): StepSignal | null {
  const uniqueValues = unique(values.map((value) => value.trim()).filter((value) => value !== ""));
  if (uniqueValues.length === 0) {
    return null;
  }
  return {
    label,
    text: enumeratedList(uniqueValues, limit)
  };
}

function semanticSignal(label: string, text: string): StepSignal | null {
  if (text.trim() === "") {
    return null;
  }
  return { label, text };
}

function compactDecisionSignals(
  lines: readonly string[],
  sourceLabels: readonly SourceReferenceLabel[]
): readonly StepSignal[] {
  const signals: StepSignal[] = [];
  const eligibility = lines
    .map((line) => line.match(/retained (\d+) source candidates?, dropped (\d+)/i))
    .find((match) => match !== null);
  if (eligibility !== undefined && eligibility !== null) {
    signals.push({
      label: "Eligibility",
      text: `${eligibility[1]} retained, ${eligibility[2]} dropped`
    });
  }

  const usedResources = unique(
    lines
      .map((line) => {
        const sourceMatch = line.match(/\bsource[\s_-]*(\d+)\s+([^:]+):\s*(RETAIN|DIRECT|REFERENCE)\b/i);
        if (sourceMatch !== null) {
          return titleWords(sourceMatch[2] ?? "");
        }
        const bareSourceMatch = line.match(/\bsource[\s_-]*(\d+)\s*:\s*(RETAIN|DIRECT|REFERENCE)\b/i);
        if (bareSourceMatch !== null) {
          return sourceReferenceLabelFor(
            `source_${bareSourceMatch[1] ?? ""}`,
            sourceLabels
          ) ?? "";
        }
        const fitMatch = line.match(/^([^:]+):\s+.+->\s+FITS_REQUESTED_ANSWER\b/i);
        if (fitMatch !== null) {
          return titleWords(fitMatch[1] ?? "");
        }
        return "";
      })
      .filter((resource) => resource !== "")
  );
  if (usedResources.length > 0) {
    signals.push({
      label: usedResources.length === 1 ? "Resource" : "Resources",
      text: compactList(usedResources, 3)
    });
  }
  return signals;
}

function resolverCandidateText(candidate: StepSemantic["resolverCandidates"][number]): string {
  const resolver = candidate.resolverLabel || titleWords(candidate.resolverReadId);
  if (candidate.basis === "") {
    return resolver;
  }
  if (resolver === "") {
    return formatEvidenceText(candidate.basis);
  }
  return `${resolver}: ${formatEvidenceText(candidate.basis)}`;
}

function groundingResultText(result: StepSemantic["groundingResults"][number]): string {
  const input = result.inputText || result.matchedLabel;
  const resolver = result.resolverLabel || titleWords(result.resolverReadId);
  const matched = `${result.matchedField}=${result.matchedValue}`;
  return `${input}: ${resolver} matched ${matched}`;
}

function interpretedInputText(input: StepSemantic["interpretedInputs"][number]): string {
  const source = input.inputText || input.label;
  if (source === "") {
    return input.value;
  }
  return `${source}: ${input.value}`;
}

function conversationClauseSignals(
  clause: StepSemantic["conversationClauses"][number]
): readonly StepSignal[] {
  return compactSignals([
    semanticSignal("Current clause", clause.currentClauseText),
    semanticSignal("Resolved value", resolvedValueText(clause)),
    semanticSignal("Resolved question", clause.resolvedClauseText)
  ]);
}

function resolvedValueText(
  clause: StepSemantic["conversationClauses"][number]
): string {
  if (clause.currentValueText === "" || clause.resolvedFrameText === "") {
    return "";
  }
  return `${clause.currentValueText} -> ${clause.resolvedFrameText}`;
}

function sourceReferenceFromDecision(line: string): SourceReferenceLabel | null {
  const match = line.match(/\bsource[\s_-]*(\d+)\s+([^:]+):\s*(RETAIN|DROP|DIRECT|REFERENCE|SKIP)\b/i);
  if (match === null) {
    return null;
  }
  const ref = `source_${match[1] ?? ""}`;
  const resourceName = (match[2] ?? "").trim();
  if (ref === "" || resourceName === "") {
    return null;
  }
  return {
    ref,
    resourceName,
    label: `${ref} (${titleWords(resourceName)})`
  };
}

function uniqueSourceLabels(
  labels: readonly SourceReferenceLabel[]
): readonly SourceReferenceLabel[] {
  const seen = new Set<string>();
  const uniqueLabels: SourceReferenceLabel[] = [];
  for (const label of labels) {
    const key = label.ref.toLowerCase();
    if (!seen.has(key)) {
      uniqueLabels.push(label);
      seen.add(key);
    }
  }
  return uniqueLabels;
}

function decisionLinesFromTimeline(timeline: LineageTimeline): readonly string[] {
  return timeline.questions.flatMap((question) =>
    question.runs.flatMap((run) =>
      run.steps.flatMap((step) =>
        step.decisions.flatMap((decision) => decision.lines)
      )
    )
  );
}

function formatSignalValue(value: string): string {
  return formatEvidenceText(value);
}

function compactList(values: readonly string[], limit: number): string {
  const visible = values.slice(0, limit);
  const hiddenCount = values.length - visible.length;
  if (hiddenCount <= 0) {
    return visible.join(", ");
  }
  return `${visible.join(", ")} +${hiddenCount} more`;
}

function enumeratedList(values: readonly string[], limit: number): string {
  const visible = values.slice(0, limit);
  const hiddenCount = values.length - visible.length;
  const prefix = visible
    .map((value, index) => `${index + 1}: ${value}`)
    .join(" · ");
  if (hiddenCount <= 0) {
    return prefix;
  }
  return `${prefix} · +${hiddenCount} more`;
}

function unique(values: readonly string[]): readonly string[] {
  return [...new Set(values)];
}
