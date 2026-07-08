export function formatStepKey(value: string): string {
  return titleWords(value);
}

export function formatTriggerKind(value: string): string {
  return titleWords(value);
}

export function formatClarificationReason(value: string): string {
  return titleWords(value);
}

export function formatEvidenceText(value: string): string {
  const withoutDebugIds = value
    .replace(/\b[a-z]+(?:_[a-z]+)*_[0-9a-f]{12,}\b/gi, "reference")
    .replace(/\b[a-z]+(?:_[a-z]+)*:[a-z0-9_./-]+\b/gi, (match) =>
      match.split(":").map(titleWords).join(": ")
    )
    .replace(/\b[a-z]+(?:_[a-z]+)*=/gi, (match) =>
      `${titleWords(match.slice(0, -1))}: `
    )
    .replace(/_/g, " ");
  return sentenceCase(withoutDebugIds.replace(/\s+/g, " ").trim());
}

export function formatRoutePath(path: string): string {
  return path.replace(/_/g, "-");
}

export function titleWords(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function sentenceCase(value: string): string {
  if (value.length === 0) {
    return value;
  }
  return `${value[0]?.toUpperCase() ?? ""}${value.slice(1)}`;
}

export function sentenceWithPeriod(value: string): string {
  const trimmed = value.trim();
  if (trimmed === "") {
    return "";
  }
  if (/[.!?]$/.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed}.`;
}
