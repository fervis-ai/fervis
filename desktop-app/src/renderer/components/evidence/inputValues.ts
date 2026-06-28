import { formatEvidenceText } from "../../textFormat";

export function formatInputValue(value: string): string {
  return softenInputKeyCase(
    formatEvidenceText(
      normalizeBooleanValue(removeEndpointArgumentPhrase(value))
    )
  );
}

function removeEndpointArgumentPhrase(value: string): string {
  const match = value.trim().match(/^(.+?)\s+was used as an endpoint argument\.?$/i);
  return match?.[1] ?? value;
}

function normalizeBooleanValue(value: string): string {
  return value
    .replace(/:\s*true\b/gi, ": yes")
    .replace(/:\s*false\b/gi, ": no")
    .replace(/=true\b/gi, "=yes")
    .replace(/=false\b/gi, "=no");
}

function softenInputKeyCase(value: string): string {
  return value.replace(/^([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)+):/, (match) => {
    const label = match.slice(0, -1).toLowerCase();
    return `${label[0]?.toUpperCase() ?? ""}${label.slice(1)}:`;
  });
}
