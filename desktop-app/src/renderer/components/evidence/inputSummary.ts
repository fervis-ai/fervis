import type { InputResult } from "../../../fervis-api/contracts";

export function inputSummary(input: InputResult): string {
  const parts = [
    countLabel(input.explicit.length, "explicit"),
    countLabel(input.derived.length, "derived"),
    countLabel(input.contextual.length, "contextual")
  ].filter((value) => value !== "");
  return parts.length === 0 ? "no values" : parts.join(" · ");
}

function countLabel(count: number, label: string): string {
  return count === 0 ? "" : `${count} ${label}`;
}
