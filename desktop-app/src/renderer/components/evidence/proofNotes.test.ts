import { describe, expect, it } from "vitest";

import type { LineageStep } from "../../../fervis-api/contracts";
import {
  completedRunFixture,
  emptyStepSemanticFixture
} from "../../../fervis-api/__fixtures__/payloads";
import { proofNotes } from "./proofNotes";

describe("proof notes", () => {
  it("renders source-selection rationale without model-facing row/field prefixes", () => {
    const notes = proofNotes(
      lineageStepWithDecision(
        "source_1 list_sale_list: RETAIN - rows=2 - fields=7 - Exposes sale rows with sold_at, status, is_deleted, and sale_type, which can support counting."
      ),
      "verbose",
      completedRunFixture
    );

    expect(notes[0]).toEqual({
      label: "source_1 (List Sale List) · used",
      text: "Exposes sale rows with sold at, status, is deleted, and sale type, which can support counting."
    });
  });

  it("labels reviewed source handles with endpoint names", () => {
    const notes = proofNotes(
      lineageStepWithDecisions([
        "source_1 list_sale_list: RETAIN - rows=2 - fields=7 - Exposes sale rows.",
        "source_2 list_sales_summary: RETAIN - rows=3 - fields=12 - Exposes sales summaries.",
        "Reviewed source candidates: source_1, source_2"
      ]),
      "verbose",
      completedRunFixture
    );

    expect(notes.find((note) => note.label === "Source candidates")).toEqual({
      label: "Source candidates",
      text: "Reviewed source_1 (List Sale List), source_2 (List Sales Summary)."
    });
  });
});

function lineageStepWithDecision(line: string): LineageStep {
  return lineageStepWithDecisions([line]);
}

function lineageStepWithDecisions(lines: readonly string[]): LineageStep {
  return {
    decisions: lines.map((line) => ({ lines: [line] })),
    runtimeErrors: [],
    semantic: emptyStepSemanticFixture,
    sequence: 1,
    sourceReads: [],
    stepId: "step_1",
    stepKey: "read_eligibility"
  };
}
