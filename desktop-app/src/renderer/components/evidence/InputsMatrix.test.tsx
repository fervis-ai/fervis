import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { InputResult } from "../../../fervis-api/contracts";
import { InputsMatrix } from "./InputsMatrix";

describe("InputsMatrix", () => {
  it("shows human input signals without applied or technical reference noise", () => {
    render(<InputsMatrix inputs={[inputWithLongAppliedContext()]} />);

    fireEvent.click(screen.getByText("Count sales").closest("summary") ?? failMissingSummary());

    expect(screen.getByText("This month")).toBeInTheDocument();
    expect(screen.getByText("Count rows")).toBeInTheDocument();
    expect(screen.getByText("Include items: yes")).toBeInTheDocument();
    expect(screen.queryByText("Applied")).not.toBeInTheDocument();
    expect(screen.queryByText("Technical refs")).not.toBeInTheDocument();
    expect(screen.queryByText("Applied value 7")).not.toBeInTheDocument();
    expect(screen.queryByText("Proof: answer 1.source 1.query.status")).not.toBeInTheDocument();
  });
});

function inputWithLongAppliedContext(): InputResult {
  return {
    applied: [
      "status: completed",
      "include_items=true",
      "limit: 50",
      "offset: 0",
      "is deleted: false",
      "include items: true",
      "applied value 7"
    ],
    contextual: ["include_items=true was used as an endpoint argument"],
    derived: ["count rows"],
    evidenceRefs: ["source_1.row.data"],
    explicit: ["this month"],
    factDescription: "Count sales",
    factResultId: "fact_1",
    proofHandles: ["answer_1.source_1.query.status"],
    requestedFactId: "requested_fact_1"
  };
}

function failMissingSummary(): HTMLElement {
  throw new Error("expected input summary");
}
