import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  completedContinuedRunFixture,
  completedRunFixture,
  runningContinuedRunFixture
} from "../../fervis-api/__fixtures__/payloads";
import { RunBlock } from "./RunBlock";

describe("RunBlock", () => {
  it("renders structured answer output when plain answer text is absent", () => {
    render(
      <RunBlock
        apiClient={null}
        onActionError={vi.fn()}
        onClarificationState={vi.fn()}
        onToggle={vi.fn()}
        open
        run={{
          ...completedRunFixture,
          answer: null,
          resultData: {
            kind: "answer",
            outputs: [
              {
                key: "answer_1",
                valueKind: "number",
                value: { kind: "number", value: "13" },
                displayValue: "13"
              }
            ]
          }
        }}
      />
    );

    expect(screen.getByText("13", { selector: ".answer-prose" })).toBeInTheDocument();
    expect(screen.queryByText("no answer produced")).not.toBeInTheDocument();
  });

  it.each([
    ["running", runningContinuedRunFixture],
    ["completed", completedContinuedRunFixture]
  ] as const)("identifies a %s callable prior-request run", (_state, run) => {
    render(
      <RunBlock
        apiClient={null}
        onActionError={vi.fn()}
        onClarificationState={vi.fn()}
        onToggle={vi.fn()}
        open
        run={run}
      />
    );

    expect(screen.getByText(/Continue Prior Request/)).toBeInTheDocument();
  });
});

it("renders a decoded factual limitation without requiring an answer record", async () => {
  const { decodeRun } = await import("../../fervis-api/decoder");
  const { default: results } = await import("../../../../python/tests/contracts/fixtures/public_terminal_results.json");
  const decoded = decodeRun({ ...completedRunFixture, answer: null, resultData: results.impossible });
  if (!decoded.ok) throw new Error(decoded.error.message);
  render(<RunBlock apiClient={null} onActionError={vi.fn()} onClarificationState={vi.fn()} onToggle={vi.fn()} open run={decoded.value} />);
  expect(screen.getByText("The requested observation is unavailable.", { selector: ".answer-prose" })).toBeInTheDocument();
});

it("preserves equal values from distinct requested outputs in public prose", () => {
  render(<RunBlock apiClient={null} onActionError={vi.fn()} onClarificationState={vi.fn()} onToggle={vi.fn()} open run={{ ...completedRunFixture, answer: "3\n3" }} />);
  expect(screen.getByText("3 3", { selector: ".answer-prose" })).toBeInTheDocument();
});
