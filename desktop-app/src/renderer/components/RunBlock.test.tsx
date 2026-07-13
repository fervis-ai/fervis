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
