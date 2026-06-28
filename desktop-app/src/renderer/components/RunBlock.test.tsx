import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { completedRunFixture } from "../../fervis-api/__fixtures__/payloads";
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
            outputs: [{ key: "answer_1", valueKind: "number", value: "13" }]
          }
        }}
      />
    );

    expect(screen.getByText("13", { selector: ".answer-prose" })).toBeInTheDocument();
    expect(screen.queryByText("no answer produced")).not.toBeInTheDocument();
  });
});
