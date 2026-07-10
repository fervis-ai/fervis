import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  clarificationRunFixture,
  completedRunFixture
} from "../../fervis-api/__fixtures__/payloads";
import type { RunPayload } from "../../fervis-api/contracts";
import { createDemoFervisClient } from "../demoClient";
import { ConversationLedger } from "./ConversationLedger";

describe("ConversationLedger", () => {
  it("opens the primary model-assisted run while keeping a newer rerun selectable", async () => {
    const deterministicRun = {
      ...completedRunFixture,
      runId: "run_variant",
      runNumber: 3,
      kind: "deterministic",
      triggerKind: "rerun",
      baseRunId: completedRunFixture.runId,
      invocationId: "pi_variant",
      patchId: "bp_variant",
      answer: "20 sales including placed orders."
    } satisfies RunPayload;

    const askQuestion = vi.fn(createDemoFervisClient().askQuestion);
    const apiClient = { ...createDemoFervisClient(), askQuestion };
    render(
      <ConversationLedger
        apiClient={apiClient}
        conversation={{
          summary: {
            conversationId: "conv_sales",
            firstQuestion: "How many sales did we make today?",
            latestQuestionId: "q_sales",
            primaryRunId: completedRunFixture.runId,
            latestRunId: deterministicRun.runId,
            activeRunId: null,
            status: "COMPLETED",
            runCount: 3,
            updatedAt: "2026-07-10T00:00:00+03:00"
          },
          question: "How many sales did we make today?",
          runs: [clarificationRunFixture, completedRunFixture, deterministicRun]
        }}
        pollingErrorMessage={null}
        onActionError={vi.fn()}
        onQuestionState={vi.fn(async () => undefined)}
      />
    );

    expect(
      await screen.findByText("18 in-person sales happened this month.")
    ).toBeInTheDocument();
    expect(screen.queryByText("20 sales including placed orders.")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /run_variant/ }));

    expect(
      await screen.findByText("20 sales including placed orders.")
    ).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Ask a follow-up question" }), {
      target: { value: "What about yesterday?" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(askQuestion).toHaveBeenCalledWith({
      contextRunId: "run_variant",
      conversationId: "conv_sales",
      question: "What about yesterday?"
    });
  });
});
