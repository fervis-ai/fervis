import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  completedRunFixture,
  runningRunFixture
} from "../fervis-api/__fixtures__/payloads";
import { App } from "./App";
import { createPollingClient } from "./appTestSupport";

describe("Ledger app polling", () => {
  it("polls the active running run until it reaches a terminal answer", async () => {
    const currentRun = {
      ...runningRunFixture,
      conversationId: "conv_running",
      questionId: "q_running",
      runId: "run_running"
    };
    const terminalRun = {
      ...completedRunFixture,
      conversationId: "conv_running",
      questionId: "q_running",
      runId: "run_running"
    };
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(currentRun)
      .mockResolvedValue(terminalRun);
    render(<App initialClient={createPollingClient({ getRun })} />);

    expect(
      await screen.findByText(
        "Interpreted input: this month: 2026-06-01 to 2026-06-30"
      )
    ).toBeInTheDocument();
    expect(screen.getByText("Evidence").closest("details")).not.toHaveAttribute(
      "open"
    );

    expect(
      await screen.findByText(
        "18 in-person sales happened this month.",
        {},
        { timeout: 2500 }
      )
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Evidence").closest("details")).toHaveAttribute(
        "open"
      );
    });
    expect(getRun).toHaveBeenCalledWith("q_running", "run_running");
  });

  it("keeps the run view open when polling has a transient failure", async () => {
    const getRun = vi.fn(async () => {
      throw new Error("temporary polling failure");
    });
    render(<App initialClient={createPollingClient({ getRun })} />);

    expect(
      await screen.findByText(
        "Interpreted input: this month: 2026-06-01 to 2026-06-30"
      )
    ).toBeInTheDocument();

    expect(await screen.findByText(/polling paused · temporary polling failure/)).toBeInTheDocument();
    expect(
      screen.getByText(
        "Interpreted input: this month: 2026-06-01 to 2026-06-30"
      )
    ).toBeInTheDocument();
  });

  it("keeps polling when a completed run is not yet renderable", async () => {
    const currentRun = {
      ...runningRunFixture,
      conversationId: "conv_running",
      questionId: "q_running",
      runId: "run_running"
    };
    const incompleteTerminalRun = {
      ...currentRun,
      status: "COMPLETED",
      steps: completedRunFixture.steps
    } as const;
    const terminalRun = {
      ...completedRunFixture,
      conversationId: "conv_running",
      questionId: "q_running",
      runId: "run_running"
    };
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(incompleteTerminalRun)
      .mockResolvedValue(terminalRun);
    render(<App initialClient={createPollingClient({ getRun })} />);

    await waitFor(() => expect(getRun).toHaveBeenCalledTimes(1));
    expect(screen.queryByText("no answer produced")).not.toBeInTheDocument();

    expect(
      await screen.findByText(
        "18 in-person sales happened this month.",
        {},
        { timeout: 3000 }
      )
    ).toBeInTheDocument();
    expect(screen.queryByText("no answer produced")).not.toBeInTheDocument();
    expect(getRun).toHaveBeenCalledTimes(2);
  });
});
