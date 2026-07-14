import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ClarificationResponseRequest, QuestionStatePayload } from "../fervis-api/contracts";
import { App } from "./App";
import {
  completedAfterClarificationState,
  createEmptyConversationClient,
  createInteractiveClient,
  createInteractiveTextClarificationClient,
  initialQuestionState,
  textClarificationAnswerState
} from "./appTestSupport";

describe("Ledger app actions", () => {
  it("submits follow-up questions through the configured Fervis client", async () => {
    const askQuestion = vi.fn(async () => ({
      answer: null,
      conversationId: "conv_sales",
      primaryRunId: "run_followup",
      latestRunId: "run_followup",
      activeRunId: "run_followup",
      nextActions: [],
      question: "What about yesterday?",
      questionId: "q_followup",
      resultData: null,
      status: "RUNNING"
    } satisfies QuestionStatePayload));
    render(<App initialClient={createInteractiveClient({ askQuestion })} />);

    const input = await screen.findByLabelText("Ask a follow-up question");
    fireEvent.change(input, {
      target: { value: "What about yesterday?" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    const submittedQuestionLabels = await screen.findAllByText("What about yesterday?");
    expect(submittedQuestionLabels[0]).toBeInTheDocument();
    expect(askQuestion).toHaveBeenCalledWith({
      conversationId: "conv_sales",
      question: "What about yesterday?"
    });
  });

  it("submits the first question when no conversations exist yet", async () => {
    const askQuestion = vi.fn(async () => initialQuestionState);
    render(<App initialClient={createEmptyConversationClient({ askQuestion })} />);

    const input = await screen.findByLabelText("Ask a question");
    fireEvent.change(input, {
      target: { value: "How many in-person sales happened this month?" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    const labels = await screen.findAllByText(
      "How many in-person sales happened this month?"
    );
    expect(labels.length).toBeGreaterThan(0);
    expect(await screen.findByText("Inputs:")).toBeInTheDocument();
    expect(
      screen.getByText("\"this month\": 2026-06-01 to 2026-06-30")
    ).toBeInTheDocument();
    expect(askQuestion).toHaveBeenCalledWith({
      conversationId: null,
      question: "How many in-person sales happened this month?"
    });
  });

  it("starts a new independent question from the conversation rail", async () => {
    const askQuestion = vi.fn(async () => initialQuestionState);
    render(<App initialClient={createInteractiveClient({ askQuestion })} />);

    await screen.findByLabelText("Ask a follow-up question");
    fireEvent.click(screen.getByRole("button", { name: "New question" }));

    const input = await screen.findByLabelText("Ask a question");
    fireEvent.change(input, {
      target: { value: "How many in-person sales happened this month?" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(askQuestion).toHaveBeenCalledWith({
      conversationId: null,
      question: "How many in-person sales happened this month?"
    });
    expect(await screen.findByText("Inputs:")).toBeInTheDocument();
    expect(
      screen.getByText("\"this month\": 2026-06-01 to 2026-06-30")
    ).toBeInTheDocument();
  });

  it("submits choice clarification answers under the same question", async () => {
    const answerClarification = vi.fn(async () => completedAfterClarificationState);
    render(<App initialClient={createInteractiveClient({ answerClarification })} />);

    fireEvent.click(
      await screen.findByText("How many sales happened at the matching store?")
    );
    fireEvent.click(await screen.findByRole("radio", { name: "BBS Outlet" }));
    fireEvent.click(screen.getByRole("button", { name: "Send clarification" }));

    expect(await screen.findByText("18 in-person sales happened this month.")).toBeInTheDocument();
    expect(answerClarification).toHaveBeenCalledWith("q_choice_clarification", {
      clarificationId: "clar_store",
      responseText: "BBS Outlet",
      selectedOptionId: "store:store_id:70707070-0000-0000-0001-000000000002",
      runId: "run_clarify"
    } satisfies ClarificationResponseRequest);
  });

  it("shows immediate focus feedback while clarification resumes the run", async () => {
    const answerClarification = vi.fn(
      () =>
        new Promise<QuestionStatePayload>(() => {
          return;
        })
    );
    render(<App initialClient={createInteractiveClient({ answerClarification })} />);

    fireEvent.click(
      await screen.findByText("How many sales happened at the matching store?")
    );
    fireEvent.click(await screen.findByRole("radio", { name: "ABC Mall" }));
    fireEvent.click(screen.getByRole("button", { name: "Send clarification" }));

    expect(await screen.findByRole("status")).toHaveTextContent("answer sent");
    expect(screen.getByRole("status")).toHaveTextContent("ABC Mall");
    expect(screen.queryByRole("button", { name: "Send clarification" })).not.toBeInTheDocument();
  });

  it("does not expose transport mechanics inside clarification cards", async () => {
    render(<App initialClient={createInteractiveClient({})} />);

    fireEvent.click(
      await screen.findByText("How many sales happened at the matching store?")
    );
    expect(await screen.findByText("Which matching store should I use?")).toBeInTheDocument();
    expect(screen.queryByText(/POST \/questions/)).not.toBeInTheDocument();
    expect(screen.queryByText(/triggerKind/)).not.toBeInTheDocument();
  });

  it("submits free-text clarification answers under the same question", async () => {
    const answerClarification = vi.fn(async () => textClarificationAnswerState);
    render(<App initialClient={createInteractiveTextClarificationClient(answerClarification)} />);

    await screen.findByText("Which March should I use?");
    expect(screen.getByText("Ambiguous Interpretation · text answer requested")).toBeInTheDocument();
    expect(screen.queryByText(/step_clarify/)).not.toBeInTheDocument();
    expect(screen.queryByText(/fact_result/)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Clarification answer"), {
      target: { value: "March 2026" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("March 2026 sales were 14.")).toBeInTheDocument();
    expect(answerClarification).toHaveBeenCalledWith("q_clarification", {
      clarificationId: "clar_period",
      responseText: "March 2026",
      runId: "run_clarify_text"
    } satisfies ClarificationResponseRequest);
  });

  it("renders action failures through the main error surface", async () => {
    const askQuestion = vi.fn(async () => {
      throw new Error("Unauthorized Fervis request");
    });
    render(<App initialClient={createInteractiveClient({ askQuestion })} />);

    const input = await screen.findByLabelText("Ask a follow-up question");
    fireEvent.change(input, {
      target: { value: "What about yesterday?" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Fervis is not reachable")).toBeInTheDocument();
    expect(screen.getByText("Unauthorized Fervis request")).toBeInTheDocument();
  });
});
