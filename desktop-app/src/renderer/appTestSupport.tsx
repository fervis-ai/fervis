import { render } from "@testing-library/react";
import { vi } from "vitest";

import type { FervisApiClient } from "../fervis-api/client";
import type {
  QuestionStatePayload,
  RunPayload
} from "../fervis-api/contracts";
import {
  completedRunFixture,
  conversationListFixture,
  freeTextClarificationRunFixture,
  questionStateFixture,
  runningRunFixture
} from "../fervis-api/__fixtures__/payloads";
import { App } from "./App";
import { createDemoFervisClient } from "./demoClient";

export function renderDemoApp() {
  render(<App initialClient={createDemoFervisClient()} />);
}

export function httpPayloadFor(url: string): object {
  if (url.endsWith("/conversations/")) {
    return conversationListFixture;
  }
  if (url.endsWith("/questions/q_new/")) {
    return { ...questionStateFixture, conversationId: "conv_new", questionId: "q_new" };
  }
  if (url.endsWith("/questions/q_new/runs/")) {
    return {
      questionId: "q_new",
      runs: [{ ...completedRunFixture, conversationId: "conv_new", questionId: "q_new" }]
    };
  }
  throw new Error(`Unexpected test URL ${url}`);
}

export const followUpQuestionState = {
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
} satisfies QuestionStatePayload;

export const initialQuestionState = {
  ...followUpQuestionState,
  conversationId: "conv_initial",
  question: "How many in-person sales happened this month?",
  questionId: "q_initial"
} satisfies QuestionStatePayload;

export const completedAfterClarificationState = {
  answer: completedRunFixture.answer,
  conversationId: "conv_choice_clarification",
  primaryRunId: "run_clarify",
  latestRunId: "run_clarify",
  activeRunId: null,
  nextActions: completedRunFixture.nextActions,
  question: "How many sales happened at the matching store?",
  questionId: "q_choice_clarification",
  resultData: completedRunFixture.resultData,
  status: "COMPLETED"
} satisfies QuestionStatePayload;

export const textClarificationAnswerState = {
  answer: "March 2026 sales were 14.",
  conversationId: "conv_clarification",
  primaryRunId: "run_clarify_text",
  latestRunId: "run_clarify_text",
  activeRunId: null,
  nextActions: completedRunFixture.nextActions,
  question: "What were sales for BBS last month?",
  questionId: "q_clarification",
  resultData: completedRunFixture.resultData,
  status: "COMPLETED"
} satisfies QuestionStatePayload;

export function createInteractiveClient({
  askQuestion = vi.fn(async () => followUpQuestionState),
  answerClarification = vi.fn(async () => completedAfterClarificationState)
}: {
  readonly askQuestion?: FervisApiClient["askQuestion"];
  readonly answerClarification?: FervisApiClient["answerClarification"];
}): FervisApiClient {
  const demoClient = createDemoFervisClient();
  let choiceClarificationAnswered = false;
  return {
    ...demoClient,
    getRun: demoClient.getRun,
    askQuestion,
    answerClarification: async (questionId, request) => {
      const result = await answerClarification(questionId, request);
      choiceClarificationAnswered = true;
      return result;
    },
    listConversations: async () => {
      const payload = await demoClient.listConversations();
      if (!choiceClarificationAnswered) {
        return payload;
      }
      return {
        conversations: payload.conversations.map((conversation) =>
          conversation.conversationId === "conv_choice_clarification"
            ? {
                ...conversation,
                activeRunId: null,
                primaryRunId: "run_clarify",
                status: "COMPLETED"
              }
            : conversation
        )
      };
    },
    listQuestionRuns: async (questionId) => {
      if (questionId === "q_followup" || questionId === "q_initial") {
        return { questionId, runs: [{ ...runningRunFixture, questionId }] };
      }
      if (questionId === "q_choice_clarification" && choiceClarificationAnswered) {
        return {
          questionId,
          runs: [
            {
              ...completedRunFixture,
              conversationId: "conv_choice_clarification",
              questionId,
              runId: "run_clarify"
            }
          ]
        };
      }
      return demoClient.listQuestionRuns(questionId);
    }
  };
}

export function createEmptyConversationClient({
  askQuestion
}: {
  readonly askQuestion: FervisApiClient["askQuestion"];
}): FervisApiClient {
  const demoClient = createDemoFervisClient();
  return {
    ...demoClient,
    askQuestion,
    listConversations: async () => ({ conversations: [] }),
    listQuestionRuns: async (questionId) => ({
      questionId,
      runs: [
        {
          ...runningRunFixture,
          conversationId: "conv_initial",
          questionId
        }
      ]
    })
  };
}

export function createInteractiveTextClarificationClient(
  answerClarification: FervisApiClient["answerClarification"]
): FervisApiClient {
  const demoClient = createDemoFervisClient();
  let answered = false;
  return {
    ...demoClient,
    getRun: demoClient.getRun,
    answerClarification: async (questionId, request) => {
      const result = await answerClarification(questionId, request);
      answered = true;
      return result;
    },
    listConversations: async () => ({
      conversations: [
        {
          conversationId: "conv_clarification",
          primaryRunId: "run_clarify_text",
          latestRunId: "run_clarify_text",
          activeRunId: answered ? null : "run_clarify_text",
          firstQuestion: "What were sales for BBS last month?",
          latestQuestionId: "q_clarification",
          runCount: 1,
          status: answered ? "COMPLETED" : "WAITING_FOR_CLARIFICATION",
          updatedAt: "2026-06-27T10:17:00+00:00"
        }
      ]
    }),
    listQuestionRuns: async (questionId) => {
      if (questionId === "q_clarification") {
        return {
          questionId,
          runs: answered
            ? [
                {
                  ...completedRunFixture,
                  answer: "March 2026 sales were 14.",
                  conversationId: "conv_clarification",
                  questionId: "q_clarification",
                  runId: "run_clarify_text"
                } satisfies RunPayload
              ]
            : [
                {
                  ...freeTextClarificationRunFixture,
                  conversationId: "conv_clarification",
                  questionId: "q_clarification",
                  runId: "run_clarify_text"
                }
              ]
        };
      }
      return demoClient.listQuestionRuns(questionId);
    }
  };
}

export function createPollingClient({
  getRun
}: {
  readonly getRun: FervisApiClient["getRun"];
}): FervisApiClient {
  const demoClient = createDemoFervisClient();
  return {
    ...demoClient,
    getRun,
    listConversations: async () => ({
      conversations: [
        {
          conversationId: "conv_running",
          primaryRunId: "run_running",
          latestRunId: "run_running",
          activeRunId: "run_running",
          firstQuestion: "Which store has the most inventory at risk today?",
          latestQuestionId: "q_running",
          runCount: 1,
          status: "RUNNING",
          updatedAt: "2026-06-27T10:16:00+00:00"
        }
      ]
    })
  };
}

export function failMissingEvidenceLabel(): HTMLElement {
  throw new Error("expected Evidence label");
}
