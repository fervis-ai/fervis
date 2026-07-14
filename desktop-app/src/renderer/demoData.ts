import {
  clarificationRunFixture,
  completedRunFixture,
  failedRunFixture,
  freeTextClarificationRunFixture,
  runningRunFixture
} from "../fervis-api/__fixtures__/payloads";
import type { RunPayload } from "../fervis-api/contracts";
import type { ConversationDetails } from "./viewTypes";

const choiceClarificationDemoRun = {
  ...clarificationRunFixture,
  conversationId: "conv_choice_clarification",
  questionId: "q_choice_clarification"
} satisfies RunPayload;

const runningDemoRun = {
  ...runningRunFixture,
  conversationId: "conv_running",
  questionId: "q_running",
  nextActions: [
    {
      kind: "inspect_question",
      description: "Inspect the question and all runs attempted for it.",
      command: "fervis explain --question-id q_running",
      request: null
    }
  ]
} satisfies RunPayload;

const textClarificationDemoRun = {
  ...freeTextClarificationRunFixture,
  conversationId: "conv_clarification",
  questionId: "q_clarification",
  runId: "run_clarify_text",
  nextActions: [
    {
      kind: "provide_clarification",
      description: "Continue the same question by answering the clarification.",
      command:
        'fervis runtime ask "<answer>" --question-id q_clarification --run-id run_clarify_text --clarification-id clar_period',
      request: null
    }
  ]
} satisfies RunPayload;

const failedDemoRun = {
  ...failedRunFixture,
  conversationId: "conv_failed",
  questionId: "q_failed",
  nextActions: [
    {
      kind: "inspect_question",
      description: "Inspect the question and all runs attempted for it.",
      command: "fervis explain --question-id q_failed --debug",
      request: null
    }
  ]
} satisfies RunPayload;

export const demoConversations: readonly ConversationDetails[] = [
  {
    summary: {
      conversationId: "conv_sales",
      firstQuestion: "How many in-person sales happened this month?",
      latestQuestionId: "q_sales",
      primaryRunId: "run_sales",
      latestRunId: "run_sales",
      activeRunId: null,
      status: "COMPLETED",
      runCount: 1,
      updatedAt: "2026-06-27T10:15:00+00:00"
    },
    question: "How many in-person sales happened this month?",
    runs: [completedRunFixture]
  },
  {
    summary: {
      conversationId: "conv_choice_clarification",
      firstQuestion: "How many sales happened at the matching store?",
      latestQuestionId: "q_choice_clarification",
      primaryRunId: null,
      latestRunId: "run_clarify",
      activeRunId: "run_clarify",
      status: "WAITING_FOR_CLARIFICATION",
      runCount: 1,
      updatedAt: "2026-06-27T10:15:30+00:00"
    },
    question: "How many sales happened at the matching store?",
    runs: [choiceClarificationDemoRun]
  },
  {
    summary: {
      conversationId: "conv_running",
      firstQuestion: "Which store has the most inventory at risk today?",
      latestQuestionId: "q_running",
      primaryRunId: "run_running",
      latestRunId: "run_running",
      activeRunId: "run_running",
      status: "RUNNING",
      runCount: 1,
      updatedAt: "2026-06-27T10:16:00+00:00"
    },
    question: "Which store has the most inventory at risk today?",
    runs: [runningDemoRun]
  },
  {
    summary: {
      conversationId: "conv_clarification",
      firstQuestion: "What were sales for BBS last month?",
      latestQuestionId: "q_clarification",
      primaryRunId: "run_clarify_text",
      latestRunId: "run_clarify_text",
      activeRunId: "run_clarify_text",
      status: "WAITING_FOR_CLARIFICATION",
      runCount: 1,
      updatedAt: "2026-06-27T10:17:00+00:00"
    },
    question: "What were sales for BBS last month?",
    runs: [textClarificationDemoRun]
  },
  {
    summary: {
      conversationId: "conv_failed",
      firstQuestion: "Which returns endpoint failed during settlement review?",
      latestQuestionId: "q_failed",
      primaryRunId: "run_failed",
      latestRunId: "run_failed",
      activeRunId: null,
      status: "FAILED",
      runCount: 1,
      updatedAt: "2026-06-27T10:18:00+00:00"
    },
    question: "Which returns endpoint failed during settlement review?",
    runs: [failedDemoRun]
  }
];
