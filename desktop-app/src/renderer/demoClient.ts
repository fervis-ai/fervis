import type { FervisApiClient } from "../fervis-api/client";
import type {
  ConversationListPayload,
  QuestionRunListPayload,
  QuestionStatePayload
} from "../fervis-api/contracts";
import { demoConversations } from "./demoData";

export function createDemoFervisClient(): FervisApiClient {
  return {
    listConversations: async (): Promise<ConversationListPayload> => ({
      conversations: demoConversations.map((conversation) => conversation.summary)
    }),
    getQuestion: async (questionId: string): Promise<QuestionStatePayload> => {
      const conversation = conversationByQuestion(questionId);
      const latestRun = conversation.runs[conversation.runs.length - 1];
      return {
        questionId,
        conversationId: conversation.summary.conversationId,
        question: conversation.question,
        primaryRunId: conversation.summary.primaryRunId,
        latestRunId: conversation.summary.latestRunId,
        activeRunId: conversation.summary.activeRunId,
        status: latestRun.status,
        answer: latestRun.answer,
        resultData: latestRun.resultData,
        nextActions: latestRun.nextActions
      };
    },
    listQuestionRuns: async (questionId: string): Promise<QuestionRunListPayload> => {
      const conversation = conversationByQuestion(questionId);
      return {
        questionId,
        runs: conversation.runs
      };
    },
    getRun: async (questionId: string, runId: string) => {
      const conversation = conversationByQuestion(questionId);
      const run = conversation.runs.find((candidate) => candidate.runId === runId);
      if (run === undefined) {
        throw new Error(`Demo run not found for run ${runId}`);
      }
      return run;
    },
    askQuestion: async (): Promise<QuestionStatePayload> =>
      createDemoFervisClient().getQuestion(demoConversations[0].summary.latestQuestionId),
    answerClarification: async (): Promise<QuestionStatePayload> =>
      createDemoFervisClient().getQuestion(demoConversations[0].summary.latestQuestionId),
    rerunQuestion: async (): Promise<QuestionStatePayload> =>
      createDemoFervisClient().getQuestion(demoConversations[0].summary.latestQuestionId)
  };
}

function conversationByQuestion(questionId: string) {
  const conversation = demoConversations.find(
    (candidate) => candidate.summary.latestQuestionId === questionId
  );
  if (conversation === undefined) {
    throw new Error(`Demo conversation not found for question ${questionId}`);
  }
  return conversation;
}
