import type { FervisApiClient } from "../fervis-api/client";
import type {
  ConversationSummary,
  RunPayload,
  RunStatus
} from "../fervis-api/contracts";
import type { ConversationDetails, NonEmptyRuns, ThemeMode } from "./viewTypes";

export type LoadState = "idle" | "loading" | "ready" | "failed";

export async function loadConversation(
  client: FervisApiClient,
  summary: ConversationSummary
): Promise<ConversationDetails> {
  const [question, runs] = await Promise.all([
    client.getQuestion(summary.latestQuestionId),
    client.listQuestionRuns(summary.latestQuestionId)
  ]);
  return {
    summary,
    question: question.question,
    runs: requireRuns(runs.runs)
  };
}

export function requireRuns(runs: readonly RunPayload[]): NonEmptyRuns {
  if (runs.length === 0) {
    throw new Error("Question did not include any runs");
  }
  return runs as NonEmptyRuns;
}

export function upsertConversationSummary(
  conversations: readonly ConversationSummary[],
  next: ConversationSummary
): readonly ConversationSummary[] {
  const existing = conversations.filter(
    (conversation) => conversation.conversationId !== next.conversationId
  );
  return [next, ...existing];
}

export function mergeCanonicalConversations(
  canonical: readonly ConversationSummary[],
  optimistic: ConversationSummary
): readonly ConversationSummary[] {
  const canonicalConversation = canonical.find(
    (conversation) => conversation.conversationId === optimistic.conversationId
  );
  if (canonicalConversation?.latestQuestionId === optimistic.latestQuestionId) {
    return canonical;
  }
  return upsertConversationSummary(canonical, optimistic);
}

export function replaceConversationRun(
  conversation: ConversationDetails,
  nextRun: RunPayload
): ConversationDetails {
  const runs = conversation.runs.map((run) =>
    run.runId === nextRun.runId ? nextRun : run
  );
  return {
    ...conversation,
    summary: {
      ...conversation.summary,
      currentRunId: nextRun.runId,
      status: nextRun.status,
      updatedAt: new Date().toISOString()
    },
    runs: requireRuns(runs)
  };
}

export function updateConversationStatus(
  conversations: readonly ConversationSummary[],
  conversationId: string,
  status: RunStatus
): readonly ConversationSummary[] {
  return conversations.map((conversation) =>
    conversation.conversationId === conversationId
      ? { ...conversation, status, updatedAt: new Date().toISOString() }
      : conversation
  );
}

export function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : "Fervis request failed";
}

export function nextThemeMode(mode: ThemeMode): ThemeMode {
  if (mode === "system") {
    return "light";
  }
  if (mode === "light") {
    return "dark";
  }
  return "system";
}
