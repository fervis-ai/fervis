import { useEffect } from "react";

import type { FervisApiClient } from "../fervis-api/client";
import type { ConversationSummary } from "../fervis-api/contracts";
import {
  type LoadState,
  loadConversation,
  messageFromError,
  replaceConversationRun,
  updateConversationStatus
} from "./appModel";
import { pollableRun } from "./runView";
import type { ConversationDetails } from "./viewTypes";

const POLL_INTERVAL_MS = 1200;

export function useLoadInitialConversation({
  apiClient,
  setConversations,
  setErrorMessage,
  setLoadState,
  setLoadedConversation,
  setPollingErrorMessage,
  setSelectedConversationId
}: {
  readonly apiClient: FervisApiClient | null;
  readonly setConversations: (value: readonly ConversationSummary[]) => void;
  readonly setErrorMessage: (value: string | null) => void;
  readonly setLoadState: (value: LoadState) => void;
  readonly setLoadedConversation: (value: ConversationDetails | null) => void;
  readonly setPollingErrorMessage: (value: string | null) => void;
  readonly setSelectedConversationId: (value: string | null) => void;
}) {
  useEffect(() => {
    if (apiClient === null) {
      return;
    }

    let active = true;
    setLoadState("loading");
    setErrorMessage(null);
    setPollingErrorMessage(null);
    apiClient
      .listConversations()
      .then(async (payload) => {
        if (!active) {
          return;
        }
        setConversations(payload.conversations);
        const firstConversation = payload.conversations[0] ?? null;
        if (firstConversation === null) {
          setLoadedConversation(null);
          setSelectedConversationId(null);
          setLoadState("ready");
          return;
        }
        setSelectedConversationId(firstConversation.conversationId);
        const loaded = await loadConversation(apiClient, firstConversation);
        if (!active) {
          return;
        }
        setLoadedConversation(loaded);
        setLoadState("ready");
      })
      .catch((error: Error) => {
        if (!active) {
          return;
        }
        setErrorMessage(error.message);
        setLoadState("failed");
      });

    return () => {
      active = false;
    };
  }, [
    apiClient,
    setConversations,
    setErrorMessage,
    setLoadState,
    setLoadedConversation,
    setPollingErrorMessage,
    setSelectedConversationId
  ]);
}

export function usePollLatestRun({
  apiClient,
  latestLoadedRun,
  setConversations,
  setLoadedConversation,
  setPollingErrorMessage
}: {
  readonly apiClient: FervisApiClient | null;
  readonly latestLoadedRun: ConversationDetails["runs"][number] | null;
  readonly setConversations: (
    value: (current: readonly ConversationSummary[]) => readonly ConversationSummary[]
  ) => void;
  readonly setLoadedConversation: (
    value: (current: ConversationDetails | null) => ConversationDetails | null
  ) => void;
  readonly setPollingErrorMessage: (value: string | null) => void;
}) {
  const pollingQuestionId =
    latestLoadedRun === null ? null : latestLoadedRun.questionId;
  const pollingRunId = latestLoadedRun === null ? null : latestLoadedRun.runId;
  const pollingStatus = latestLoadedRun === null ? null : latestLoadedRun.status;

  useEffect(() => {
    if (apiClient === null || latestLoadedRun === null) {
      return;
    }
    if (!pollableRun(latestLoadedRun)) {
      return;
    }

    let active = true;
    const poll = async (): Promise<void> => {
      try {
        const nextRun = await apiClient.getRun(
          latestLoadedRun.questionId,
          latestLoadedRun.runId
        );
        if (!active) {
          return;
        }
        if (nextRun.runId !== latestLoadedRun.runId) {
          throw new Error("Polled run identity changed unexpectedly");
        }
        setPollingErrorMessage(null);
        if (completedRunAwaitingProjection(nextRun)) {
          return;
        }
        setLoadedConversation((current) =>
          current === null ? null : replaceConversationRun(current, nextRun)
        );
        setConversations((current) =>
          updateConversationStatus(current, nextRun.conversationId, nextRun.status)
        );
      } catch (error) {
        if (!active) {
          return;
        }
        setPollingErrorMessage(messageFromError(error));
      }
    };

    void poll();
    const intervalId = window.setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [
    apiClient,
    latestLoadedRun,
    pollingQuestionId,
    pollingRunId,
    pollingStatus,
    setConversations,
    setLoadedConversation,
    setPollingErrorMessage
  ]);
}

function completedRunAwaitingProjection(
  run: ConversationDetails["runs"][number]
): boolean {
  return run.status === "COMPLETED" && pollableRun(run);
}
