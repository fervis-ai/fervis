import { useMemo, useState } from "react";

import { createFervisHttpClient, type FervisApiClient } from "../fervis-api/client";
import type { ConversationSummary } from "../fervis-api/contracts";
import { useLoadInitialConversation, usePollLatestRun } from "./appEffects";
import {
  type LoadState,
  loadConversation,
  mergeCanonicalConversations,
  messageFromError,
  nextThemeMode,
  requireRuns,
  upsertConversationSummary
} from "./appModel";
import { ConversationSurface } from "./components/ConversationSurface";
import { ConversationsRail } from "./components/ConversationsRail";
import { SettingsModal } from "./components/SettingsModal";
import { TopBar } from "./components/TopBar";
import {
  loadConnectionSettings,
  normalizeConnectionSettings,
  saveConnectionSettings
} from "./connectionSettings";
import type {
  ConversationDetails,
  QuestionRefreshPayload,
  ThemeMode
} from "./viewTypes";

export interface AppProps {
  readonly initialClient: FervisApiClient | null;
}

export function App({ initialClient }: AppProps) {
  const [apiClient, setApiClient] = useState<FervisApiClient | null>(
    initialClient
  );
  const [conversations, setConversations] = useState<readonly ConversationSummary[]>([]);
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(
    null
  );
  const [loadedConversation, setLoadedConversation] =
    useState<ConversationDetails | null>(null);
  const [loadState, setLoadState] = useState<LoadState>(
    initialClient === null ? "idle" : "loading"
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pollingErrorMessage, setPollingErrorMessage] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [connectionSettings, setConnectionSettings] = useState(() =>
    loadConnectionSettings()
  );
  const [themeMode, setThemeMode] = useState<ThemeMode>("system");

  const refreshQuestionState = async (
    question: QuestionRefreshPayload,
    submittedQuestion: string
  ): Promise<void> => {
    if (apiClient === null) {
      return;
    }
    setLoadState("loading");
    setErrorMessage(null);
    setPollingErrorMessage(null);
    try {
      const runs = await apiClient.listQuestionRuns(question.questionId);
      const nextSummary = {
        conversationId: question.conversationId,
        primaryRunId: question.primaryRunId,
        latestRunId: question.latestRunId,
        activeRunId: question.activeRunId,
        firstQuestion: submittedQuestion,
        latestQuestionId: question.questionId,
        runCount: runs.runs.length,
        status: question.status,
        updatedAt: new Date().toISOString()
      } satisfies ConversationSummary;
      setConversations((current) => upsertConversationSummary(current, nextSummary));
      setSelectedConversationId(question.conversationId);
      setLoadedConversation({
        summary: nextSummary,
        question: question.question,
        runs: requireRuns(runs.runs)
      });
      setLoadState("ready");
      void apiClient
        .listConversations()
        .then((payload) => {
          setConversations(
            mergeCanonicalConversations(payload.conversations, nextSummary)
          );
        })
        .catch(() => undefined);
    } catch (error) {
      setErrorMessage(messageFromError(error));
      setLoadState("failed");
    }
  };

  const handleActionError = (error: unknown): void => {
    setErrorMessage(messageFromError(error));
    setPollingErrorMessage(null);
    setLoadedConversation(null);
    setLoadState("failed");
  };

  useLoadInitialConversation({
    apiClient,
    setConversations,
    setErrorMessage,
    setLoadState,
    setLoadedConversation,
    setPollingErrorMessage,
    setSelectedConversationId
  });

  const latestLoadedRun =
    loadedConversation === null
      ? null
      : loadedConversation.runs[loadedConversation.runs.length - 1];

  usePollLatestRun({
    apiClient,
    latestLoadedRun,
    setConversations,
    setLoadedConversation,
    setPollingErrorMessage
  });

  const selectedSummary = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.conversationId === selectedConversationId
      ) ?? null,
    [conversations, selectedConversationId]
  );

  return (
    <div className={`app-root theme-${themeMode}`}>
      <TopBar
        themeMode={themeMode}
        onToggleTheme={() => setThemeMode(nextThemeMode(themeMode))}
      />
      <div className="ledger-shell">
        <ConversationsRail
          conversations={conversations}
          selectedConversationId={selectedConversationId}
          onNewQuestion={() => {
            setSelectedConversationId(null);
            setLoadedConversation(null);
            setLoadState("ready");
            setErrorMessage(null);
          }}
          onSelect={(conversationId) => {
            const summary = conversations.find(
              (conversation) => conversation.conversationId === conversationId
            );
            if (summary !== undefined && apiClient !== null) {
              setSelectedConversationId(conversationId);
              setLoadState("loading");
              setErrorMessage(null);
              void loadConversation(apiClient, summary)
                .then((loaded) => {
                  setLoadedConversation(loaded);
                  setLoadState("ready");
                })
                .catch((error: Error) => {
                  setErrorMessage(error.message);
                  setLoadState("failed");
                });
            }
          }}
        />
        <ConversationSurface
          apiClient={apiClient}
          errorMessage={errorMessage}
          loadState={loadState}
          pollingErrorMessage={pollingErrorMessage}
          selectedConversation={loadedConversation}
          selectedSummary={selectedSummary}
          onOpenSettings={() => setSettingsOpen(true)}
          onActionError={handleActionError}
          onQuestionState={refreshQuestionState}
        />
      </div>
      <button
        aria-label="Open connection settings"
        className="settings-button"
        type="button"
        onClick={() => setSettingsOpen(true)}
      >
        <span>Settings</span>
        <span className="settings-state">
          {apiClient === null ? "not set" : "connected"}
        </span>
      </button>
      {settingsOpen ? (
        <SettingsModal
          initialBaseUrl={connectionSettings.baseUrl}
          onClose={() => setSettingsOpen(false)}
          onSave={(connection) => {
            const nextSettings = normalizeConnectionSettings({
              baseUrl: connection.baseUrl
            });
            saveConnectionSettings(nextSettings);
            setConnectionSettings(nextSettings);
            setApiClient(
              createFervisHttpClient({
                authToken: connection.authToken,
                baseUrl: nextSettings.baseUrl
              })
            );
            setSettingsOpen(false);
          }}
        />
      ) : null}
    </div>
  );
}
