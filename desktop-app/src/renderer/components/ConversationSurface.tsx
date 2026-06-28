import type { FervisApiClient } from "../../fervis-api/client";
import type { ConversationSummary } from "../../fervis-api/contracts";
import type { LoadState } from "../appModel";
import type { ConversationDetails, QuestionRefreshPayload } from "../viewTypes";
import { AskBar } from "./AskBar";
import { ConversationLedger } from "./ConversationLedger";

export function ConversationSurface({
  apiClient,
  loadState,
  errorMessage,
  selectedConversation,
  pollingErrorMessage,
  selectedSummary,
  onOpenSettings,
  onActionError,
  onQuestionState
}: {
  readonly apiClient: FervisApiClient | null;
  readonly loadState: LoadState;
  readonly errorMessage: string | null;
  readonly pollingErrorMessage: string | null;
  readonly selectedConversation: ConversationDetails | null;
  readonly selectedSummary: ConversationSummary | null;
  readonly onOpenSettings: () => void;
  readonly onActionError: (error: unknown) => void;
  readonly onQuestionState: (
    question: QuestionRefreshPayload,
    submittedQuestion: string
  ) => Promise<void>;
}) {
  if (selectedConversation !== null) {
    return (
      <ConversationLedger
        apiClient={apiClient}
        conversation={selectedConversation}
        pollingErrorMessage={pollingErrorMessage}
        onActionError={onActionError}
        onQuestionState={onQuestionState}
      />
    );
  }
  return (
    <main className="conversation-main">
      <section className="empty-state">
        <p className="mono-muted">{emptyEyebrow(loadState, selectedSummary)}</p>
        <h1>{emptyTitle(loadState)}</h1>
        <p>{emptyMessage(loadState, errorMessage)}</p>
        {apiClient === null || loadState === "failed" ? (
          <button type="button" onClick={onOpenSettings}>
            Open settings
          </button>
        ) : null}
        {apiClient !== null && loadState === "ready" ? (
          <AskBar
            apiClient={apiClient}
            conversationId={null}
            status="COMPLETED"
            onActionError={onActionError}
            onQuestionState={(question, submittedQuestion) =>
              onQuestionState(question, submittedQuestion)
            }
          />
        ) : null}
      </section>
    </main>
  );
}

function emptyEyebrow(
  state: LoadState,
  selectedSummary: ConversationSummary | null
): string {
  if (selectedSummary !== null) {
    return `conversation · ${selectedSummary.conversationId}`;
  }
  if (state === "loading") {
    return "loading";
  }
  return "connection";
}

function emptyTitle(state: LoadState): string {
  if (state === "loading") {
    return "Loading conversations";
  }
  if (state === "failed") {
    return "Fervis is not reachable";
  }
  return "Connect to Fervis";
}

function emptyMessage(state: LoadState, errorMessage: string | null): string {
  if (state === "failed") {
    return errorMessage ?? "The configured Fervis endpoint did not respond.";
  }
  if (state === "ready") {
    return "No conversations have been created for this subject yet.";
  }
  return "Set the Base API URL and optional auth token to load conversations.";
}
