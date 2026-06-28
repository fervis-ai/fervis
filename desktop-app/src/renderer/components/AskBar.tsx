import { useState } from "react";

import type { FervisApiClient } from "../../fervis-api/client";
import type { RunStatus } from "../../fervis-api/contracts";
import { askHint, askPlaceholder } from "../runView";
import type { QuestionRefreshPayload } from "../viewTypes";

export function AskBar({
  apiClient,
  conversationId,
  status,
  onActionError,
  onQuestionState
}: {
  readonly apiClient: FervisApiClient | null;
  readonly conversationId: string | null;
  readonly status: RunStatus;
  readonly onActionError: (error: unknown) => void;
  readonly onQuestionState: (
    question: QuestionRefreshPayload,
    submittedQuestion: string
  ) => Promise<void>;
}) {
  const [question, setQuestion] = useState("");
  const [submitting, setSubmitting] = useState(false);

  return (
    <section className="askbar">
      <div className="ask-label">
        {conversationId === null ? "New question" : "Follow up"}
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const submittedQuestion = question.trim();
          if (apiClient === null || submittedQuestion === "") {
            return;
          }
          setSubmitting(true);
          void apiClient
            .askQuestion({ conversationId, question: submittedQuestion })
            .then((state) => onQuestionState(state, submittedQuestion))
            .then(() => setQuestion(""))
            .catch(onActionError)
            .finally(() => setSubmitting(false));
        }}
      >
        <input
          aria-label={conversationId === null ? "Ask a question" : "Ask a follow-up question"}
          placeholder={
            conversationId === null ? "Ask a factual question…" : askPlaceholder(status)
          }
          value={question}
          onChange={(event) => setQuestion(event.currentTarget.value)}
        />
        <button
          disabled={apiClient === null || submitting || question.trim() === ""}
          type="submit"
        >
          {submitting ? "Asking…" : "Ask"}
        </button>
      </form>
      <div className="hint">
        {conversationId === null
          ? "Enter sends · Fervis creates a conversation and queues the first run."
          : askHint(status)}
      </div>
    </section>
  );
}
