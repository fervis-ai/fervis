import { useEffect, useState } from "react";

import type { FervisApiClient } from "../../fervis-api/client";
import { titleFromQuestion } from "../conversationTitle";
import type { ConversationDetails, QuestionRefreshPayload } from "../viewTypes";
import { AskBar } from "./AskBar";
import { QuestionBlock } from "./QuestionBlock";
import { RunBlock } from "./RunBlock";
import { RunFootnote } from "./RunFootnote";

export function ConversationLedger({
  apiClient,
  conversation,
  pollingErrorMessage,
  onActionError,
  onQuestionState
}: {
  readonly apiClient: FervisApiClient | null;
  readonly conversation: ConversationDetails;
  readonly pollingErrorMessage: string | null;
  readonly onActionError: (error: unknown) => void;
  readonly onQuestionState: (
    question: QuestionRefreshPayload,
    submittedQuestion: string
  ) => Promise<void>;
}) {
  const latestRun = conversation.runs[conversation.runs.length - 1];
  const defaultRunId =
    conversation.runs.find(
      (run) => run.runId === conversation.summary.primaryRunId
    )?.runId ?? latestRun.runId;
  const [openRunId, setOpenRunId] = useState<string | null>(defaultRunId);
  const selectedContextRun = conversation.runs.find(
    (run) => run.runId === openRunId && run.answer !== null
  );
  const contextRunId =
    selectedContextRun?.runId === conversation.summary.primaryRunId
      ? undefined
      : selectedContextRun?.runId;

  useEffect(() => {
    setOpenRunId(defaultRunId);
  }, [defaultRunId]);

  return (
    <main className="conversation-main ledger-main">
      <div className="conversation-scroll">
        <header className="conversation-header">
          <div className="mono-muted">
            conversation · {conversation.summary.conversationId}
          </div>
          <h1>{titleFromQuestion(conversation.summary.firstQuestion, 12)}</h1>
        </header>
        <section className="turn-wrapper" aria-label="Conversation turn">
          <article className="turn">
            <div className="run-count">
              {conversation.runs.length} run
              {conversation.runs.length === 1 ? "" : "s"} in this question
            </div>
            <QuestionBlock question={conversation.question} />
            <div className="runs">
              {conversation.runs.map((run) => (
                <RunBlock
                  open={run.runId === openRunId}
                  key={run.runId}
                  run={run}
                  onToggle={() =>
                    setOpenRunId((current) =>
                      current === run.runId ? null : run.runId
                    )
                  }
                  onClarificationState={async (question) => {
                    await onQuestionState(question, conversation.summary.firstQuestion);
                    setOpenRunId(question.primaryRunId ?? question.latestRunId);
                  }}
                  onActionError={onActionError}
                  apiClient={apiClient}
                />
              ))}
            </div>
          </article>
          <RunFootnote pollingErrorMessage={pollingErrorMessage} run={latestRun} />
        </section>
      </div>
      <AskBar
        apiClient={apiClient}
        conversationId={conversation.summary.conversationId}
        contextRunId={contextRunId}
        status={conversation.summary.status}
        onActionError={onActionError}
        onQuestionState={(question, submittedQuestion) =>
          onQuestionState(question, submittedQuestion)
        }
      />
    </main>
  );
}
