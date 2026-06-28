import type { ConversationSummary, RunStatus } from "../../fervis-api/contracts";
import { titleFromQuestion } from "../conversationTitle";
import { statusClassName } from "../runView";

export function ConversationsRail({
  conversations,
  selectedConversationId,
  onNewQuestion,
  onSelect
}: {
  readonly conversations: readonly ConversationSummary[];
  readonly selectedConversationId: string | null;
  readonly onNewQuestion: () => void;
  readonly onSelect: (conversationId: string) => void;
}) {
  return (
    <aside className="rail" aria-label="Conversations">
      <div className="rail-head">
        <h2>Conversations</h2>
        <button type="button" onClick={onNewQuestion}>
          New question
        </button>
      </div>
      <div className="conversation-list">
        {conversations.map((conversation) => (
          <button
            className={
              conversation.conversationId === selectedConversationId
                ? "conversation-card selected"
                : "conversation-card"
            }
            key={conversation.conversationId}
            type="button"
            onClick={() => onSelect(conversation.conversationId)}
          >
            <span className="conversation-label">
              {titleFromQuestion(conversation.firstQuestion)}
            </span>
            <span className="conversation-summary">
              <StatusDot status={conversation.status} />
              {conversation.runCount} run
              {conversation.runCount === 1 ? "" : "s"} · {conversation.status}
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}

function StatusDot({ status }: { readonly status: RunStatus }) {
  return <span className={`status-dot ${statusClassName(status)}`} />;
}
