import type { FervisApiClient } from "../../fervis-api/client";
import type { NextAction, RunPayload } from "../../fervis-api/contracts";
import { completedAnswerText, runSummary, statusClassName } from "../runView";
import type { QuestionRefreshPayload } from "../viewTypes";
import { ClarificationForm } from "./Clarification";
import { EvidencePanel } from "./EvidencePanel";
import { LiveZone } from "./LiveZone";

export function RunBlock({
  run,
  open,
  onToggle,
  apiClient,
  onActionError,
  onClarificationState
}: {
  readonly run: RunPayload;
  readonly open: boolean;
  readonly onToggle: () => void;
  readonly apiClient: FervisApiClient | null;
  readonly onActionError: (error: unknown) => void;
  readonly onClarificationState: (
    question: QuestionRefreshPayload
  ) => Promise<void>;
}) {
  const statusClass = statusClassName(run.status);

  return (
    <section className={open ? "run-block expanded" : "run-block"}>
      <button className="run-header" type="button" onClick={onToggle}>
        <span className="marker" aria-hidden="true" />
        <span className="run-number">run {run.runNumber}</span>
        <span className="run-id">{run.runId}</span>
        <span className={`run-status ${statusClass}`}>{run.status}</span>
        <span className="run-summary">{runSummary(run)}</span>
      </button>
      {open ? (
        <div className="run-body">
          <AnswerZone
            apiClient={apiClient}
            onActionError={onActionError}
            run={run}
            onClarificationState={onClarificationState}
          />
          <EvidencePanel defaultOpen={run.status === "COMPLETED"} run={run} />
        </div>
      ) : null}
    </section>
  );
}

function AnswerZone({
  apiClient,
  onActionError,
  run,
  onClarificationState
}: {
  readonly apiClient: FervisApiClient | null;
  readonly onActionError: (error: unknown) => void;
  readonly run: RunPayload;
  readonly onClarificationState: (
    question: QuestionRefreshPayload
  ) => Promise<void>;
}) {
  return (
    <section className="answer-zone">
      <div className="section-label">
        <span>§2</span> Answer
      </div>
      {run.status === "COMPLETED" ? <CompletedAnswer run={run} /> : null}
      {run.status === "RUNNING" || run.status === "QUEUED" ? (
        <LiveZone run={run} />
      ) : null}
      {run.status === "NEEDS_CLARIFICATION" ? (
        <ClarificationForm
          apiClient={apiClient}
          onActionError={onActionError}
          run={run}
          onClarificationState={onClarificationState}
        />
      ) : null}
      {run.status === "FAILED" ? <FailureBlock run={run} /> : null}
    </section>
  );
}

function CompletedAnswer({ run }: { readonly run: RunPayload }) {
  const answer = completedAnswerText(run);
  if (answer !== null) {
    return <p className="answer-prose">{answer}</p>;
  }
  return <p className="quiet">no answer produced</p>;
}

function FailureBlock({ run }: { readonly run: RunPayload }) {
  const error = run.error;
  return (
    <div className="failure">
      <div className="error-kind">
        runtime_error · {error?.code ?? "unknown_error"}
      </div>
      <p>{error?.message ?? "The run failed before producing an answer."}</p>
      {run.nextActions.length > 0 ? (
        <div className="hint">{nextActionText(run.nextActions[0])}</div>
      ) : null}
    </div>
  );
}

function nextActionText(action: NextAction): string {
  if (action.command !== null) {
    return action.command;
  }
  if (action.request !== null) {
    return `${action.request.method} ${action.request.path}`;
  }
  return action.description ?? action.kind;
}
