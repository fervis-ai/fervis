import { useState } from "react";

import type { FervisApiClient } from "../../fervis-api/client";
import type { ClarificationOption, ClarificationRequest, RunPayload } from "../../fervis-api/contracts";
import type { QuestionRefreshPayload } from "../viewTypes";
import { formatClarificationReason } from "../textFormat";
import { clarificationOptions, failMissingOption, firstClarification } from "../runView";

export function ClarificationForm({
  apiClient,
  run,
  onActionError,
  onClarificationState
}: {
  readonly apiClient: FervisApiClient | null;
  readonly run: RunPayload;
  readonly onActionError: (error: unknown) => void;
  readonly onClarificationState: (
    question: QuestionRefreshPayload
  ) => Promise<void>;
}) {
  const clarification = firstClarification(run);

  if (clarification === null) {
    return (
      <ContractErrorBlock
        code="invalid_clarification_contract"
        message="WAITING_FOR_CLARIFICATION did not include a clarification."
      />
    );
  }

  const options = clarificationOptions(clarification);

  if (options.length > 0) {
    return (
      <ChoiceClarification
        apiClient={apiClient}
        run={run}
        clarification={clarification}
        options={options}
        onActionError={onActionError}
        onClarificationState={onClarificationState}
      />
    );
  }

  return (
    <TextClarification
      apiClient={apiClient}
      run={run}
      clarification={clarification}
      onActionError={onActionError}
      onClarificationState={onClarificationState}
    />
  );
}

function ChoiceClarification({
  apiClient,
  run,
  clarification,
  options,
  onActionError,
  onClarificationState
}: {
  readonly apiClient: FervisApiClient | null;
  readonly run: RunPayload;
  readonly clarification: ClarificationRequest;
  readonly options: readonly ClarificationOption[];
  readonly onActionError: (error: unknown) => void;
  readonly onClarificationState: (
    question: QuestionRefreshPayload
  ) => Promise<void>;
}) {
  const [selectedOption, setSelectedOption] = useState<ClarificationOption>(
    options[0] ?? failMissingOption()
  );
  const [submitting, setSubmitting] = useState(false);

  if (submitting) {
    return (
      <ClarificationPending
        clarification={clarification}
        selectedAnswer={clarificationOptionAnswer(selectedOption)}
      />
    );
  }

  return (
    <div className="clarification">
      <ClarificationHeader clarification={clarification} />
      <div className="clar-options">
        {options.map((option) => (
          <label
            className={option.id === selectedOption.id ? "option selected" : "option"}
            key={option.id}
          >
            <input
              aria-label={clarificationOptionAnswer(option)}
              checked={option.id === selectedOption.id}
              name={`clarification-${clarification.id}`}
              type="radio"
              value={option.id}
              onChange={() => setSelectedOption(option)}
            />
            <ClarificationOptionSummary
              clarification={clarification}
              option={option}
            />
          </label>
        ))}
      </div>
      <button
        className="primary-action"
        disabled={apiClient === null || submitting}
        type="button"
        onClick={() => {
          if (apiClient === null) {
            return;
          }
          setSubmitting(true);
          void apiClient
            .answerClarification(run.questionId, {
              clarificationId: clarification.id,
              responseText: clarificationOptionAnswer(selectedOption),
              selectedOptionId: selectedOption.id,
              runId: run.runId
            })
            .then(onClarificationState)
            .catch(onActionError)
            .finally(() => setSubmitting(false));
        }}
      >
        {submitting ? "Sending…" : "Send clarification"}
      </button>
    </div>
  );
}

function TextClarification({
  apiClient,
  run,
  clarification,
  onActionError,
  onClarificationState
}: {
  readonly apiClient: FervisApiClient | null;
  readonly run: RunPayload;
  readonly clarification: ClarificationRequest;
  readonly onActionError: (error: unknown) => void;
  readonly onClarificationState: (
    question: QuestionRefreshPayload
  ) => Promise<void>;
}) {
  const [answer, setAnswer] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (submitting) {
    return (
      <ClarificationPending
        clarification={clarification}
        selectedAnswer={answer.trim()}
      />
    );
  }

  return (
    <div className="clarification">
      <ClarificationHeader clarification={clarification} />
      <form
        className="clar-text"
        onSubmit={(event) => {
          event.preventDefault();
          if (apiClient === null || answer.trim() === "") {
            return;
          }
          setSubmitting(true);
          void apiClient
            .answerClarification(run.questionId, {
              clarificationId: clarification.id,
              responseText: answer.trim(),
              runId: run.runId
            })
            .then(onClarificationState)
            .catch(onActionError)
            .finally(() => setSubmitting(false));
        }}
      >
        <input
          aria-label="Clarification answer"
          placeholder="Type your answer…"
          value={answer}
          onChange={(event) => setAnswer(event.currentTarget.value)}
        />
        <button
          disabled={apiClient === null || submitting || answer.trim() === ""}
          type="submit"
        >
          {submitting ? "Sending…" : "Send"}
        </button>
      </form>
    </div>
  );
}

function ClarificationPending({
  clarification,
  selectedAnswer
}: {
  readonly clarification: ClarificationRequest;
  readonly selectedAnswer: string;
}) {
  return (
    <div className="clarification clarification-pending" role="status">
      <ClarificationHeader clarification={clarification} />
      <div className="pending-answer">
        <span>answer sent</span>
        <strong>{selectedAnswer}</strong>
      </div>
      <p className="quiet">Continuing this question run.</p>
    </div>
  );
}

function ClarificationHeader({
  clarification
}: {
  readonly clarification: ClarificationRequest;
}) {
  const options = clarificationOptions(clarification);
  const metadata = [
    formatClarificationReason(clarification.reason),
    options.length > 0
      ? `${options.length} choices offered`
      : "text answer requested"
  ].filter((value): value is string => value !== null);

  return (
    <>
      <div className="clar-meta">{metadata.join(" · ")}</div>
      <div className="clar-question">{clarification.question}</div>
    </>
  );
}

function ClarificationOptionSummary({
  clarification,
  option
}: {
  readonly clarification: ClarificationRequest;
  readonly option: ClarificationOption;
}) {
  const subject = clarification.subjects[0] ?? null;
  const reference = subject?.sourceText || option.matchedLabel || option.label;
  const entity = titleWords(option.entityKind ?? "");
  const field = option.matchedField ?? "";
  const value = option.matchedValue ?? option.value ?? "";
  const resolver = option.resolverLabel || titleWords(option.resolverReadId ?? "");

  if (field === "" || value === "") {
    return <span>{option.label}</span>;
  }

  return (
    <span className="option-summary">
      <span>Reference: {reference}</span>
      {entity !== "" ? <span>Matched entity: {entity}</span> : null}
      <span>
        {field}: {value}
        {resolver !== "" ? ` (via ${resolver})` : ""}
      </span>
    </span>
  );
}

function clarificationOptionAnswer(option: ClarificationOption): string {
  return option.matchedLabel || option.label;
}

function titleWords(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .split(" ")
    .filter((word) => word !== "")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function ContractErrorBlock({
  code,
  message
}: {
  readonly code: string;
  readonly message: string;
}) {
  return (
    <div className="failure">
      <div className="error-kind">runtime_error · {code}</div>
      <p>{message}</p>
    </div>
  );
}
