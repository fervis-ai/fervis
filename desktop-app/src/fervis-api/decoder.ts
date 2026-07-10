import type {
  ConversationListPayload,
  ConversationSummary,
  DecodeResult,
  QuestionRunListPayload,
  RunError,
  RunIdentity,
  QuestionStatePayload,
  RunStep,
  RunPayload
} from "./contracts";
import { decodeNextActions } from "./decoding/actions";
import { decodeRunKind, decodeRunStatus, decodeTriggerKind } from "./decoding/enums";
import { decodeExplanation, emptyExplanation } from "./decoding/explanation";
import { decode, expectArray, expectNumber, expectNullableString, expectObject, expectString } from "./decoding/primitives";
import { decodeResultData } from "./decoding/resultData";
import {
  decodeRunError,
  decodeRunStep,
  decodeUsage,
  decodeWorker
} from "./decoding/runDetails";

export function decodeConversationList(raw: unknown): DecodeResult<ConversationListPayload> {
  return decode("conversation list", () => {
    const object = expectObject(raw, "payload");
    return {
      conversations: expectArray(object.conversations, "conversations").map(
        decodeConversationSummary
      )
    };
  });
}

export function decodeQuestionState(raw: unknown): DecodeResult<QuestionStatePayload> {
  return decode("question state", () => {
    const object = expectObject(raw, "payload");
    return {
      questionId: expectString(object.questionId, "questionId"),
      conversationId: expectString(object.conversationId, "conversationId"),
      question: expectString(object.question, "question"),
      primaryRunId: expectNullableString(object.primaryRunId, "primaryRunId"),
      latestRunId: expectNullableString(object.latestRunId, "latestRunId"),
      activeRunId: expectNullableString(object.activeRunId, "activeRunId"),
      status: decodeRunStatus(object.status, "status"),
      answer: expectNullableString(object.answer, "answer"),
      resultData: decodeResultData(object.resultData),
      nextActions: decodeNextActions(object.nextActions)
    };
  });
}

export function decodeQuestionRunList(raw: unknown): DecodeResult<QuestionRunListPayload> {
  return decode("question run list", () => {
    const object = expectObject(raw, "payload");
    return {
      questionId: expectString(object.questionId, "questionId"),
      runs: expectArray(object.runs, "runs").map(decodeRunPayload)
    };
  });
}

export function decodeRun(raw: unknown): DecodeResult<RunPayload> {
  return decode("run", () => decodeRunPayload(raw));
}

function decodeConversationSummary(raw: unknown): ConversationSummary {
  const object = expectObject(raw, "conversation");
  return {
    conversationId: expectString(object.conversationId, "conversationId"),
    firstQuestion: expectString(object.firstQuestion, "firstQuestion"),
    latestQuestionId: expectString(object.latestQuestionId, "latestQuestionId"),
    primaryRunId: expectNullableString(object.primaryRunId, "primaryRunId"),
    latestRunId: expectNullableString(object.latestRunId, "latestRunId"),
    activeRunId: expectNullableString(object.activeRunId, "activeRunId"),
    status: decodeRunStatus(object.status, "status"),
    runCount: expectNumber(object.runCount, "runCount"),
    updatedAt: expectString(object.updatedAt, "updatedAt")
  };
}

function decodeRunPayload(raw: unknown): RunPayload {
  const object = expectObject(raw, "run");
  return {
    ...decodeRunIdentity(object),
    runId: expectString(object.runId, "runId"),
    questionId: expectString(object.questionId, "questionId"),
    conversationId: expectString(object.conversationId, "conversationId"),
    runNumber: expectNumber(object.runNumber, "runNumber"),
    patchId: expectNullableString(object.patchId, "patchId"),
    revisionId: expectNullableString(object.revisionId, "revisionId"),
    status: decodeRunStatus(object.status, "status"),
    answer: expectNullableString(object.answer, "answer"),
    resultData: decodeResultData(object.resultData),
    explanation:
      object.explanation === undefined || object.explanation === null
        ? emptyExplanation()
        : decodeExplanation(object.explanation),
    steps: decodeRunSteps(object.steps),
    error: decodeRunErrorPayload(object.error),
    worker:
      object.worker === undefined || object.worker === null
        ? null
        : decodeWorker(object.worker),
    usage:
      object.usage === undefined || object.usage === null
        ? null
        : decodeUsage(object.usage),
    nextActions: decodeNextActions(object.nextActions)
  };
}

function decodeRunIdentity(
  object: ReturnType<typeof expectObject>
): RunIdentity {
  const kind = decodeRunKind(object.kind, "kind");
  const triggerKind = decodeTriggerKind(object.triggerKind, "triggerKind");
  if (kind === "deterministic") {
    if (triggerKind !== "rerun") {
      throw new Error("deterministic run requires triggerKind rerun");
    }
    return {
      kind,
      triggerKind,
      baseRunId: expectString(object.baseRunId, "baseRunId"),
      programId: expectString(object.programId, "programId"),
      invocationId: expectString(object.invocationId, "invocationId")
    };
  }
  if (triggerKind === "rerun") {
    throw new Error("model-assisted run cannot use triggerKind rerun");
  }
  if (triggerKind === "initial") {
    if (object.baseRunId !== null) {
      throw new Error("initial run requires a null baseRunId");
    }
    return {
      kind,
      triggerKind,
      baseRunId: null,
      programId: expectNullableString(object.programId, "programId"),
      invocationId: expectNullableString(object.invocationId, "invocationId")
    };
  }
  return {
    kind,
    triggerKind,
    baseRunId: expectString(object.baseRunId, "baseRunId"),
    programId: expectNullableString(object.programId, "programId"),
    invocationId: expectNullableString(object.invocationId, "invocationId")
  };
}

function decodeRunErrorPayload(raw: unknown): RunError | null {
  if (raw === undefined || raw === null || raw === "") {
    return null;
  }
  if (typeof raw === "string") {
    return {
      code: raw,
      message: raw,
      retryable: false
    };
  }
  return decodeRunError(raw);
}

function decodeRunSteps(raw: unknown): readonly RunStep[] {
  if (raw === undefined || raw === null) {
    return [];
  }
  return expectArray(raw, "steps").map(decodeRunStep);
}
