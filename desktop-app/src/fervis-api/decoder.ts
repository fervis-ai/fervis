import type {
  ConversationListPayload,
  ConversationSummary,
  DecodeResult,
  QuestionRunListPayload,
  QuestionStatePayload,
  RunPayload
} from "./contracts";
import { decodeNextActions } from "./decoding/actions";
import { decodeRunStatus, decodeTriggerKind } from "./decoding/enums";
import { decodeExplanation } from "./decoding/explanation";
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
      currentRunId: expectString(object.currentRunId, "currentRunId"),
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
    currentRunId: expectString(object.currentRunId, "currentRunId"),
    status: decodeRunStatus(object.status, "status"),
    runCount: expectNumber(object.runCount, "runCount"),
    updatedAt: expectString(object.updatedAt, "updatedAt")
  };
}

function decodeRunPayload(raw: unknown): RunPayload {
  const object = expectObject(raw, "run");
  return {
    runId: expectString(object.runId, "runId"),
    questionId: expectString(object.questionId, "questionId"),
    conversationId: expectString(object.conversationId, "conversationId"),
    runNumber: expectNumber(object.runNumber, "runNumber"),
    triggerKind: decodeTriggerKind(object.triggerKind, "triggerKind"),
    status: decodeRunStatus(object.status, "status"),
    answer: expectNullableString(object.answer, "answer"),
    resultData: decodeResultData(object.resultData),
    explanation: decodeExplanation(object.explanation),
    steps: expectArray(object.steps, "steps").map(decodeRunStep),
    error: object.error === null ? null : decodeRunError(object.error),
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
