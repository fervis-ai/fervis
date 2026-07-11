import type {
  AskQuestionRequest,
  ClarificationResponseRequest,
  ConversationListPayload,
  DecodeResult,
  QuestionRunListPayload,
  QuestionStatePayload,
  RerunQuestionRequest,
  RunPayload
} from "./contracts";
import {
  decodeConversationList,
  decodeQuestionRunList,
  decodeRun,
  decodeQuestionState
} from "./decoder";

export interface FervisConnection {
  readonly baseUrl: string;
  readonly authToken: string;
}

export interface FervisApiClient {
  readonly listConversations: () => Promise<ConversationListPayload>;
  readonly getQuestion: (questionId: string) => Promise<QuestionStatePayload>;
  readonly listQuestionRuns: (questionId: string) => Promise<QuestionRunListPayload>;
  readonly getRun: (questionId: string, runId: string) => Promise<RunPayload>;
  readonly askQuestion: (
    request: AskQuestionRequest
  ) => Promise<QuestionStatePayload>;
  readonly answerClarification: (
    questionId: string,
    request: ClarificationResponseRequest
  ) => Promise<QuestionStatePayload>;
  readonly rerunQuestion: (
    questionId: string,
    request: RerunQuestionRequest
  ) => Promise<QuestionStatePayload>;
}

export class FervisApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "FervisApiError";
    this.status = status;
  }
}

export function createFervisHttpClient(
  connection: FervisConnection
): FervisApiClient {
  return {
    listConversations: async () =>
      decodePayload(
        "conversation list",
        await requestJson(connection, "/conversations/"),
        decodeConversationList
      ),
    getQuestion: async (questionId) =>
      decodePayload(
        "question state",
        await requestJson(connection, `/questions/${encodeURIComponent(questionId)}/`),
        decodeQuestionState
      ),
    listQuestionRuns: async (questionId) =>
      decodePayload(
        "question run list",
        await requestJson(
          connection,
          `/questions/${encodeURIComponent(questionId)}/runs/`
        ),
        decodeQuestionRunList
      ),
    getRun: async (questionId, runId) =>
      decodePayload(
        "run",
        await requestJson(
          connection,
          `/questions/${encodeURIComponent(questionId)}/runs/${encodeURIComponent(runId)}/`
        ),
        decodeRun
      ),
    askQuestion: async (request) =>
      decodePayload(
        "question state",
        await requestJson(connection, "/questions/", {
          body: JSON.stringify({
            question: request.question,
            conversationId: request.conversationId,
            ...(request.contextRunId === undefined
              ? {}
              : { contextRunId: request.contextRunId })
          }),
          method: "POST"
        }),
        decodeQuestionState
      ),
    answerClarification: async (questionId, request) =>
      decodePayload(
        "question state",
        await requestJson(
          connection,
          `/questions/${encodeURIComponent(questionId)}/runs/`,
          {
            body: JSON.stringify(request),
            method: "POST"
          }
        ),
        decodeQuestionState
      ),
    rerunQuestion: async (questionId, request) =>
      decodePayload(
        "question state",
        await requestJson(
          connection,
          `/questions/${encodeURIComponent(questionId)}/runs/`,
          {
            body: JSON.stringify(request),
            method: "POST"
          }
        ),
        decodeQuestionState
      )
  };
}

async function requestJson(
  connection: FervisConnection,
  path: string,
  init: RequestInit = {}
): Promise<unknown> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body !== undefined) {
    headers.set("Content-Type", "application/json");
    headers.set("Idempotency-Key", crypto.randomUUID());
  }
  if (connection.authToken.trim() !== "") {
    headers.set("Authorization", `Bearer ${connection.authToken}`);
  }

  const response = await fetch(`${trimTrailingSlash(connection.baseUrl)}${path}`, {
    ...init,
    headers
  });

  const body = await responseJson(response);
  if (!response.ok) {
    throw new FervisApiError(response.status, errorMessage(body));
  }
  return body;
}

function decodePayload<T>(
  label: string,
  payload: unknown,
  decoder: (payload: unknown) => DecodeResult<T>
): T {
  const decoded = decoder(payload);
  if (!decoded.ok) {
    throw new FervisApiError(0, `${label} decode failed: ${decoded.error.message}`);
  }
  return decoded.value;
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

async function responseJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (text.trim() === "") {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function errorMessage(body: unknown): string {
  if (typeof body !== "object" || body === null || !("error" in body)) {
    return "Fervis request failed";
  }
  const error = body.error;
  if (typeof error !== "object" || error === null || !("message" in error)) {
    return "Fervis request failed";
  }
  return typeof error.message === "string" ? error.message : "Fervis request failed";
}
