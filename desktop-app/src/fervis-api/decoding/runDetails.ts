import type {
  RunError,
  RunStep,
  StepSemantic,
  SourceRead,
  StepDecision,
  UsageSnapshot,
  WorkerSnapshot
} from "../contracts";
import {
  expectArray,
  expectBoolean,
  expectNullableString,
  expectNumber,
  expectObject,
  expectString,
  expectStringArray
} from "./primitives";

export function decodeRunStep(raw: unknown): RunStep {
  const object = expectObject(raw, "run step");
  return {
    stepId: expectString(object.stepId, "stepId"),
    stepKey: expectString(object.stepKey, "stepKey"),
    decisions: decodeRunStepDecisions(object),
    semantic: decodeStepSemantic(object.semantic)
  };
}

export function decodeStepSemantic(raw: unknown): StepSemantic {
  const object = expectObject(raw, "step.semantic");
  return {
    requestedFacts: expectArray(
      object.requestedFacts,
      "step.semantic.requestedFacts"
    ).map((item) => {
      const fact = expectObject(item, "semantic requested fact");
      return {
        requestedFactId: expectString(fact.requestedFactId, "requestedFactId"),
        description: expectString(fact.description, "description")
      };
    }),
    knownInputs: expectArray(object.knownInputs, "step.semantic.knownInputs").map(
      (item) => {
        const input = expectObject(item, "semantic known input");
        return {
          inputId: expectString(input.inputId, "inputId"),
          text: expectString(input.text, "text"),
          kind: expectString(input.kind, "kind"),
          description: expectString(input.description, "description"),
          lookupText: semanticInputLookupText(input)
        };
      }
    ),
    resolverCandidates: expectArray(
      object.resolverCandidates,
      "step.semantic.resolverCandidates"
    ).map((item) => {
      const candidate = expectObject(item, "semantic resolver candidate");
      return {
        inputId: expectString(candidate.inputId, "inputId"),
        resolverReadId: expectString(candidate.resolverReadId, "resolverReadId"),
        resolverLabel: expectString(candidate.resolverLabel, "resolverLabel"),
        basis: expectString(candidate.basis, "basis")
      };
    }),
    groundingResults: expectArray(
      object.groundingResults,
      "step.semantic.groundingResults"
    ).map((item) => {
      const result = expectObject(item, "semantic grounding result");
      return {
        inputId: expectString(result.inputId, "inputId"),
        inputText: expectString(result.inputText, "inputText"),
        resolverReadId: expectString(result.resolverReadId, "resolverReadId"),
        resolverLabel: expectString(result.resolverLabel, "resolverLabel"),
        matchedField: expectString(result.matchedField, "matchedField"),
        matchedValue: expectString(result.matchedValue, "matchedValue"),
        matchedLabel: expectString(result.matchedLabel, "matchedLabel")
      };
    }),
    interpretedInputs: expectArray(
      object.interpretedInputs,
      "step.semantic.interpretedInputs"
    ).map((item) => {
      const input = expectObject(item, "semantic interpreted input");
      return {
        inputId: expectString(input.inputId, "inputId"),
        inputText: expectString(input.inputText, "inputText"),
        kind: expectString(input.kind, "kind"),
        value: expectString(input.value, "value"),
        label: expectString(input.label, "label"),
        detail: expectString(input.detail, "detail")
      };
    }),
    conversationClauses: expectArray(
      object.conversationClauses,
      "step.semantic.conversationClauses"
    ).map((item) => {
      const clause = expectObject(item, "semantic conversation clause");
      return {
        currentClauseText: expectString(
          clause.currentClauseText,
          "currentClauseText"
        ),
        currentValueText: expectString(clause.currentValueText, "currentValueText"),
        resolvedFrameText: expectString(
          clause.resolvedFrameText,
          "resolvedFrameText"
        ),
        resolvedClauseText: expectString(
          clause.resolvedClauseText,
          "resolvedClauseText"
        )
      };
    })
  };
}

export function decodeStepDecision(raw: unknown): StepDecision {
  const object = expectObject(raw, "step decision");
  return {
    lines: expectStringArray(object.lines, "decision.lines")
  };
}

export function decodeSourceRead(raw: unknown): SourceRead {
  const object = expectObject(raw, "source read");
  const endpoint =
    object.catalogEndpoint === undefined
      ? null
      : expectObject(object.catalogEndpoint, "sourceRead.catalogEndpoint");
  return {
    sourceReadId: expectString(object.sourceReadId, "sourceReadId"),
    method:
      endpoint === null
        ? expectString(object.method, "method")
        : expectString(endpoint.routeMethod, "catalogEndpoint.routeMethod"),
    path:
      endpoint === null
        ? expectString(object.path, "path")
        : expectString(endpoint.routePathTemplate, "catalogEndpoint.routePathTemplate"),
    rowCount: expectNumber(object.rowCount, "rowCount"),
    status: expectString(object.status, "status")
  };
}

export function decodeRunError(raw: unknown): RunError {
  const object = expectObject(raw, "run error");
  if (typeof object.errorKind === "string") {
    return {
      code: object.errorKind,
      message: expectString(object.message, "error.message"),
      retryable: false
    };
  }
  return {
    code: expectString(object.code, "error.code"),
    message: expectString(object.message, "error.message"),
    retryable: expectBoolean(object.retryable, "error.retryable")
  };
}

export function decodeWorker(raw: unknown): WorkerSnapshot {
  const object = expectObject(raw, "worker");
  return {
    status: expectString(object.status, "worker.status"),
    attemptCount: expectNumber(object.attemptCount, "worker.attemptCount"),
    activeAttempt: expectNumber(object.activeAttempt, "worker.activeAttempt"),
    leaseOwner: expectNullableString(object.leaseOwner, "worker.leaseOwner"),
    leaseExpiresAt: expectNullableString(
      object.leaseExpiresAt,
      "worker.leaseExpiresAt"
    ),
    lastError: emptyStringToNull(
      expectNullableString(object.lastError, "worker.lastError")
    ),
    createdAt: expectString(object.createdAt, "worker.createdAt"),
    startedAt: expectNullableString(object.startedAt, "worker.startedAt"),
    completedAt: expectNullableString(object.completedAt, "worker.completedAt")
  };
}

export function decodeUsage(raw: unknown): UsageSnapshot {
  const object = expectObject(raw, "usage");
  return {
    inputTokens: expectNumber(object.inputTokens, "usage.inputTokens"),
    outputTokens: expectNumber(object.outputTokens, "usage.outputTokens"),
    thinkingTokens: expectNumber(object.thinkingTokens, "usage.thinkingTokens"),
    inputCostUsd: expectNumber(object.inputCostUsd, "usage.inputCostUsd"),
    outputCostUsd: expectNumber(object.outputCostUsd, "usage.outputCostUsd"),
    thinkingCostUsd: expectNumber(object.thinkingCostUsd, "usage.thinkingCostUsd"),
    costUsd: expectNumber(object.costUsd, "usage.costUsd"),
    costSource: expectString(object.costSource, "usage.costSource"),
    pricingVersion: expectString(object.pricingVersion, "usage.pricingVersion"),
    durationMs: expectNumber(object.durationMs, "usage.durationMs")
  };
}

function decodeRunStepDecisions(object: Record<string, unknown>): readonly StepDecision[] {
  if (object.decisions !== undefined) {
    return expectArray(object.decisions, "decisions").map(decodeStepDecision);
  }
  const responseBody = expectObject(object.responseBody, "responseBody");
  const decisions = responseBody.decisions;
  if (decisions === undefined) {
    return [];
  }
  return [{ lines: expectStringArray(decisions, "responseBody.decisions") }];
}

function emptyStringToNull(value: string | null): string | null {
  return value === "" ? null : value;
}

function semanticInputLookupText(input: Record<string, unknown>): string {
  if (input.lookupText !== undefined && input.lookupText !== null) {
    return expectString(input.lookupText, "lookupText");
  }
  if (input.resolvedValueText !== undefined && input.resolvedValueText !== null) {
    return expectString(input.resolvedValueText, "resolvedValueText");
  }
  return "";
}
