import type {
  AnswerOutput,
  ClarificationOption,
  ClarificationRequest,
  ResultData
} from "../contracts";
import {
  expectArray,
  expectNullableString,
  expectObject,
  expectString,
  expectStringArray
} from "./primitives";
import { decodeValueKind } from "./enums";

export function decodeResultData(raw: unknown): ResultData {
  if (raw === null) {
    return null;
  }
  const object = expectObject(raw, "resultData");
  const kind = expectString(object.kind, "resultData.kind");
  if (kind === "answer") {
    return {
      kind,
      outputs: expectArray(object.outputs, "resultData.outputs").map(decodeAnswerOutput)
    };
  }
  if (kind === "needs_clarification") {
    const details = expectObject(object.details, "resultData.details");
    const clarifications = expectArray(
      details.clarifications,
      "resultData.details.clarifications"
    ).map(decodeClarification);
    if (clarifications.length === 0) {
      throw new Error("resultData.details.clarifications must include an actionable clarification");
    }
    return {
      kind,
      details: { clarifications }
    };
  }
  throw new Error(`unsupported resultData.kind: ${kind}`);
}

function decodeAnswerOutput(raw: unknown): AnswerOutput {
  const object = expectObject(raw, "answer output");
  return {
    key: expectString(object.key, "key"),
    valueKind: decodeValueKind(object.valueKind, "valueKind"),
    value: expectString(object.value, "value")
  };
}

function decodeClarification(raw: unknown): ClarificationRequest {
  const object = expectObject(raw, "clarification");
  const id = expectString(object.id, "clarification.id");
  if (id.trim() === "") {
    throw new Error("clarification.id must not be empty");
  }
  return {
    id,
    basis: expectString(object.basis, "clarification.basis"),
    question: expectString(object.question, "clarification.question"),
    availableOptions: decodeClarificationOptions(object.availableOptions),
    evidenceRefs: expectStringArray(object.evidenceRefs, "clarification.evidenceRefs"),
    factResultId: expectNullableString(
      object.factResultId,
      "clarification.factResultId"
    ),
    stepId: expectNullableString(object.stepId, "clarification.stepId")
  };
}

function decodeClarificationOptions(
  raw: unknown
): readonly ClarificationOption[] {
  if (raw === undefined) {
    return [];
  }
  return expectArray(raw, "clarification.availableOptions").map(
    decodeClarificationOption
  );
}

function decodeClarificationOption(raw: unknown): ClarificationOption {
  const object = expectObject(raw, "clarification option");
  return {
    id: expectString(object.id, "option.id"),
    label: expectString(object.label, "option.label")
  };
}
