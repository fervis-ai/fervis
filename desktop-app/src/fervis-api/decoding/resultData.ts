import type {
  AnswerOutput,
  ClarificationEvidence,
  ClarificationOption,
  ClarificationRequest,
  ClarificationSubject,
  ResultData
} from "../contracts";
import {
  expectArray,
  expectNullableString,
  expectObject,
  expectString
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
    need: expectString(object.need, "clarification.need"),
    reason: expectString(object.reason, "clarification.reason"),
    question: expectString(object.question, "clarification.question"),
    requestedFactId: expectString(
      object.requestedFactId,
      "clarification.requestedFactId"
    ),
    subjects: expectArray(object.subjects, "clarification.subjects").map(
      decodeClarificationSubject
    ),
    evidence: expectArray(object.evidence, "clarification.evidence").map(
      decodeClarificationEvidence
    )
  };
}

function decodeOptionalNullableString(raw: unknown, label: string): string | null {
  if (raw === undefined) {
    return null;
  }
  return expectNullableString(raw, label);
}

function decodeClarificationSubject(raw: unknown): ClarificationSubject {
  const object = expectObject(raw, "clarification subject");
  return {
    kind: expectString(object.kind, "clarification.subject.kind"),
    id: expectString(object.id, "clarification.subject.id"),
    label: expectString(object.label, "clarification.subject.label"),
    sourceText: expectString(object.sourceText, "clarification.subject.sourceText"),
    options: expectArray(
      object.options,
      "clarification.subject.options"
    ).map(decodeClarificationOption)
  };
}

function decodeClarificationOption(raw: unknown): ClarificationOption {
  const object = expectObject(raw, "clarification option");
  return {
    id: expectString(object.id, "option.id"),
    label: expectString(object.label, "option.label"),
    value: decodeOptionalNullableString(object.value, "option.value"),
    entityKind: decodeOptionalNullableString(object.entityKind, "option.entityKind"),
    matchedLabel: decodeOptionalNullableString(object.matchedLabel, "option.matchedLabel"),
    matchedField: decodeOptionalNullableString(object.matchedField, "option.matchedField"),
    matchedValue: decodeOptionalNullableString(object.matchedValue, "option.matchedValue"),
    resolverReadId: decodeOptionalNullableString(object.resolverReadId, "option.resolverReadId"),
    resolverLabel: decodeOptionalNullableString(object.resolverLabel, "option.resolverLabel")
  };
}

function decodeClarificationEvidence(raw: unknown): ClarificationEvidence {
  const object = expectObject(raw, "clarification evidence");
  return {
    kind: expectString(object.kind, "clarification.evidence.kind"),
    id: expectString(object.id, "clarification.evidence.id"),
    readId: decodeOptionalNullableString(
      object.readId,
      "clarification.evidence.readId"
    ),
    endpointName: decodeOptionalNullableString(
      object.endpointName,
      "clarification.evidence.endpointName"
    ),
    fieldId: decodeOptionalNullableString(
      object.fieldId,
      "clarification.evidence.fieldId"
    ),
    identityField: decodeOptionalNullableString(
      object.identityField,
      "clarification.evidence.identityField"
    )
  };
}
