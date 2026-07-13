import type {
  AnswerOutput,
  ClarificationEvidence,
  ClarificationContinuation,
  ClarificationOwner,
  ClarificationOption,
  ClarificationRequest,
  ClarificationSubject,
  ResultData
} from "../contracts";
import {
  expectArray,
  expectBoolean,
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
    value: expectObject(object.value, "value"),
    displayValue: expectString(object.displayValue, "displayValue")
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
    ...decodeClarificationOwnerSpec(object.owner, object.continuation),
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

function decodeClarificationOwnerSpec(
  rawOwner: unknown,
  rawContinuation: unknown
): { owner: ClarificationOwner; continuation: ClarificationContinuation } {
  const owner = expectString(rawOwner, "clarification.owner") as ClarificationOwner;
  const continuation = decodeClarificationContinuation(rawContinuation);
  const expectedOwner: Record<ClarificationContinuation["kind"], ClarificationOwner> = {
    conversation_resolution: "conversation_resolution",
    question_contract: "question_contract",
    grounding: "grounding",
    source_binding_catalog_input: "source_binding",
    fact_planning_catalog_input: "fact_planning"
  };
  if (expectedOwner[continuation.kind] !== owner) {
    throw new Error("clarification owner and continuation must match");
  }
  return { owner, continuation };
}

function decodeClarificationContinuation(raw: unknown): ClarificationContinuation {
  const object = expectObject(raw, "clarification.continuation");
  const kind = expectString(object.kind, "clarification.continuation.kind");
  if (kind === "conversation_resolution") {
    return {
      kind,
      candidates: expectArray(object.candidates, "clarification.continuation.candidates").map(
        (rawCandidate) => {
          const candidate = expectObject(rawCandidate, "clarification candidate");
          return {
            id: expectString(candidate.id, "clarification candidate.id"),
            contextualizedQuestion: expectString(
              candidate.contextualizedQuestion,
              "clarification candidate.contextualizedQuestion"
            ),
            sourceEvidence: expectArray(
              candidate.sourceEvidence,
              "clarification candidate.sourceEvidence"
            ).map((rawEvidence) => {
              const evidence = expectObject(rawEvidence, "clarification candidate evidence");
              return {
                sourceId: expectString(evidence.sourceId, "candidate evidence.sourceId"),
                exactSourceTexts: expectArray(
                  evidence.exactSourceTexts,
                  "candidate evidence.exactSourceTexts"
                ).map((text) => expectString(text, "candidate evidence text"))
              };
            })
          };
        }
      ),
      acceptsFreeText: expectBoolean(
        object.acceptsFreeText,
        "clarification.continuation.acceptsFreeText"
      )
    };
  }
  if (kind === "question_contract") {
    return {
      kind,
      missingItemId: expectString(object.missingItemId, "clarification.continuation.missingItemId"),
      expectedValueKind: expectString(object.expectedValueKind, "clarification.continuation.expectedValueKind")
    };
  }
  if (kind === "grounding") {
    return {
      kind,
      knownInputId: expectString(object.knownInputId, "clarification.continuation.knownInputId"),
      acceptsFreeText: expectBoolean(object.acceptsFreeText, "clarification.continuation.acceptsFreeText")
    };
  }
  if (kind === "source_binding_catalog_input" || kind === "fact_planning_catalog_input") {
    const target = decodeCatalogInputTarget(object.target);
    const common = {
      requestedFactId: expectString(object.requestedFactId, "clarification.continuation.requestedFactId"),
      target
    };
    if (kind === "source_binding_catalog_input") {
      return { kind, ...common };
    }
    return {
      kind,
      ...common,
      planningRequirementId: expectString(
        object.planningRequirementId,
        "clarification.continuation.planningRequirementId"
      )
    };
  }
  throw new Error(`unsupported clarification continuation: ${kind}`);
}

function decodeCatalogInputTarget(raw: unknown) {
  const target = expectObject(raw, "clarification catalog target");
  return {
    rowSourceId: expectString(target.rowSourceId, "catalog target.rowSourceId"),
    paramId: expectString(target.paramId, "catalog target.paramId"),
    paramRef: expectString(target.paramRef, "catalog target.paramRef"),
    valueType: expectString(target.valueType, "catalog target.valueType"),
    choices: expectArray(target.choices, "catalog target.choices").map((choice) =>
      expectString(choice, "catalog target choice")
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
