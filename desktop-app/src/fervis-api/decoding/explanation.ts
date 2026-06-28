import type {
  ExplanationPayload,
  InputResult,
  LineageQuestion,
  LineageRun,
  LineageStep
} from "../contracts";
import {
  expectArray,
  expectNumber,
  expectObject,
  expectString,
  expectStringArray
} from "./primitives";
import {
  decodeRunError,
  decodeSourceRead,
  decodeStepDecision,
  decodeStepSemantic
} from "./runDetails";

export function decodeExplanation(raw: unknown): ExplanationPayload {
  const object = expectObject(raw, "explanation");
  const inputs = expectObject(object.inputs, "explanation.inputs");
  const lineage = expectObject(object.lineage, "explanation.lineage");
  return {
    inputs: {
      results: expectArray(inputs.results, "explanation.inputs.results").map(
        decodeInputResult
      )
    },
    lineage: {
      compact: {
        questions: expectArray(
          expectObject(lineage.compact, "lineage.compact").questions,
          "lineage.compact.questions"
        ).map(decodeLineageQuestion)
      },
      verbose: {
        questions: expectArray(
          expectObject(lineage.verbose, "lineage.verbose").questions,
          "lineage.verbose.questions"
        ).map(decodeLineageQuestion)
      }
    }
  };
}

function decodeInputResult(raw: unknown): InputResult {
  const object = expectObject(raw, "input result");
  return {
    factResultId: expectString(object.factResultId, "factResultId"),
    requestedFactId: expectString(object.requestedFactId, "requestedFactId"),
    factDescription: expectString(object.factDescription, "factDescription"),
    explicit: expectStringArray(object.explicit, "explicit"),
    derived: expectStringArray(object.derived, "derived"),
    contextual: expectStringArray(object.contextual, "contextual"),
    applied: expectStringArray(object.applied, "applied"),
    evidenceRefs: expectStringArray(object.evidenceRefs, "evidenceRefs"),
    proofHandles: expectStringArray(object.proofHandles, "proofHandles")
  };
}

function decodeLineageQuestion(raw: unknown): LineageQuestion {
  const object = expectObject(raw, "lineage question");
  return {
    questionId: expectString(object.questionId, "questionId"),
    conversationId: expectString(object.conversationId, "conversationId"),
    text: expectString(object.text, "text"),
    runs: expectArray(object.runs, "runs").map(decodeLineageRun)
  };
}

function decodeLineageRun(raw: unknown): LineageRun {
  const object = expectObject(raw, "lineage run");
  return {
    runId: expectString(object.runId, "runId"),
    runNumber: expectNumber(object.runNumber, "runNumber"),
    triggerKind: expectString(object.triggerKind, "triggerKind"),
    steps: expectArray(object.steps, "steps").map(decodeLineageStep)
  };
}

function decodeLineageStep(raw: unknown): LineageStep {
  const object = expectObject(raw, "lineage step");
  return {
    stepId: expectString(object.stepId, "stepId"),
    stepKey: expectString(object.stepKey, "stepKey"),
    sequence: expectNumber(object.sequence, "sequence"),
    decisions: expectArray(object.decisions, "decisions").map(decodeStepDecision),
    semantic: decodeStepSemantic(object.semantic),
    sourceReads: expectArray(object.sourceReads, "sourceReads").map(decodeSourceRead),
    runtimeErrors: expectArray(object.runtimeErrors, "runtimeErrors").map(decodeRunError)
  };
}
