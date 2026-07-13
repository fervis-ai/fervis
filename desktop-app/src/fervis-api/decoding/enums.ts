import type {
  AnswerValueKind,
  RunKind,
  RunStatus,
  RunTriggerKind
} from "../contracts";
import { expectString } from "./primitives";

export function decodeRunStatus(raw: unknown, label: string): RunStatus {
  const value = expectString(raw, label);
  if (
    value === "QUEUED" ||
    value === "RUNNING" ||
    value === "COMPLETED" ||
    value === "WAITING_FOR_CLARIFICATION" ||
    value === "FAILED"
  ) {
    return value;
  }
  throw new Error(`${label} has unsupported run status: ${value}`);
}

export function decodeTriggerKind(
  raw: unknown,
  label: string
): RunTriggerKind {
  const value = expectString(raw, label);
  if (
    value === "initial" ||
    value === "retry" ||
    value === "rerun"
  ) {
    return value;
  }
  throw new Error(`${label} has unsupported trigger kind: ${value}`);
}

export function decodeRunKind(raw: unknown, label: string): RunKind {
  const value = expectString(raw, label);
  if (value === "model_assisted" || value === "deterministic") {
    return value;
  }
  throw new Error(`${label} has unsupported run kind: ${value}`);
}

export function decodeValueKind(raw: unknown, label: string): AnswerValueKind {
  const value = expectString(raw, label);
  if (
    value === "entity" ||
    value === "number" ||
    value === "money" ||
    value === "boolean" ||
    value === "text" ||
    value === "date" ||
    value === "datetime" ||
    value === "table" ||
    value === "list" ||
    value === "object"
  ) {
    return value;
  }
  throw new Error(`${label} has unsupported answer value kind: ${value}`);
}

export function decodeNextActionKind(
  raw: unknown,
  label: string
): "inspect_question" | "provide_clarification" | "retry" {
  const value = expectString(raw, label);
  if (
    value === "inspect_question" ||
    value === "provide_clarification" ||
    value === "retry"
  ) {
    return value;
  }
  throw new Error(`${label} has unsupported action kind: ${value}`);
}
