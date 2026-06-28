import type { NextAction, NextActionRequest } from "../contracts";
import { decodeNextActionKind } from "./enums";
import { expectArray, expectNullableString, expectObject, expectString } from "./primitives";

export function decodeNextActions(raw: unknown): readonly NextAction[] {
  if (raw === undefined || raw === null) {
    return [];
  }
  return expectArray(raw, "nextActions").map(decodeNextAction);
}

function decodeNextAction(raw: unknown): NextAction {
  const object = expectObject(raw, "next action");
  return {
    kind: decodeNextActionKind(object.kind, "nextAction.kind"),
    description:
      object.description === undefined
        ? null
        : expectNullableString(object.description, "nextAction.description"),
    command:
      object.command === undefined
        ? null
        : expectNullableString(object.command, "nextAction.command"),
    request: decodeNextActionRequest(object.request)
  };
}

function decodeNextActionRequest(raw: unknown): NextActionRequest | null {
  if (raw === undefined || raw === null) {
    return null;
  }
  const object = expectObject(raw, "nextAction.request");
  return {
    method: expectString(object.method, "nextAction.request.method"),
    path: expectString(object.path, "nextAction.request.path")
  };
}
