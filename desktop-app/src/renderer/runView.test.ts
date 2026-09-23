import { describe, expect, it } from "vitest";

import {
  clarificationRunFixture,
  completedRunFixture
} from "../fervis-api/__fixtures__/payloads";
import type { RunPayload } from "../fervis-api/contracts";
import { firstClarification, runSummary } from "./runView";

describe("run view projection", () => {
  it("does not present historical clarification data as pending on a completed run", () => {
    const run = {
      ...completedRunFixture,
      resultData: clarificationRunFixture.resultData
    } satisfies RunPayload;

    expect(firstClarification(run)).toBeNull();
    expect(runSummary(run)).not.toContain("clarification:");
  });
});

it("does not poll completed factual limitations indefinitely", async () => {
  const { decodeRun } = await import("../fervis-api/decoder");
  const { pollableRun, completedAnswerText } = await import("./runView");
  const decoded = decodeRun({ ...completedRunFixture, answer: null,
    resultData: { kind: "no_data", message: "No matching data was found.", emptyRelation: { relationId: "empty" } } });
  if (!decoded.ok) throw new Error(decoded.error.message);
  expect(pollableRun(decoded.value)).toBe(false);
  expect(completedAnswerText(decoded.value)).toBe("No matching data was found.");
});
