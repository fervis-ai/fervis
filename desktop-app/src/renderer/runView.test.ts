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
