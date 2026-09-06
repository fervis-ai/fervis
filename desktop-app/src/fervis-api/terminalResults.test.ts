import { describe, expect, it } from "vitest";
import terminalResults from "../../../python/tests/contracts/fixtures/public_terminal_results.json";
import { completedRunFixture, runListFixture } from "./__fixtures__/payloads";
import { decodeRun, decodeQuestionRunList } from "./decoder";

describe("backend factual terminal contract", () => {
  for (const [kind, resultData] of Object.entries(terminalResults)) {
    it(`loads a ${kind} run without discarding its typed result`, () => {
      const decoded = decodeRun({ ...completedRunFixture, resultData });
      expect(decoded.ok).toBe(true);
      if (decoded.ok) expect(decoded.value.resultData?.kind).toBe(kind);
    });
  }
  it("keeps a run list readable when one run has a factual limitation", () => {
    const decoded = decodeQuestionRunList({
      ...runListFixture,
      runs: [{ ...completedRunFixture, resultData: terminalResults.impossible }]
    });
    expect(decoded.ok).toBe(true);
  });
});
