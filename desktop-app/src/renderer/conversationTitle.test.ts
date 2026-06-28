import { describe, expect, it } from "vitest";

import { titleFromQuestion } from "./conversationTitle";

describe("conversation label derivation", () => {
  it("uses the first tokens of the question", () => {
    expect(
      titleFromQuestion("How many in-person sales happened this month at ABC Mall?", 6)
    ).toBe("How many in-person sales happened this…");
  });

  it("does not add ellipsis when the question fits", () => {
    expect(titleFromQuestion("Count sales today", 6)).toBe("Count sales today");
  });

  it("normalizes extra whitespace", () => {
    expect(titleFromQuestion("  Count   sales   today  ", 6)).toBe("Count sales today");
  });
});
