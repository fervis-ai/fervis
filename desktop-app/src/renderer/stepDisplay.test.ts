import { describe, expect, it } from "vitest";

import { semanticStepSignalsFor } from "./stepDisplay";

describe("semantic step display", () => {
  it("renders unlabeled resolver candidates without internal punctuation noise", () => {
    expect(
      semanticStepSignalsFor("grounding", {
        requestedFacts: [],
        knownInputs: [],
        resolverCandidates: [
          {
            inputId: "input_1",
            resolverReadId: "",
            resolverLabel: "",
            basis: "The resolver can search location records by name."
          }
        ],
        groundingResults: [],
        interpretedInputs: [],
        conversationClauses: []
      })
    ).toEqual([
      {
        label: "Resolver",
        text: "The resolver can search location records by name."
      }
    ]);
  });

  it("renders interpreted time inputs as grounding proof", () => {
    expect(
      semanticStepSignalsFor("grounding", {
        requestedFacts: [],
        knownInputs: [],
        resolverCandidates: [],
        groundingResults: [],
        interpretedInputs: [
          {
            inputId: "input_period",
            inputText: "this month",
            kind: "time",
            value: "2026-06-01 to 2026-06-30",
            label: "this month",
            detail: "month"
          }
        ],
        conversationClauses: []
      })
    ).toEqual([
      {
        label: "Interpreted input",
        text: "this month: 2026-06-01 to 2026-06-30"
      }
    ]);
  });

  it("renders conversation-resolution clauses as clear follow-up meaning", () => {
    expect(
      semanticStepSignalsFor("conversation_resolution", {
        requestedFacts: [],
        knownInputs: [],
        resolverCandidates: [],
        groundingResults: [],
        interpretedInputs: [],
        conversationClauses: [
          {
            currentClauseText: "what about last month?",
            currentValueText: "what about last month?",
            resolvedFrameText: "count of completed in-person sales",
            resolvedClauseText: "how many completed in-person sales last month?"
          }
        ]
      })
    ).toEqual([
      {
        label: "Current clause",
        text: "what about last month?"
      },
      {
        label: "Resolved value",
        text: "what about last month? -> count of completed in-person sales"
      },
      {
        label: "Resolved question",
        text: "how many completed in-person sales last month?"
      }
    ]);
  });
});
