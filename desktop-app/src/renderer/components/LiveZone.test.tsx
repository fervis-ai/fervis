import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RunPayload, RunStep } from "../../fervis-api/contracts";
import {
  emptyStepSemanticFixture,
  runningRunFixture
} from "../../fervis-api/__fixtures__/payloads";
import { LiveZone } from "./LiveZone";

describe("LiveZone", () => {
  it("does not invent detail text when an early running step has no signals", () => {
    render(<LiveZone run={runningRunAt(stepWithoutDecisions("custom_empty_step"))} />);

    expect(screen.getByText("Custom Empty Step")).toBeInTheDocument();
    expect(
      screen.queryByText("Requested fact: Count of in-person sales this month")
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Known inputs: In-person sales")).not.toBeInTheDocument();
  });

  it("renders backend decision lines as running highlights", () => {
    render(
      <LiveZone
        run={runningRunAt({
          decisions: [{ lines: ["Selecting the sales endpoint for this month."] }],
          semantic: emptyStepSemanticFixture,
          stepId: "step_source_binding",
          stepKey: "source_binding"
        })}
      />
    );

    expect(
      screen.getByText("Decision: Selecting the sales endpoint for this month.")
    ).toBeInTheDocument();
  });

  it("renders backend semantic step highlights without deriving from trace strings", () => {
    render(
      <LiveZone
        run={runningRunAt({
          decisions: [],
          semantic: {
            groundingResults: [],
            knownInputs: [
              {
                description: "store",
                inputId: "fact_1_entity_1",
                kind: "named_reference_text",
                lookupText: "ABC Mall",
                text: "ABC Mall"
              },
              {
                description: "",
                inputId: "fact_1_time_1",
                kind: "time_text",
                lookupText: "",
                text: "this month"
              }
            ],
            requestedFacts: [
              {
                description: "sales at ABC Mall this month",
                requestedFactId: "fact_1"
              }
            ],
            resolverCandidates: [],
            interpretedInputs: [],
            conversationClauses: []
          },
          stepId: "step_question_contract",
          stepKey: "question_contract"
        })}
      />
    );

    expect(
      screen.getByText("Requested fact: Sales at ABC Mall this month")
    ).toBeInTheDocument();
    expect(screen.getByText("Known inputs: 1: ABC Mall · 2: this month")).toBeInTheDocument();
  });

  it("summarizes noisy decision traces into running highlights", () => {
    render(
      <LiveZone
        run={runningRunAt({
          decisions: [
            {
              lines: [
                "Read eligibility: retained 4 source candidates, dropped 6.",
                "source_1 list_sale_list: RETAIN - rows=2 - fields=8 - Exposes sale rows with sold_at, status, is_deleted, and sale_type so later turns can filter ordinary in-person sales within this month and count the matching sales.",
                "source_2 list_sales_summary: RETAIN - rows=3 - fields=12 - Exposes aggregated sales counts over a date range.",
                "source_3 list_products: DROP - Does not answer a sales count."
              ]
            }
          ],
          semantic: emptyStepSemanticFixture,
          stepId: "step_read_eligibility",
          stepKey: "read_eligibility"
        })}
      />
    );

    expect(screen.getByText("Eligibility: 4 retained, 6 dropped")).toBeInTheDocument();
    expect(
      screen.getByText("Resources: List Sale List, List Sales Summary")
    ).toBeInTheDocument();
    expect(screen.queryByText(/Exposes sale rows with sold_at/)).not.toBeInTheDocument();
  });

  it("labels bare plan-selection source references with resource names", () => {
    render(
      <LiveZone
        run={runningRunAt({
          decisions: [
            {
              lines: ["Source 1: DIRECT - best source for counting sales."]
            }
          ],
          semantic: emptyStepSemanticFixture,
          stepId: "step_plan_selection",
          stepKey: "plan_selection"
        })}
      />
    );

    expect(
      screen.getByText("Resource: source_1 (List Sale List)")
    ).toBeInTheDocument();
    expect(screen.queryByText("Resources: Source 1")).not.toBeInTheDocument();
  });
});

function runningRunAt(step: RunStep): RunPayload {
  return {
    ...runningRunFixture,
    explanation: {
      ...runningRunFixture.explanation,
      lineage: {
        ...runningRunFixture.explanation.lineage,
        compact: {
          questions: [
            {
              conversationId: runningRunFixture.conversationId,
              questionId: runningRunFixture.questionId,
              text: runningRunFixture.explanation.lineage.compact.questions[0].text,
              runs: [
                {
                  runId: runningRunFixture.runId,
                  runNumber: runningRunFixture.runNumber,
                  triggerKind: runningRunFixture.triggerKind,
                  steps: [
                    {
                      decisions: [
                        {
                          lines: [
                            "source_1 list_sale_list: RETAIN - rows=2 - fields=8 - Exposes sale rows."
                          ]
                        }
                      ],
                      runtimeErrors: [],
                      semantic: emptyStepSemanticFixture,
                      sequence: 1,
                      sourceReads: [],
                      stepId: "step_prior_read_eligibility",
                      stepKey: "read_eligibility"
                    }
                  ]
                }
              ]
            }
          ]
        }
      }
    },
    steps: [step]
  };
}

function stepWithoutDecisions(stepKey: string): RunStep {
  return {
    decisions: [],
    semantic: emptyStepSemanticFixture,
    stepId: `step_${stepKey}`,
    stepKey
  };
}
