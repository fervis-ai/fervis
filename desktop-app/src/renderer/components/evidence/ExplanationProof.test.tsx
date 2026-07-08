import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { LineageStep, RunPayload } from "../../../fervis-api/contracts";
import {
  completedRunFixture,
  emptyStepSemanticFixture
} from "../../../fervis-api/__fixtures__/payloads";
import { ExplanationProof } from "./ExplanationProof";

describe("ExplanationProof", () => {
  it("does not show unrelated details for empty early steps", () => {
    render(
      <ExplanationProof
        mode="verbose"
        run={runWithLineageSteps(
          [emptyLineageStep("query_enrichment"), emptyLineageStep("grounding")],
          [emptyLineageStep("query_enrichment"), emptyLineageStep("grounding")]
        )}
      />
    );

    expect(screen.getByText("Query Enrichment")).toBeInTheDocument();
    expect(screen.getByText("Grounding")).toBeInTheDocument();
    expect(screen.queryByText("Business context")).not.toBeInTheDocument();
    expect(screen.queryByText("Evidence refs")).not.toBeInTheDocument();
    expect(screen.queryByText("Current month")).not.toBeInTheDocument();
    expect(screen.queryByText("Ev Sales")).not.toBeInTheDocument();
  });

  it("shows semantic answer-contract signals for the owning step", () => {
    render(
      <ExplanationProof
        mode="verbose"
        run={runWithLineageSteps(
          [
            semanticLineageStep("question_contract", {
              groundingResults: [],
              knownInputs: [
                {
                  description: "store",
                  inputId: "fact_1_entity_1",
                  kind: "named_reference_text",
                  lookupText: "ABC Mall",
                  text: "ABC Mall"
                }
              ],
              requestedFacts: [
                {
                  description: "sales at ABC Mall this month",
                  requestedFactId: "fact_1"
                }
              ],
              resolverCandidates: []
            })
          ],
          [
            semanticLineageStep("question_contract", {
              groundingResults: [],
              knownInputs: [
                {
                  description: "store",
                  inputId: "fact_1_entity_1",
                  kind: "named_reference_text",
                  lookupText: "ABC Mall",
                  text: "ABC Mall"
                }
              ],
              requestedFacts: [
                {
                  description: "sales at ABC Mall this month",
                  requestedFactId: "fact_1"
                }
              ],
              resolverCandidates: []
            })
          ]
        )}
      />
    );

    expect(screen.getByText("Question Contract")).toBeInTheDocument();
    expect(screen.getByText("Requested fact")).toBeInTheDocument();
    expect(screen.getByText("Sales at ABC Mall this month")).toBeInTheDocument();
  });

  it("shows semantic resolver and grounding signals without leaking generic context", () => {
    render(
      <ExplanationProof
        mode="verbose"
        run={runWithLineageSteps(
          [
            semanticLineageStep("query_enrichment", {
              groundingResults: [],
              knownInputs: [],
              requestedFacts: [],
              resolverCandidates: [
                {
                  basis: "location can identify ABC Mall because target meaning is store.",
                  inputId: "fact_1_entity_1",
                  resolverLabel: "List Location List",
                  resolverReadId: "list_location_list"
                }
              ]
            }),
            semanticLineageStep("grounding", {
              groundingResults: [
                {
                  inputId: "fact_1_entity_1",
                  inputText: "ABC Mall",
                  entityKind: "location",
                  matchedField: "location_id",
                  matchedLabel: "ABC Mall",
                  matchedValue: "60606060-0000-0000-0001-000000000001",
                  resolverLabel: "List Location List",
                  resolverReadId: "list_location_list"
                }
              ],
              knownInputs: [],
              requestedFacts: [],
              resolverCandidates: []
            })
          ],
          [
            semanticLineageStep("query_enrichment", {
              groundingResults: [],
              knownInputs: [],
              requestedFacts: [],
              resolverCandidates: [
                {
                  basis: "location can identify ABC Mall because target meaning is store.",
                  inputId: "fact_1_entity_1",
                  resolverLabel: "List Location List",
                  resolverReadId: "list_location_list"
                }
              ]
            }),
            semanticLineageStep("grounding", {
              groundingResults: [
                {
                  inputId: "fact_1_entity_1",
                  inputText: "ABC Mall",
                  entityKind: "location",
                  matchedField: "location_id",
                  matchedLabel: "ABC Mall",
                  matchedValue: "60606060-0000-0000-0001-000000000001",
                  resolverLabel: "List Location List",
                  resolverReadId: "list_location_list"
                }
              ],
              knownInputs: [],
              requestedFacts: [],
              resolverCandidates: []
            })
          ]
        )}
      />
    );

    expect(screen.getByText("Query Enrichment")).toBeInTheDocument();
    expect(screen.getByText("Resolver")).toBeInTheDocument();
    expect(screen.getByText(/List Location List:/)).toBeInTheDocument();
    expect(screen.getByText("Grounding")).toBeInTheDocument();
    expect(
      screen.getByText(
        "\"ABC Mall\": Location (location_id: 60606060-0000-0000-0001-000000000001 via List Location List)"
      )
    ).toBeInTheDocument();
    expect(screen.queryByText("Business context")).not.toBeInTheDocument();
  });

  it("keeps semantic grounding visible when the grounding step has resolver reads", () => {
    render(
      <ExplanationProof
        mode="compact"
        run={runWithLineageSteps(
          [
            {
              ...semanticLineageStep("grounding", {
                groundingResults: [
                  {
                    inputId: "fact_1_entity_1",
                    inputText: "ABC Mall",
                    entityKind: "location",
                    matchedField: "location_id",
                    matchedLabel: "ABC Mall",
                    matchedValue: "60606060-0000-0000-0001-000000000001",
                    resolverLabel: "List Location List",
                    resolverReadId: "list_location_list"
                  }
                ],
                interpretedInputs: [
                  {
                    detail: "month",
                    inputId: "fact_1_time_1",
                    inputText: "this month",
                    kind: "time",
                    label: "this month",
                    value: "2026-06-01 to 2026-06-30"
                  }
                ]
              }),
              sourceReads: [
                {
                  method: "GET",
                  path: "/v1/locations/",
                  rowCount: 25,
                  sourceReadId: "source_read_location",
                  status: "succeeded"
                }
              ]
            }
          ],
          []
        )}
      />
    );

    expect(screen.getByText("Grounding")).toBeInTheDocument();
    expect(screen.getByText("Inputs:")).toBeInTheDocument();
    expect(
      screen.getByText("\"this month\": 2026-06-01 to 2026-06-30")
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "\"ABC Mall\": Location (location_id: 60606060-0000-0000-0001-000000000001 via List Location List)"
      )
    ).toBeInTheDocument();
  });

  it("renders source-selection rationale without model-facing row or field prefixes", () => {
    const step = lineageStepWithDecisions([
      "source_1 list_sale_list: RETAIN - rows=2 - fields=7 - Exposes sale rows with sold_at, status, is_deleted, and sale_type, which can support counting."
    ]);

    render(<ExplanationProof mode="verbose" run={runWithLineageSteps([step], [step])} />);

    expect(screen.getByText("source_1 (List Sale List) · used")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Exposes sale rows with sold at, status, is deleted, and sale type, which can support counting."
      )
    ).toBeInTheDocument();
  });

  it("labels reviewed source handles with endpoint names", () => {
    const step = lineageStepWithDecisions([
      "source_1 list_sale_list: RETAIN - rows=2 - fields=7 - Exposes sale rows.",
      "source_2 list_sales_summary: RETAIN - rows=3 - fields=12 - Exposes sales summaries.",
      "Reviewed source candidates: source_1, source_2"
    ]);

    render(<ExplanationProof mode="verbose" run={runWithLineageSteps([step], [step])} />);

    expect(screen.getByText("Source candidates")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Reviewed source_1 (List Sale List), source_2 (List Sales Summary)."
      )
    ).toBeInTheDocument();
  });

  it("keeps all semantic conversation-resolution signals in compact evidence", () => {
    render(
      <ExplanationProof
        mode="compact"
        run={runWithLineageSteps(
          [
            semanticLineageStep("conversation_resolution", {
              conversationClauses: [
                {
                  currentClauseText: "what about last month?",
                  currentValueText: "what about last month?",
                  resolvedFrameText: "count of completed in-person sales",
                  resolvedClauseText: "how many completed in-person sales last month?"
                }
              ]
            })
          ],
          []
        )}
      />
    );

    expect(screen.getByText("Current clause")).toBeInTheDocument();
    expect(screen.getByText("Resolved value")).toBeInTheDocument();
    expect(screen.getByText("Resolved question")).toBeInTheDocument();
    expect(
      screen.getByText("how many completed in-person sales last month?")
    ).toBeInTheDocument();
  });

  it("does not replace real proof lines with invented stage text", () => {
    render(<ExplanationProof mode="verbose" run={runWithManyProofNotes()} />);

    expect(
      screen.getByText("1 source candidate retained; 0 dropped.")
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Checking which API reads can answer this question.")
    ).not.toBeInTheDocument();
  });

  it("keeps decision-backed reasoning steps in compact mode", () => {
    render(<ExplanationProof mode="compact" run={runWithManyProofNotes()} />);

    expect(screen.getByText("Read Eligibility")).toBeInTheDocument();
    expect(
      screen.getByText("1 source candidate retained; 0 dropped.")
    ).toBeInTheDocument();
  });

  it("expands lower-signal details with the shared plus affordance", () => {
    render(<ExplanationProof mode="verbose" run={runWithManyProofNotes()} />);

    expect(screen.getByLabelText("Show lower-signal details")).toBeInTheDocument();
    expect(screen.queryByText("Lower signal detail 5.")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Show lower-signal details"));

    expect(screen.getByLabelText("Hide lower-signal details")).toBeInTheDocument();
    expect(screen.getByText("Lower signal detail 5.")).toBeInTheDocument();
  });
});

function runWithManyProofNotes(): RunPayload {
  const step = lineageStepWithDecisions([
    "Retained 1 source candidate, dropped 0",
    "Source 1 orders: RETAIN - Rows: 2 - Fields: 4 - Answers the requested order count.",
    "Reviewed source candidates: source 1",
    "Lower signal detail 4",
    "Lower signal detail 5"
  ]);
  return {
    ...completedRunFixture,
    explanation: {
      ...completedRunFixture.explanation,
      lineage: {
        compact: {
          questions: [
            {
              conversationId: completedRunFixture.conversationId,
              questionId: completedRunFixture.questionId,
              text: "How many orders?",
              runs: [
                {
                  runId: completedRunFixture.runId,
                  runNumber: completedRunFixture.runNumber,
                  triggerKind: completedRunFixture.triggerKind,
                  steps: [step]
                }
              ]
            }
          ]
        },
        verbose: {
          questions: [
            {
              conversationId: completedRunFixture.conversationId,
              questionId: completedRunFixture.questionId,
              text: "How many orders?",
              runs: [
                {
                  runId: completedRunFixture.runId,
                  runNumber: completedRunFixture.runNumber,
                  triggerKind: completedRunFixture.triggerKind,
                  steps: [step]
                }
              ]
            }
          ]
        }
      }
    }
  };
}

function runWithEmptyStep(stepKey: string): RunPayload {
  const step = emptyLineageStep(stepKey);
  return runWithLineageSteps([step], [step]);
}

function runWithLineageSteps(
  compactSteps: readonly LineageStep[],
  verboseSteps: readonly LineageStep[]
): RunPayload {
  return {
    ...completedRunFixture,
    explanation: {
      ...completedRunFixture.explanation,
      lineage: {
        compact: {
          questions: [
            {
              conversationId: completedRunFixture.conversationId,
              questionId: completedRunFixture.questionId,
              text: "How many orders?",
              runs: [
                {
                  runId: completedRunFixture.runId,
                  runNumber: completedRunFixture.runNumber,
                  triggerKind: completedRunFixture.triggerKind,
                  steps: compactSteps
                }
              ]
            }
          ]
        },
        verbose: {
          questions: [
            {
              conversationId: completedRunFixture.conversationId,
              questionId: completedRunFixture.questionId,
              text: "How many orders?",
              runs: [
                {
                  runId: completedRunFixture.runId,
                  runNumber: completedRunFixture.runNumber,
                  triggerKind: completedRunFixture.triggerKind,
                  steps: verboseSteps
                }
              ]
            }
          ]
        }
      }
    }
  };
}

function emptyLineageStep(stepKey: string): LineageStep {
  return {
    decisions: [],
    runtimeErrors: [],
    semantic: emptyStepSemanticFixture,
    sequence: 1,
    sourceReads: [],
    stepId: `step_${stepKey}`,
    stepKey
  };
}

function semanticLineageStep(
  stepKey: string,
  semantic: Partial<LineageStep["semantic"]>
): LineageStep {
  return {
    ...emptyLineageStep(stepKey),
    semantic: {
      ...emptyStepSemanticFixture,
      ...semantic
    }
  };
}

function lineageStepWithDecisions(lines: readonly string[]): LineageStep {
  return {
    decisions: lines.map((line) => ({ lines: [line] })),
    runtimeErrors: [],
    semantic: emptyStepSemanticFixture,
    sequence: 1,
    sourceReads: [],
    stepId: "step_many_notes",
    stepKey: "read_eligibility"
  };
}
