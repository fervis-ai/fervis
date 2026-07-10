import { describe, expect, it } from "vitest";

import {
  completedRunFixture,
  conversationListFixture,
  failedRunFixture,
  freeTextClarificationRunFixture,
  questionStateFixture,
  runListFixture
} from "./__fixtures__/payloads";
import {
  decodeConversationList,
  decodeQuestionRunList,
  decodeQuestionState,
  decodeRun
} from "./decoder";

describe("Fervis API boundary decoder", () => {
  it("decodes the conversation rail payload in backend order", () => {
    const decoded = decodeConversationList(conversationListFixture);

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.ok).toBe(true);
    expect(decoded.value.conversations.map((item) => item.conversationId)).toEqual([
      "conv_new",
      "conv_old"
    ]);
    expect(decoded.value.conversations[0]?.firstQuestion).toBe(
      "How many orders came in today?"
    );
  });

  it("decodes an empty primary-run projection", () => {
    const decoded = decodeConversationList({
      conversations: [
        {
          conversationId: "conv_pending",
          firstQuestion: "Which staff person made the most sales this month?",
          latestQuestionId: "q_pending",
          primaryRunId: null,
          latestRunId: null,
          activeRunId: null,
          status: "RUNNING",
          runCount: 1,
          updatedAt: "2026-06-29T08:36:00.438620+00:00"
        }
      ]
    });

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.conversations[0]?.primaryRunId).toBeNull();
  });

  it("decodes question state without requiring worker or usage diagnostics", () => {
    const decoded = decodeQuestionState(questionStateFixture);

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.ok).toBe(true);
    expect(decoded.value.primaryRunId).toBe("run_sales");
    expect(decoded.value.resultData?.kind).toBe("answer");
  });

  it("decodes the live Fervis question-state field names", () => {
    const decoded = decodeQuestionState({
      questionId: "q_live",
      conversationId: "conv_live",
      tenantId: "default",
      status: "COMPLETED",
      primaryRunId: "run_live",
      latestRunId: "run_live",
      activeRunId: null,
      question: "How many in-person sales happened this month?",
      answer: "1",
      resultData: {
        kind: "answer",
        outputs: [{ key: "answer_1", valueKind: "number", value: "1" }]
      },
      error: ""
    });

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.ok).toBe(true);
    expect(decoded.value.question).toBe(
      "How many in-person sales happened this month?"
    );
    expect(decoded.value.resultData?.kind).toBe("answer");
    if (decoded.value.resultData?.kind !== "answer") {
      throw new Error("expected answer result data");
    }
    expect(decoded.value.resultData.outputs[0]?.valueKind).toBe("number");
    expect(decoded.value.nextActions).toEqual([]);
  });

  it("decodes live clarification payloads with omitted optional lists", () => {
    const decoded = decodeQuestionState({
      questionId: "q_clarify",
      conversationId: "conv_clarify",
      status: "NEEDS_CLARIFICATION",
      primaryRunId: "run_clarify",
      latestRunId: "run_clarify",
      activeRunId: null,
      question: "Show me performance for ABC Mall yesterday.",
      answer: null,
      resultData: {
        kind: "needs_clarification",
        details: {
          clarifications: [
            {
              id: "clarification_1",
              need: "answer_metric",
              reason: "missing_answer_metric",
              question: "Which metric should I use?",
              requestedFactId: "fact_1",
              subjects: [
                {
                  kind: "metric_phrase",
                  id: "clarification_1",
                  label: "metric",
                  sourceText: "",
                  options: []
                }
              ],
              evidence: []
            }
          ]
        }
      }
    });

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.resultData?.kind).toBe("needs_clarification");
    if (decoded.value.resultData?.kind !== "needs_clarification") {
      throw new Error("expected clarification result data");
    }
    expect(
      decoded.value.resultData.details.clarifications[0]?.subjects[0]?.options
    ).toEqual([]);
    expect(decoded.value.resultData.details.clarifications[0]?.evidence).toEqual([]);
    expect(decoded.value.resultData.details.clarifications[0]?.requestedFactId).toBe(
      "fact_1"
    );
    expect(decoded.value.resultData.details.clarifications[0]?.reason).toBe(
      "missing_answer_metric"
    );
  });

  it("decodes semantic known inputs without lookup text", () => {
    const decoded = decodeQuestionRunList({
      questionId: "q_clarify",
      runs: [
        {
          runId: "run_clarify",
          runNumber: 1,
          kind: "model_assisted",
          triggerKind: "initial",
          baseRunId: null,
          programId: null,
          invocationId: null,
          patchId: null,
          revisionId: null,
          questionId: "q_clarify",
          conversationId: "conv_clarify",
          status: "NEEDS_CLARIFICATION",
          answer: null,
          resultData: {
            kind: "needs_clarification",
            details: {
              clarifications: [
                {
                  id: "clarification_1",
                  need: "target_reference",
                  reason: "unresolved_reference",
                  question: "Which staff identifier do you mean?",
                  requestedFactId: "fact_1",
                  subjects: [
                    {
                      kind: "question_input",
                      id: "q1",
                      label: "staff identifier",
                      sourceText: "51515151-0000-0000-0002-000000009999",
                      options: []
                    }
                  ],
                  evidence: [{ kind: "known_input", id: "known_input:q1" }]
                }
              ]
            }
          },
          explanation: null,
          steps: [
            {
              stepId: "step_question_contract",
              stepKey: "question_contract",
              decisions: [],
              semantic: {
                requestedFacts: [],
                knownInputs: [
                  {
                    inputId: "q1",
                    text: "staff_id: 51515151-0000-0000-0002-000000009999",
                    kind: "literal_text",
                    role: "reference_value",
                    description: "staff identifier",
                    resolvedValueText: "51515151-0000-0000-0002-000000009999"
                  }
                ],
                resolverCandidates: [],
                groundingResults: [],
                interpretedInputs: [],
                conversationClauses: []
              }
            }
          ],
          error: null,
          nextActions: []
        }
      ]
    });

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.runs[0]?.steps[0]?.semantic.knownInputs[0]?.lookupText).toBe(
      "51515151-0000-0000-0002-000000009999"
    );
  });

  it("decodes lineage runtime errors using errorKind", () => {
    const decoded = decodeRun({
      runId: "run_failed",
      runNumber: 1,
      kind: "model_assisted",
      triggerKind: "initial",
      baseRunId: null,
      programId: null,
      invocationId: null,
      patchId: null,
      revisionId: null,
      questionId: "q_failed",
      conversationId: "conv_failed",
      status: "FAILED",
      answer: null,
      resultData: null,
      error: "provider_bad_request",
      explanation: {
        inputs: { results: [] },
        lineage: {
          compact: {
            questions: [
              {
                questionId: "q_failed",
                conversationId: "conv_failed",
                text: "How many stores do we have?",
                runs: [
                  {
                    runId: "run_failed",
                    runNumber: 1,
                    triggerKind: "initial",
                    steps: [
                      {
                        stepId: "step_1",
                        stepKey: "question_contract",
                        sequence: 1,
                        decisions: [],
                        semantic: {
                          requestedFacts: [],
                          knownInputs: [],
                          resolverCandidates: [],
                          groundingResults: [],
                          interpretedInputs: [],
                          conversationClauses: []
                        },
                        sourceReads: [],
                        runtimeErrors: [
                          {
                            runtimeErrorDetailId: "runtime_error_1",
                            errorKind: "infrastructure_failed",
                            message: "provider_bad_request",
                            failedStepId: "step_1",
                            failedStepKey: "question_contract"
                          }
                        ]
                      }
                    ]
                  }
                ]
              }
            ]
          },
          verbose: { questions: [] }
        }
      },
      steps: [],
      worker: null,
      usage: null,
      nextActions: null
    });

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    const runtimeError =
      decoded.value.explanation.lineage.compact.questions[0]?.runs[0]?.steps[0]
        ?.runtimeErrors[0];
    expect(runtimeError?.code).toBe("infrastructure_failed");
    expect(runtimeError?.message).toBe("provider_bad_request");
    expect(runtimeError?.retryable).toBe(false);
  });

  it("decodes structured next actions without CLI command text", () => {
    const decoded = decodeQuestionState({
      questionId: "q_clarify",
      conversationId: "conv_clarify",
      status: "NEEDS_CLARIFICATION",
      primaryRunId: "run_clarify",
      latestRunId: "run_clarify",
      activeRunId: null,
      question: "Show me performance.",
      answer: null,
      resultData: {
        kind: "needs_clarification",
        details: {
          clarifications: [
            {
              id: "clarification_1",
              need: "answer_metric",
              reason: "missing_answer_metric",
              question: "Which metric should I use?",
              requestedFactId: "fact_1",
              subjects: [
                {
                  kind: "metric_phrase",
                  id: "clarification_1",
                  label: "metric",
                  sourceText: "",
                  options: []
                }
              ],
              evidence: []
            }
          ]
        }
      },
      nextActions: [
        {
          kind: "provide_clarification",
          questionId: "q_clarify",
          conversationId: "conv_clarify",
          baseRunId: "run_clarify",
          clarificationId: "clarification_1",
          request: {
            method: "POST",
            path: "/questions/q_clarify/runs/",
            body: {
              question: "<clarification-answer>",
              triggerKind: "clarification_response",
              baseRunId: "run_clarify",
              clarificationId: "clarification_1"
            }
          }
        }
      ]
    });

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.nextActions[0]?.command).toBeNull();
    expect(decoded.value.nextActions[0]?.request?.path).toBe(
      "/questions/q_clarify/runs/"
    );
  });

  it("decodes completed run result data and explanation", () => {
    const decoded = decodeRun(completedRunFixture);

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.resultData?.kind).toBe("answer");
    expect(decoded.value.explanation.inputs.results[0]?.factDescription).toBe(
      "Count of in-person sales this month"
    );
    expect(decoded.value.worker).toBeNull();
    expect(decoded.value.usage).toBeNull();
  });

  it("decodes deterministic rerun identity and lineage", () => {
    const decoded = decodeRun({
      ...completedRunFixture,
      runId: "run_rerun",
      kind: "deterministic",
      triggerKind: "rerun",
      baseRunId: "run_sales",
      programId: "ap_sales",
      invocationId: "pi_rerun",
      patchId: "bp_rerun",
      revisionId: null
    });

    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value).toMatchObject({
      kind: "deterministic",
      triggerKind: "rerun",
      baseRunId: "run_sales",
      programId: "ap_sales",
      invocationId: "pi_rerun",
      patchId: "bp_rerun",
      revisionId: null
    });
  });

  it("rejects a deterministic run whose trigger is not rerun", () => {
    const decoded = decodeRun({
      ...completedRunFixture,
      kind: "deterministic",
      triggerKind: "initial"
    });

    expect(decoded).toEqual({
      ok: false,
      error: {
        message: "deterministic run requires triggerKind rerun"
      }
    });
  });

  it.each([
    ["baseRunId", { baseRunId: null }, "baseRunId must be a string"],
    ["programId", { programId: null }, "programId must be a string"],
    ["invocationId", { invocationId: null }, "invocationId must be a string"]
  ] as const)("rejects a deterministic rerun without %s", (_field, missing, message) => {
    const decoded = decodeRun({
      ...completedRunFixture,
      runId: "run_rerun",
      kind: "deterministic",
      triggerKind: "rerun",
      baseRunId: "run_sales",
      programId: "ap_sales",
      invocationId: "pi_rerun",
      ...missing
    });

    expect(decoded).toEqual({
      ok: false,
      error: { message }
    });
  });

  it("decodes queued live runs before explanation and steps exist", () => {
    const decoded = decodeQuestionRunList({
      questionId: "ee69a71a-2ee9-4609-b083-04dc5c5e4298",
      runs: [
        {
          answer: null,
          conversationId: "5794c10b-fe5c-4b52-bccd-f233f495c1b6",
          error: null,
          modelKey: "openai:gpt-5.4-mini",
          question: "list the stores",
          questionId: "ee69a71a-2ee9-4609-b083-04dc5c5e4298",
          resultData: null,
          runId: "2f118c8b-c973-47e0-b993-262a07164669",
          runNumber: 1,
          kind: "model_assisted",
          status: "QUEUED",
          tenantId: "default",
          triggerKind: "initial",
          baseRunId: null,
          programId: null,
          invocationId: null,
          patchId: null,
          revisionId: null
        }
      ]
    });

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.runs[0]?.steps).toEqual([]);
    expect(decoded.value.runs[0]?.explanation.inputs.results).toEqual([]);
  });

  it("treats absent optional diagnostics as null", () => {
    const { worker: _worker, usage: _usage, ...payload } = completedRunFixture;
    const decoded = decodeRun(payload);

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.worker).toBeNull();
    expect(decoded.value.usage).toBeNull();
  });

  it("treats null live next actions as no actions", () => {
    const decoded = decodeRun({
      ...completedRunFixture,
      nextActions: null
    });

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.nextActions).toEqual([]);
  });

  it("decodes canonical clarification result data with actionable ids", () => {
    const decoded = decodeQuestionRunList(runListFixture);

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    const firstRun = decoded.value.runs[0];
    expect(firstRun?.resultData?.kind).toBe("needs_clarification");
    if (firstRun?.resultData?.kind !== "needs_clarification") {
      throw new Error("expected clarification result data");
    }
    expect(firstRun.resultData.details.clarifications[0]?.id).toBe("clar_store");
    const options =
      firstRun.resultData.details.clarifications[0]?.subjects[0]?.options ?? [];
    expect(options).toHaveLength(2);
    expect(options[0]).toMatchObject({
      entityKind: "location",
      matchedLabel: "ABC Mall",
      matchedField: "location_id",
      matchedValue: "60606060-0000-0000-0001-000000000001",
      resolverReadId: "list_location_list",
      resolverLabel: "List Location List"
    });
  });

  it("decodes free-text clarification as the same contract with zero options", () => {
    const decoded = decodeRun(freeTextClarificationRunFixture);

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    if (decoded.value.resultData?.kind !== "needs_clarification") {
      throw new Error("expected clarification result data");
    }
    expect(decoded.value.resultData.details.clarifications[0]?.subjects[0]?.options).toEqual([]);
  });

  it("rejects a clarification terminal payload with no clarification objects", () => {
    const broken = {
      ...freeTextClarificationRunFixture,
      resultData: {
        kind: "needs_clarification",
        details: { clarifications: [] }
      }
    };

    const decoded = decodeRun(broken);

    expect(decoded.ok).toBe(false);
    if (decoded.ok) {
      throw new Error("expected decode failure");
    }
    expect(decoded.error.message).toContain("must include an actionable clarification");
  });

  it("rejects a clarification with an empty id", () => {
    const broken = {
      ...freeTextClarificationRunFixture,
      resultData: {
        kind: "needs_clarification",
        details: {
          clarifications: [
            {
              id: "",
              need: "question_interpretation",
              reason: "ambiguous_interpretation",
              question: "Which March should I use?",
              requestedFactId: "rf_sales_count",
              subjects: [
                {
                  kind: "interpretation",
                  id: "input_period",
                  label: "period",
                  sourceText: "March",
                  options: []
                }
              ],
              evidence: [{ kind: "question_contract", id: "ev_period" }]
            }
          ]
        }
      }
    };

    const decoded = decodeRun(broken);

    expect(decoded.ok).toBe(false);
    if (decoded.ok) {
      throw new Error("expected decode failure");
    }
    expect(decoded.error.message).toContain("clarification.id must not be empty");
  });

  it("preserves debug-oriented next action for failed runs", () => {
    const decoded = decodeRun(failedRunFixture);

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.status).toBe("FAILED");
    expect(decoded.value.nextActions[0]?.command).toContain("--debug");
  });

  it("decodes live failed run string errors", () => {
    const decoded = decodeRun({
      ...failedRunFixture,
      error: "planning_failed"
    });

    expect(decoded.ok).toBe(true);
    if (!decoded.ok) {
      throw new Error(decoded.error.message);
    }
    expect(decoded.value.error).toEqual({
      code: "planning_failed",
      message: "planning_failed",
      retryable: false
    });
  });
});
