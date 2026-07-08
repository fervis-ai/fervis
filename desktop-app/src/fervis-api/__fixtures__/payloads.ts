export const conversationListFixture = {
  conversations: [
    {
      conversationId: "conv_new",
      firstQuestion: "How many orders came in today?",
      latestQuestionId: "q_new",
      currentRunId: "run_new",
      status: "RUNNING",
      runCount: 1,
      updatedAt: "2026-06-27T10:15:00+00:00"
    },
    {
      conversationId: "conv_old",
      firstQuestion: "How many orders came in yesterday?",
      latestQuestionId: "q_old",
      currentRunId: "run_old",
      status: "COMPLETED",
      runCount: 2,
      updatedAt: "2026-06-26T09:00:00+00:00"
    }
  ]
} as const;

export const emptyStepSemanticFixture = {
  requestedFacts: [],
  knownInputs: [],
  resolverCandidates: [],
  groundingResults: [],
  interpretedInputs: [],
  conversationClauses: []
} as const;

const salesQuestionContractStepFixture = {
  stepId: "step_contract",
  stepKey: "question_contract",
  sequence: 0,
  decisions: [],
  sourceReads: [],
  runtimeErrors: [],
  semantic: {
    requestedFacts: [
      {
        requestedFactId: "rf_sales_count",
        description: "sales at ABC Mall this month"
      }
    ],
    knownInputs: [
      {
        inputId: "input_store",
        text: "ABC Mall",
        kind: "reference",
        description: "store or location",
        lookupText: "ABC Mall"
      },
      {
        inputId: "input_period",
        text: "this month",
        kind: "time",
        description: "reporting period",
        lookupText: "this month"
      }
    ],
    resolverCandidates: [],
    groundingResults: [],
    interpretedInputs: [],
    conversationClauses: []
  }
} as const;

const salesQueryEnrichmentStepFixture = {
  stepId: "step_query_enrichment",
  stepKey: "query_enrichment",
  sequence: 1,
  decisions: [],
  sourceReads: [],
  runtimeErrors: [],
  semantic: {
    requestedFacts: [],
    knownInputs: [],
    resolverCandidates: [
      {
        inputId: "input_store",
        resolverReadId: "list_location_list",
        resolverLabel: "List Location List",
        basis: "location can identify ABC Mall because target meaning is store or location."
      },
      {
        inputId: "input_store",
        resolverReadId: "list_store_list",
        resolverLabel: "List Store List",
        basis: "store can identify ABC Mall because target meaning is store or location."
      }
    ],
    groundingResults: [],
    interpretedInputs: [],
    conversationClauses: []
  }
} as const;

const salesGroundingStepFixture = {
  stepId: "step_grounding",
  stepKey: "grounding",
  sequence: 2,
  decisions: [],
  sourceReads: [],
  runtimeErrors: [],
  semantic: {
    requestedFacts: [],
    knownInputs: [],
    resolverCandidates: [
      {
        inputId: "input_store",
        resolverReadId: "list_location_list",
        resolverLabel: "List Location List",
        basis: "The resolver can search location records by lookup text and return a canonical location identity."
      }
    ],
    groundingResults: [
      {
        inputId: "input_store",
        inputText: "ABC Mall",
        resolverReadId: "list_location_list",
        resolverLabel: "List Location List",
        entityKind: "location",
        matchedField: "location_id",
        matchedValue: "60606060-0000-0000-0001-000000000001",
        matchedLabel: "ABC Mall"
      }
    ],
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
  }
} as const;

const salesSourceReadStepFixture = {
  stepId: "step_source",
  stepKey: "source_read",
  sequence: 3,
  decisions: [{ lines: ["Read sales filtered to current month."] }],
  semantic: emptyStepSemanticFixture,
  sourceReads: [
    {
      sourceReadId: "source_read_sales",
      method: "GET",
      path: "/api/sales/",
      rowCount: 18,
      status: "succeeded"
    }
  ],
  runtimeErrors: []
} as const;

export const explanationFixture = {
  inputs: {
    results: [
      {
        factResultId: "fr_sales_count",
        requestedFactId: "rf_sales_count",
        factDescription: "Count of in-person sales this month",
        explicit: ["in-person sales"],
        derived: ["count rows"],
        contextual: ["current month"],
        applied: ["filter channel = in_person"],
        evidenceRefs: ["ev_sales"],
        proofHandles: ["pg_sales", "node_count"]
      }
    ]
  },
  lineage: {
    compact: {
      questions: [
        {
          questionId: "q_sales",
          conversationId: "conv_sales",
          text: "How many in-person sales happened this month?",
          runs: [
            {
          runId: "run_sales",
          runNumber: 1,
          triggerKind: "initial",
          steps: [
                salesQuestionContractStepFixture,
                salesQueryEnrichmentStepFixture,
                salesGroundingStepFixture,
                salesSourceReadStepFixture
              ]
            }
          ]
        }
      ]
    },
    verbose: {
      questions: [
        {
          questionId: "q_sales",
          conversationId: "conv_sales",
          text: "How many in-person sales happened this month?",
          runs: [
            {
          runId: "run_sales",
          runNumber: 1,
          triggerKind: "initial",
          steps: [
                salesQuestionContractStepFixture,
                salesQueryEnrichmentStepFixture,
                salesGroundingStepFixture,
                salesSourceReadStepFixture
              ]
            }
          ]
        }
      ]
    }
  }
} as const;

export const completedRunFixture = {
  runId: "run_sales",
  questionId: "q_sales",
  conversationId: "conv_sales",
  runNumber: 1,
  triggerKind: "initial",
  status: "COMPLETED",
  answer: "18 in-person sales happened this month.",
  resultData: {
    kind: "answer",
    outputs: [{ key: "total_count", valueKind: "number", value: "18" }]
  },
  explanation: explanationFixture,
  steps: [
    {
      stepId: "step_source",
      stepKey: "source_read",
      sequence: 0,
      decisions: [{ lines: ["Read sales filtered to current month."] }],
      semantic: emptyStepSemanticFixture
    }
  ],
  error: null,
  worker: null,
  usage: null,
  nextActions: [
    {
      kind: "inspect_question",
      description: "Inspect the question and all runs attempted for it.",
      command: "fervis explain --question-id q_sales",
      request: null
    }
  ]
} as const;

export const runningRunFixture = {
  ...completedRunFixture,
  runId: "run_running",
  status: "RUNNING",
  answer: null,
  resultData: null,
  steps: [
    salesQuestionContractStepFixture,
    salesQueryEnrichmentStepFixture,
    salesGroundingStepFixture
  ],
  worker: {
    status: "RUNNING",
    attemptCount: 1,
    activeAttempt: 1,
    leaseOwner: "worker-local",
    leaseExpiresAt: "2026-06-27T10:16:00+00:00",
    lastError: null,
    createdAt: "2026-06-27T10:15:00+00:00",
    startedAt: "2026-06-27T10:15:02+00:00",
    completedAt: null
  },
  usage: {
    inputTokens: 1200,
    outputTokens: 240,
    thinkingTokens: 0,
    inputCostUsd: 0.0012,
    outputCostUsd: 0.0018,
    thinkingCostUsd: 0,
    costUsd: 0.003,
    costSource: "lineage_model_call_usage",
    pricingVersion: "models.dev:openai/gpt-5.4-mini",
    durationMs: 3500
  }
} as const;

export const clarificationRunFixture = {
  ...completedRunFixture,
  runId: "run_clarify",
  status: "NEEDS_CLARIFICATION",
  answer: null,
  resultData: {
    kind: "needs_clarification",
    details: {
      clarifications: [
        {
          id: "clar_store",
          need: "target_reference",
          reason: "multiple_matching_entities",
          question: "Which matching store should I use?",
          requestedFactId: "rf_sales_count",
          subjects: [
            {
              kind: "question_input",
              id: "input_store",
              label: "store",
              sourceText: "ABC Mall",
              options: [
                {
                  id: "location:location_id:60606060-0000-0000-0001-000000000001",
                  label: "ABC Mall",
                  value: "60606060-0000-0000-0001-000000000001",
                  entityKind: "location",
                  matchedLabel: "ABC Mall",
                  matchedField: "location_id",
                  matchedValue: "60606060-0000-0000-0001-000000000001",
                  resolverReadId: "list_location_list",
                  resolverLabel: "List Location List"
                },
                {
                  id: "store:store_id:70707070-0000-0000-0001-000000000002",
                  label: "BBS Outlet",
                  value: "70707070-0000-0000-0001-000000000002",
                  entityKind: "store",
                  matchedLabel: "BBS Outlet",
                  matchedField: "store_id",
                  matchedValue: "70707070-0000-0000-0001-000000000002",
                  resolverReadId: "list_store_list",
                  resolverLabel: "List Store List"
                }
              ]
            }
          ],
          evidence: [
            {
              kind: "known_input",
              id: "known_input:input_store",
              readId: null,
              endpointName: null,
              fieldId: null,
              identityField: null
            },
            {
              kind: "candidate",
              id: "location:location_id:60606060-0000-0000-0001-000000000001",
              readId: null,
              endpointName: null,
              fieldId: null,
              identityField: null
            },
            {
              kind: "candidate",
              id: "store:store_id:70707070-0000-0000-0001-000000000002",
              readId: null,
              endpointName: null,
              fieldId: null,
              identityField: null
            }
          ]
        }
      ]
    }
  },
  nextActions: [
    {
      kind: "provide_clarification",
      description: "Continue the same question by answering the clarification.",
      command:
        'fervis runtime ask "<answer>" --question-id q_sales --previous-run-id run_clarify --clarification-id clar_store',
      request: null
    }
  ]
} as const;

export const freeTextClarificationRunFixture = {
  ...clarificationRunFixture,
  resultData: {
    kind: "needs_clarification",
    details: {
      clarifications: [
        {
          id: "clar_period",
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
          evidence: [
            {
              kind: "question_contract",
              id: "ev_period",
              readId: null,
              endpointName: null,
              fieldId: null,
              identityField: null
            }
          ]
        }
      ]
    }
  }
} as const;

export const failedRunFixture = {
  ...completedRunFixture,
  runId: "run_failed",
  status: "FAILED",
  answer: null,
  resultData: null,
  error: {
    code: "provider_runtime_failed",
    message: "The provider request failed.",
    retryable: false
  },
  nextActions: [
    {
      kind: "inspect_question",
      description: "Inspect the question and all runs attempted for it.",
      command: "fervis explain --question-id q_sales --debug",
      request: null
    }
  ]
} as const;

export const questionStateFixture = {
  questionId: "q_sales",
  conversationId: "conv_sales",
  question: "How many in-person sales happened this month?",
  currentRunId: "run_sales",
  status: "COMPLETED",
  answer: "18 in-person sales happened this month.",
  resultData: completedRunFixture.resultData,
  nextActions: completedRunFixture.nextActions
} as const;

export const runListFixture = {
  questionId: "q_sales",
  runs: [clarificationRunFixture, completedRunFixture]
} as const;

export const demoRunsFixture = [
  clarificationRunFixture,
  completedRunFixture,
  runningRunFixture,
  freeTextClarificationRunFixture,
  failedRunFixture
] as const;
