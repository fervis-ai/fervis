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
          basis: "ambiguous_store",
          question: "Which store do you mean?",
          requestedFactId: "rf_sales_count",
          knownInputId: "input_store",
          availableOptions: [
            { id: "store_mall", label: "ABC Mall" },
            { id: "store_outlet", label: "BBS Outlet" }
          ],
          evidenceRefs: ["ev_store_candidates"],
          factResultId: "fr_store",
          stepId: "step_clarify"
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
          basis: "ambiguous_period",
          question: "Which March should I use?",
          requestedFactId: "rf_sales_count",
          knownInputId: "input_period",
          availableOptions: [],
          evidenceRefs: ["ev_period"],
          factResultId: null,
          stepId: "step_clarify"
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
