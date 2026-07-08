export type RunStatus =
  | "QUEUED"
  | "RUNNING"
  | "COMPLETED"
  | "NEEDS_CLARIFICATION"
  | "FAILED";

export interface ConversationSummary {
  readonly conversationId: string;
  readonly firstQuestion: string;
  readonly latestQuestionId: string;
  readonly currentRunId: string | null;
  readonly status: RunStatus;
  readonly runCount: number;
  readonly updatedAt: string;
}

export interface ConversationListPayload {
  readonly conversations: readonly ConversationSummary[];
}

export interface NextAction {
  readonly kind: "inspect_question" | "provide_clarification" | "retry";
  readonly description: string | null;
  readonly command: string | null;
  readonly request: NextActionRequest | null;
}

export interface NextActionRequest {
  readonly method: string;
  readonly path: string;
}

export interface QuestionStatePayload {
  readonly questionId: string;
  readonly conversationId: string;
  readonly question: string;
  readonly currentRunId: string;
  readonly status: RunStatus;
  readonly answer: string | null;
  readonly resultData: ResultData;
  readonly nextActions: readonly NextAction[];
}

export interface QuestionRunListPayload {
  readonly questionId: string;
  readonly runs: readonly RunPayload[];
}

export interface AskQuestionRequest {
  readonly question: string;
  readonly conversationId: string | null;
}

export interface ClarificationResponseRequest {
  readonly question: string;
  readonly triggerKind: "clarification_response";
  readonly triggerRunId: string;
  readonly clarificationId: string;
  readonly selectedOptionId?: string;
}

export interface RunPayload {
  readonly runId: string;
  readonly questionId: string;
  readonly conversationId: string;
  readonly runNumber: number;
  readonly triggerKind: "initial" | "clarification_response";
  readonly status: RunStatus;
  readonly answer: string | null;
  readonly resultData: ResultData;
  readonly explanation: ExplanationPayload;
  readonly steps: readonly RunStep[];
  readonly error: RunError | null;
  readonly worker: WorkerSnapshot | null;
  readonly usage: UsageSnapshot | null;
  readonly nextActions: readonly NextAction[];
}

export type ResultData = AnswerResultData | ClarificationResultData | null;

export interface AnswerResultData {
  readonly kind: "answer";
  readonly outputs: readonly AnswerOutput[];
}

export interface AnswerOutput {
  readonly key: string;
  readonly valueKind: AnswerValueKind;
  readonly value: string;
}

export type AnswerValueKind =
  | "entity"
  | "number"
  | "money"
  | "boolean"
  | "text"
  | "date"
  | "datetime"
  | "table"
  | "list"
  | "object";

export interface ClarificationResultData {
  readonly kind: "needs_clarification";
  readonly details: ClarificationDetails;
}

export interface ClarificationDetails {
  readonly clarifications: readonly ClarificationRequest[];
}

export interface ClarificationRequest {
  readonly id: string;
  readonly need: string;
  readonly reason: string;
  readonly question: string;
  readonly requestedFactId: string;
  readonly subjects: readonly ClarificationSubject[];
  readonly evidence: readonly ClarificationEvidence[];
}

export interface ClarificationSubject {
  readonly kind: string;
  readonly id: string;
  readonly label: string;
  readonly sourceText: string;
  readonly options: readonly ClarificationOption[];
}

export interface ClarificationOption {
  readonly id: string;
  readonly label: string;
  readonly value: string | null;
  readonly entityKind: string | null;
  readonly matchedLabel: string | null;
  readonly matchedField: string | null;
  readonly matchedValue: string | null;
  readonly resolverReadId: string | null;
  readonly resolverLabel: string | null;
}

export interface ClarificationEvidence {
  readonly kind: string;
  readonly id: string;
  readonly readId: string | null;
  readonly endpointName: string | null;
  readonly fieldId: string | null;
  readonly identityField: string | null;
}

export interface ExplanationPayload {
  readonly inputs: InputExplanation;
  readonly lineage: LineageExplanation;
}

export interface InputExplanation {
  readonly results: readonly InputResult[];
}

export interface InputResult {
  readonly factResultId: string;
  readonly requestedFactId: string;
  readonly factDescription: string;
  readonly explicit: readonly string[];
  readonly derived: readonly string[];
  readonly contextual: readonly string[];
  readonly applied: readonly string[];
  readonly evidenceRefs: readonly string[];
  readonly proofHandles: readonly string[];
}

export interface LineageExplanation {
  readonly compact: LineageTimeline;
  readonly verbose: LineageTimeline;
}

export interface LineageTimeline {
  readonly questions: readonly LineageQuestion[];
}

export interface LineageQuestion {
  readonly questionId: string;
  readonly conversationId: string;
  readonly text: string;
  readonly runs: readonly LineageRun[];
}

export interface LineageRun {
  readonly runId: string;
  readonly runNumber: number;
  readonly triggerKind: string;
  readonly steps: readonly LineageStep[];
}

export interface LineageStep {
  readonly stepId: string;
  readonly stepKey: string;
  readonly sequence: number;
  readonly decisions: readonly StepDecision[];
  readonly semantic: StepSemantic;
  readonly sourceReads: readonly SourceRead[];
  readonly runtimeErrors: readonly RunError[];
}

export interface RunStep {
  readonly stepId: string;
  readonly stepKey: string;
  readonly decisions: readonly StepDecision[];
  readonly semantic: StepSemantic;
}

export interface StepSemantic {
  readonly requestedFacts: readonly SemanticRequestedFact[];
  readonly knownInputs: readonly SemanticKnownInput[];
  readonly resolverCandidates: readonly SemanticResolverCandidate[];
  readonly groundingResults: readonly SemanticGroundingResult[];
  readonly interpretedInputs: readonly SemanticInterpretedInput[];
  readonly conversationClauses: readonly SemanticConversationClause[];
}

export interface SemanticRequestedFact {
  readonly requestedFactId: string;
  readonly description: string;
}

export interface SemanticKnownInput {
  readonly inputId: string;
  readonly text: string;
  readonly kind: string;
  readonly description: string;
  readonly lookupText: string;
}

export interface SemanticResolverCandidate {
  readonly inputId: string;
  readonly resolverReadId: string;
  readonly resolverLabel: string;
  readonly basis: string;
}

export interface SemanticGroundingResult {
  readonly inputId: string;
  readonly inputText: string;
  readonly resolverReadId: string;
  readonly resolverLabel: string;
  readonly entityKind: string;
  readonly matchedField: string;
  readonly matchedValue: string;
  readonly matchedLabel: string;
}

export interface SemanticInterpretedInput {
  readonly inputId: string;
  readonly inputText: string;
  readonly kind: string;
  readonly value: string;
  readonly label: string;
  readonly detail: string;
}

export interface SemanticConversationClause {
  readonly currentClauseText: string;
  readonly currentValueText: string;
  readonly resolvedFrameText: string;
  readonly resolvedClauseText: string;
}

export interface StepDecision {
  readonly lines: readonly string[];
}

export interface SourceRead {
  readonly sourceReadId: string;
  readonly method: string;
  readonly path: string;
  readonly rowCount: number;
  readonly status: string;
}

export interface RunError {
  readonly code: string;
  readonly message: string;
  readonly retryable: boolean;
}

export interface WorkerSnapshot {
  readonly status: string;
  readonly attemptCount: number;
  readonly activeAttempt: number;
  readonly leaseOwner: string | null;
  readonly leaseExpiresAt: string | null;
  readonly lastError: string | null;
  readonly createdAt: string;
  readonly startedAt: string | null;
  readonly completedAt: string | null;
}

export interface UsageSnapshot {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly thinkingTokens: number;
  readonly inputCostUsd: number;
  readonly outputCostUsd: number;
  readonly thinkingCostUsd: number;
  readonly costUsd: number;
  readonly costSource: string;
  readonly pricingVersion: string;
  readonly durationMs: number;
}

export interface DecodeFailure {
  readonly message: string;
}

export type DecodeResult<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: DecodeFailure };
