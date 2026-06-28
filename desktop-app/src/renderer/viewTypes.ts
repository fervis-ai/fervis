import type {
  ConversationSummary,
  QuestionStatePayload,
  RunPayload
} from "../fervis-api/contracts";

export type ThemeMode = "system" | "light" | "dark";

export type QuestionRefreshPayload = Pick<
  QuestionStatePayload,
  | "questionId"
  | "conversationId"
  | "question"
  | "currentRunId"
  | "status"
>;

export type NonEmptyRuns = readonly [RunPayload, ...RunPayload[]];

export interface ConversationDetails {
  readonly summary: ConversationSummary;
  readonly question: string;
  readonly runs: NonEmptyRuns;
}
