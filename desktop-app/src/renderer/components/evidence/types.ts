export type ProofMode = "compact" | "verbose";

export interface EvidenceInsight {
  readonly label: string;
  readonly value: string;
  readonly detail: string | null;
}

export interface ProofNote {
  readonly label: string;
  readonly text: string;
}
