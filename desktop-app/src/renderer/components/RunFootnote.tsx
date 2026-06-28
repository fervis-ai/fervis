import type { RunPayload } from "../../fervis-api/contracts";
import { pollableStatus } from "../runView";

const POLL_INTERVAL_MS = 1200;

export function RunFootnote({
  pollingErrorMessage,
  run
}: {
  readonly pollingErrorMessage: string | null;
  readonly run: RunPayload;
}) {
  const parts = [
    `latest run ${run.runNumber}`,
    run.runId,
    `terminal ${run.status}`,
    `trigger ${run.triggerKind}`
  ];

  if (run.worker !== null) {
    if (run.worker.leaseOwner !== null) {
      parts.push(`worker ${run.worker.leaseOwner}`);
    }
    parts.push(`attempt ${run.worker.activeAttempt}`);
  }

  if (run.usage !== null) {
    parts.push(`cost $${run.usage.costUsd.toFixed(3)}`);
  }

  if (pollableStatus(run.status)) {
    parts.unshift(`polling every ${(POLL_INTERVAL_MS / 1000).toFixed(1)}s`);
  }

  if (pollingErrorMessage !== null && pollableStatus(run.status)) {
    parts.unshift(`polling paused · ${pollingErrorMessage}`);
  }

  return (
    <footer className="footnote">
      {pollingErrorMessage !== null && pollableStatus(run.status)
        ? parts.join(" · ")
        : `latest run ${run.runNumber} · ${run.status}`}
    </footer>
  );
}
