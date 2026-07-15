import { useEffect, useRef, useState } from "react";

import type { FervisApiClient } from "../../fervis-api/client";
import {
  startAudioQuestionRecording,
  type AudioQuestionRecording
} from "../audioQuestionRecorder";
import { FervisMark } from "./FervisMark";

type AskState =
  | "idle"
  | "requesting_permission"
  | "recording"
  | "thinking"
  | "speaking";

export function AskAboutAnswerControl({
  apiClient,
  questionId,
  runId,
  onPlaybackChange
}: {
  readonly apiClient: FervisApiClient;
  readonly questionId: string;
  readonly runId: string;
  readonly onPlaybackChange: (playing: boolean) => void;
}) {
  const [state, setState] = useState<AskState>("idle");
  const [error, setError] = useState<string | null>(null);
  const stateRef = useRef<AskState>("idle");
  const audioRef = useRef<HTMLAudioElement>(null);
  const recordingRef = useRef<AudioQuestionRecording | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const releasePendingRef = useRef(false);
  const aliveRef = useRef(true);
  const attemptRef = useRef(0);

  const transition = (next: AskState): void => {
    stateRef.current = next;
    setState(next);
  };

  const releasePlayback = (): void => {
    const audio = audioRef.current;
    if (audio !== null) {
      audio.pause();
      audio.removeAttribute("src");
    }
    if (objectUrlRef.current !== null) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    onPlaybackChange(false);
  };

  const cancel = (): void => {
    attemptRef.current += 1;
    releasePendingRef.current = false;
    recordingRef.current?.cancel();
    recordingRef.current = null;
    abortRef.current?.abort();
    abortRef.current = null;
    releasePlayback();
    transition("idle");
  };

  useEffect(() => {
    aliveRef.current = true;
    stateRef.current = "idle";
    setState("idle");
    return () => {
      aliveRef.current = false;
      attemptRef.current += 1;
      stateRef.current = "idle";
      recordingRef.current?.cancel();
      abortRef.current?.abort();
      releasePlayback();
    };
  }, [questionId, runId]);

  const submit = async (recording: AudioQuestionRecording): Promise<void> => {
    recordingRef.current = null;
    transition("thinking");
    const abort = new AbortController();
    abortRef.current = abort;
    try {
      const question = await recording.stop();
      if (abort.signal.aborted) {
        return;
      }
      const answer = await apiClient.askAboutAnswer(questionId, runId, question, {
        signal: abort.signal
      });
      if (abort.signal.aborted || !aliveRef.current) {
        return;
      }
      const objectUrl = URL.createObjectURL(answer);
      objectUrlRef.current = objectUrl;
      const audio = audioRef.current;
      if (audio === null) {
        throw new Error("Audio playback is unavailable.");
      }
      audio.src = objectUrl;
      transition("speaking");
      onPlaybackChange(true);
      await audio.play();
    } catch (caught) {
      if (!abort.signal.aborted && aliveRef.current) {
        releasePlayback();
        transition("idle");
        setError(messageFrom(caught));
      }
    } finally {
      if (abortRef.current === abort) {
        abortRef.current = null;
      }
    }
  };

  const begin = async (): Promise<void> => {
    if (stateRef.current !== "idle") {
      cancel();
      return;
    }
    setError(null);
    releasePendingRef.current = false;
    const attempt = attemptRef.current + 1;
    attemptRef.current = attempt;
    transition("requesting_permission");
    try {
      const recording = await startAudioQuestionRecording();
      if (
        !aliveRef.current ||
        attemptRef.current !== attempt ||
        stateRef.current === "idle"
      ) {
        recording.cancel();
        return;
      }
      recordingRef.current = recording;
      if (releasePendingRef.current) {
        releasePendingRef.current = false;
        await submit(recording);
      } else {
        transition("recording");
      }
    } catch (caught) {
      if (aliveRef.current) {
        transition("idle");
        setError(permissionMessage(caught));
      }
    }
  };

  const release = (): void => {
    if (stateRef.current === "requesting_permission") {
      releasePendingRef.current = true;
      return;
    }
    if (stateRef.current === "recording" && recordingRef.current !== null) {
      void submit(recordingRef.current);
    }
  };

  return (
    <span className="ask-answer-control">
      <button
        aria-label={buttonLabel(state)}
        className="ask-answer-button"
        data-state={state}
        type="button"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
        }}
        onKeyDown={(event) => {
          if ((event.key === " " || event.key === "Enter") && !event.repeat) {
            event.preventDefault();
            event.stopPropagation();
            void begin();
          }
        }}
        onKeyUp={(event) => {
          if (event.key === " " || event.key === "Enter") {
            event.preventDefault();
            event.stopPropagation();
            release();
          }
        }}
        onPointerCancel={(event) => {
          event.preventDefault();
          event.stopPropagation();
          cancel();
        }}
        onPointerDown={(event) => {
          event.preventDefault();
          event.stopPropagation();
          event.currentTarget.setPointerCapture?.(event.pointerId);
          void begin();
        }}
        onPointerUp={(event) => {
          event.preventDefault();
          event.stopPropagation();
          release();
        }}
      >
        <StateMark state={state} />
        <span>{stateText(state)}</span>
      </button>
      {error === null ? null : (
        <span className="ask-answer-error" role="status">
          {error}
        </span>
      )}
      <audio
        aria-hidden="true"
        ref={audioRef}
        onEnded={() => {
          releasePlayback();
          transition("idle");
        }}
      />
    </span>
  );
}

function StateMark({ state }: { readonly state: AskState }) {
  if (state === "thinking") {
    return (
      <span aria-hidden="true" className="fervis-thinking-mark">
        <FervisMark />
      </span>
    );
  }
  if (state === "recording") {
    return <span aria-hidden="true" className="recording-pulse" />;
  }
  if (state === "speaking") {
    return (
      <span aria-hidden="true" className="fervis-speaking-mark">
        <FervisMark />
      </span>
    );
  }
  return <MicrophoneIcon />;
}

function MicrophoneIcon() {
  return (
    <svg aria-hidden="true" className="microphone-icon" viewBox="0 0 18 18">
      <path d="M6 1h6v2h2v7h-2v2h-2v2h4v3H4v-3h4v-2H6v-2H4V3h2zm1 2v7h4V3z" />
    </svg>
  );
}

function stateText(state: AskState): string {
  if (state === "requesting_permission") {
    return "Opening mic";
  }
  if (state === "recording") {
    return "Release to ask";
  }
  if (state === "thinking") {
    return "Thinking";
  }
  if (state === "speaking") {
    return "Stop";
  }
  return "Ask";
}

function buttonLabel(state: AskState): string {
  if (state === "requesting_permission") {
    return "Release to send when the microphone opens";
  }
  if (state === "recording") {
    return "Release to ask about this answer";
  }
  if (state === "thinking") {
    return "Cancel explanation question";
  }
  if (state === "speaking") {
    return "Stop spoken explanation";
  }
  return "Hold to ask about this answer";
}

function permissionMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === "NotAllowedError") {
    return "Microphone access is needed to ask about this answer.";
  }
  return messageFrom(error);
}

function messageFrom(error: unknown): string {
  return error instanceof Error ? error.message : "Ask failed.";
}
