import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { completedRunFixture } from "../../fervis-api/__fixtures__/payloads";
import type { FervisApiClient } from "../../fervis-api/client";
import { createDemoFervisClient } from "../demoClient";
import { EvidencePanel } from "./EvidencePanel";

const recorder = vi.hoisted(() => ({
  cancel: vi.fn(),
  start: vi.fn(),
  stop: vi.fn()
}));

vi.mock("../audioQuestionRecorder", () => ({
  startAudioQuestionRecording: recorder.start
}));

describe("Ask about answer hold interaction", () => {
  const questionAudio = new Blob(["RIFFquestion"], { type: "audio/wav" });
  const createObjectURL = vi.fn(() => "blob:answer");
  const revokeObjectURL = vi.fn();
  const play = vi.fn(async () => undefined);
  const pause = vi.fn();

  beforeEach(() => {
    recorder.cancel.mockReset();
    recorder.stop.mockReset().mockResolvedValue(questionAudio);
    recorder.start.mockReset().mockResolvedValue({
      cancel: recorder.cancel,
      stop: recorder.stop
    });
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
    play.mockClear();
    pause.mockClear();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: createObjectURL
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: revokeObjectURL
    });
    vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(play);
    vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(pause);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("records while held, submits on release, plays immediately, and disposes", async () => {
    const askAboutAnswer = vi.fn(async () => new Blob(["RIFFanswer"], { type: "audio/wav" }));
    const { container } = renderEvidence(askAboutAnswer);
    const control = screen.getByRole("button", { name: "Hold to ask about this answer" });

    fireEvent.pointerDown(control, { pointerId: 1 });
    expect(await screen.findByRole("button", { name: "Release to ask about this answer" })).toHaveTextContent(
      "Release to ask"
    );
    fireEvent.pointerUp(
      screen.getByRole("button", { name: "Release to ask about this answer" }),
      { pointerId: 1 }
    );

    expect(await screen.findByRole("button", { name: "Stop spoken explanation" })).toHaveTextContent(
      "Stop"
    );
    expect(askAboutAnswer).toHaveBeenCalledWith(
      completedRunFixture.questionId,
      completedRunFixture.runId,
      questionAudio,
      { signal: expect.any(AbortSignal) }
    );
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(play).toHaveBeenCalledTimes(1);
    expect(container.querySelector(".evidence-panel")).toHaveClass("answering");
    expect(control.querySelector(".fervis-speaking-mark .fervis-mark")).toBeInstanceOf(
      SVGElement
    );
    expect(control.querySelector(".playing-bars")).not.toBeInTheDocument();

    const audio = container.querySelector("audio");
    if (audio === null) {
      throw new Error("expected disposable audio element");
    }
    fireEvent.ended(audio);

    expect(await screen.findByRole("button", { name: "Hold to ask about this answer" })).toHaveTextContent(
      "Ask"
    );
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:answer");
    expect(container.querySelector(".evidence-panel")).not.toHaveClass("answering");
  });

  it("shows the Fervis thinking state and lets the user cancel it", async () => {
    let signal: AbortSignal | undefined;
    const askAboutAnswer = vi.fn(
      async (_questionId, _runId, _question, options) =>
        new Promise<Blob>((_resolve, reject) => {
          signal = options?.signal;
          signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        })
    );
    renderEvidence(askAboutAnswer);

    const control = screen.getByRole("button", { name: "Hold to ask about this answer" });
    fireEvent.pointerDown(control);
    fireEvent.pointerUp(
      await screen.findByRole("button", { name: "Release to ask about this answer" })
    );
    const thinking = await screen.findByRole("button", { name: "Cancel explanation question" });
    expect(thinking).toHaveTextContent("Thinking");
    expect(thinking.querySelector(".fervis-thinking-mark .fervis-mark")).toBeInstanceOf(
      SVGElement
    );

    fireEvent.pointerDown(thinking);

    await waitFor(() => expect(signal?.aborted).toBe(true));
    expect(screen.getByRole("button", { name: "Hold to ask about this answer" })).toBeInTheDocument();
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("keeps microphone denial local to Evidence", async () => {
    recorder.start.mockRejectedValueOnce(new DOMException("Denied", "NotAllowedError"));
    renderEvidence(vi.fn());

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Hold to ask about this answer" })
    );

    expect(await screen.findByRole("status")).toHaveTextContent(
      "Microphone access is needed to ask about this answer."
    );
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hold to ask about this answer" })).toBeInTheDocument();
  });

  it("stops microphone capture when the run leaves the screen", async () => {
    const rendered = renderEvidence(vi.fn());
    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Hold to ask about this answer" })
    );
    await screen.findByRole("button", { name: "Release to ask about this answer" });

    rendered.unmount();

    expect(recorder.cancel).toHaveBeenCalledTimes(1);
  });

  it("discards microphone permission that resolves after switching runs", async () => {
    let resolveRecorder: ((value: { cancel: () => void; stop: () => Promise<Blob> }) => void) | undefined;
    recorder.start.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveRecorder = resolve;
      })
    );
    const askAboutAnswer = vi.fn();
    const apiClient = { ...createDemoFervisClient(), askAboutAnswer };
    const rendered = render(
      <EvidencePanel apiClient={apiClient} defaultOpen run={completedRunFixture} />
    );
    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Hold to ask about this answer" })
    );
    await screen.findByRole("button", {
      name: "Release to send when the microphone opens"
    });

    rendered.rerender(
      <EvidencePanel
        apiClient={apiClient}
        defaultOpen
        run={{ ...completedRunFixture, runId: "run-next" }}
      />
    );
    resolveRecorder?.({ cancel: recorder.cancel, stop: recorder.stop });

    await waitFor(() => expect(recorder.cancel).toHaveBeenCalledTimes(1));
    expect(askAboutAnswer).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Hold to ask about this answer" })
    ).toHaveTextContent("Ask");
  });
});

function renderEvidence(askAboutAnswer: FervisApiClient["askAboutAnswer"]) {
  return render(
    <EvidencePanel
      apiClient={{ ...createDemoFervisClient(), askAboutAnswer }}
      defaultOpen
      run={completedRunFixture}
    />
  );
}
