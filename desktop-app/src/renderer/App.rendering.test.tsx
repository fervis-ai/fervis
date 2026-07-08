import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import {
  failMissingEvidenceLabel,
  httpPayloadFor,
  renderDemoApp
} from "./appTestSupport";
import { saveConnectionSettings } from "./connectionSettings";

describe("Ledger app rendering", () => {
  afterEach(() => {
    window.localStorage.clear();
    vi.unstubAllGlobals();
  });

  it("starts with a connection gate without fixture conversations", () => {
    render(<App initialClient={null} />);

    expect(screen.getByText("Connect to Fervis")).toBeInTheDocument();
    expect(
      screen.queryByText("How many in-person sales happened this month?")
    ).not.toBeInTheDocument();
  });

  it("renders conversation labels from first questions", async () => {
    renderDemoApp();

    const matchingLabels = await screen.findAllByText(
      "How many in-person sales happened this month?"
    );
    expect(matchingLabels[0]).toBeInTheDocument();
    expect(
      screen.getByText("Which store has the most inventory at risk today?")
    ).toBeInTheDocument();
  });

  it("renders completed answer, inputs, and explanation proof", async () => {
    renderDemoApp();

    expect(
      await screen.findByText(
        "18 in-person sales happened this month.",
        {},
        { timeout: 2500 }
      )
    ).toBeInTheDocument();
    expect(screen.getAllByText(/sales at ABC Mall this month/i)[0]).toBeInTheDocument();
    expect(screen.getByText("Total Count")).toBeInTheDocument();
    expect(screen.getByText("Fact used")).toBeInTheDocument();
    expect(screen.getByText("Read source data")).toBeInTheDocument();
    expect(screen.getByText(/GET \/api\/sales\/ returned 18 rows/)).toBeInTheDocument();

    fireEvent.click(screen.getByText("More"));
    expect(screen.getByText("Decision")).toBeInTheDocument();
  });

  it("lets users expand lower-signal explanation details", async () => {
    renderDemoApp();

    await screen.findByText("18 in-person sales happened this month.");
    fireEvent.click(screen.getByText("More"));

    expect(screen.queryByLabelText("Show lower-signal details")).not.toBeInTheDocument();
  });

  it("opens completed-answer evidence at the top level", async () => {
    renderDemoApp();

    await screen.findByText("18 in-person sales happened this month.");

    const evidence = screen.getAllByText("Evidence")[0]?.closest("details");
    expect(evidence).toHaveAttribute("open");
  });

  it("renders choice and text clarification states", async () => {
    renderDemoApp();

    fireEvent.click(await screen.findByRole("button", { name: /run_clarify/ }));
    expect(screen.getByText("Which store do you mean?")).toBeInTheDocument();
    expect(screen.getByText("ABC Mall")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /What were sales for BBS last month?/ })
    );

    expect(await screen.findByText("Which March should I use?")).toBeInTheDocument();
    expect(screen.getByLabelText("Clarification answer")).toBeInTheDocument();
  });

  it("keeps only one run body expanded as the focus area", async () => {
    renderDemoApp();

    expect(
      await screen.findByText("18 in-person sales happened this month.")
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /run_clarify/ }));

    expect(await screen.findByText("Which store do you mean?")).toBeInTheDocument();
    expect(
      screen.queryByText("18 in-person sales happened this month.")
    ).not.toBeInTheDocument();
  });

  it("renders running and failed states without fake progress", async () => {
    renderDemoApp();

    fireEvent.click(
      await screen.findByText("Which store has the most inventory at risk today?")
    );
    expect(
      await screen.findByText(
        "Interpreted input: this month: 2026-06-01 to 2026-06-30"
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Matched entity: ABC Mall: List Location List matched location_id=60606060-0000-0000-0001-000000000001"
      )
    ).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("/ 3")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", {
        name: /Which returns endpoint failed during settlement review?/
      })
    );
    expect(await screen.findByText(/provider_runtime_failed/)).toBeInTheDocument();
    expect(
      screen.getByText("fervis explain --question-id q_failed --debug")
    ).toBeInTheDocument();
  });

  it("opens settings and cycles theme from the top-right control", async () => {
    renderDemoApp();
    await screen.findAllByText("How many in-person sales happened this month?");

    fireEvent.click(screen.getByLabelText("Open connection settings"));
    expect(screen.getByRole("dialog", { name: "Connection settings" })).toBeInTheDocument();

    fireEvent.click(screen.getByText("Cancel"));
    expect(
      screen.queryByRole("dialog", { name: "Connection settings" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Switch theme from system"));
    expect(screen.getByLabelText("Switch theme from light")).toBeInTheDocument();
  });

  it("saves connection settings with a masked session token", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(httpPayloadFor(input.toString())), {
        headers: { "Content-Type": "application/json" },
        status: 200
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App initialClient={null} />);

    fireEvent.click(screen.getByLabelText("Open connection settings"));
    expect(screen.getByLabelText("Base API URL")).toHaveValue(
      "http://127.0.0.1:8000/fervis"
    );
    fireEvent.change(screen.getByLabelText("Base API URL"), {
      target: { value: "http://127.0.0.1:9000/fervis" }
    });
    fireEvent.change(screen.getByLabelText("Auth token"), {
      target: { value: "secret-token" }
    });

    const tokenInput = screen.getByLabelText("Auth token");
    expect(tokenInput).toHaveAttribute("type", "password");
    fireEvent.click(screen.getByText("Save"));

    expect(await screen.findByText("connected")).toBeInTheDocument();
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "http://127.0.0.1:9000/fervis/conversations/"
    );
    const firstRequest = fetchMock.mock.calls[0]?.[1];
    if (firstRequest === undefined) {
      throw new Error("expected fetch request init");
    }
    expect(new Headers(firstRequest.headers).get("Authorization")).toBe(
      "Bearer secret-token"
    );

    fireEvent.click(screen.getByLabelText("Open connection settings"));
    expect(screen.getByLabelText("Base API URL")).toHaveValue(
      "http://127.0.0.1:9000/fervis"
    );
    expect(screen.getByLabelText("Auth token")).toHaveValue("");
  });

  it("prefills settings with the cached API URL", () => {
    saveConnectionSettings({ baseUrl: "http://127.0.0.1:9100/v1/" });

    render(<App initialClient={null} />);
    fireEvent.click(screen.getByLabelText("Open connection settings"));

    expect(screen.getByLabelText("Base API URL")).toHaveValue(
      "http://127.0.0.1:9100/v1/"
    );
    expect(screen.getByLabelText("Auth token")).toHaveValue("");
  });

  it("keeps typed follow-up text when the theme changes", async () => {
    renderDemoApp();

    const input = await screen.findByLabelText("Ask a follow-up question");
    fireEvent.change(input, {
      target: { value: "What about yesterday?" }
    });

    fireEvent.click(screen.getByLabelText("Switch theme from system"));

    expect(screen.getByLabelText("Switch theme from light")).toBeInTheDocument();
    expect(screen.getByLabelText("Ask a follow-up question")).toHaveValue(
      "What about yesterday?"
    );
  });
});
