import { afterEach, describe, expect, it, vi } from "vitest";

import { completedRunFixture, conversationListFixture } from "./__fixtures__/payloads";
import {
  createFervisHttpClient,
  type FervisApiError
} from "./client";

describe("Fervis HTTP API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests Fervis endpoints with the configured base URL and bearer token", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(conversationListFixture), {
        headers: { "Content-Type": "application/json" },
        status: 200
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = createFervisHttpClient({
      authToken: "token-123",
      baseUrl: "http://127.0.0.1:8000/fervis/"
    });

    const payload = await client.listConversations();

    expect(payload.conversations[0]?.conversationId).toBe("conv_new");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const firstCall = fetchMock.mock.calls[0];
    if (firstCall === undefined || firstCall[1] === undefined) {
      throw new Error("expected fetch call with request init");
    }
    expect(firstCall[0]).toBe("http://127.0.0.1:8000/fervis/conversations/");
    const init = firstCall[1];
    expect(new Headers(init.headers).get("Authorization")).toBe(
      "Bearer token-123"
    );
  });

  it("turns non-JSON failed responses into a stable Fervis API error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("upstream unavailable", { status: 502 }))
    );
    const client = createFervisHttpClient({
      authToken: "",
      baseUrl: "http://127.0.0.1:8000/fervis"
    });

    await expect(client.listConversations()).rejects.toMatchObject({
      message: "Fervis request failed",
      name: "FervisApiError",
      status: 502
    } satisfies Partial<FervisApiError>);
  });

  it("requests one run by question and run id for progress polling", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(completedRunFixture), {
        headers: { "Content-Type": "application/json" },
        status: 200
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = createFervisHttpClient({
      authToken: "",
      baseUrl: "http://127.0.0.1:8000/fervis"
    });

    const run = await client.getRun("q/with slash", "run:poll");

    expect(run.runId).toBe("run_sales");
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "http://127.0.0.1:8000/fervis/questions/q%2Fwith%20slash/runs/run%3Apoll/"
    );
  });
});
