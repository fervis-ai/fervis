import { afterEach, describe, expect, it, vi } from "vitest";

import {
  completedRunFixture,
  conversationListFixture,
  questionStateFixture
} from "./__fixtures__/payloads";
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

  it("requests disposable explanation audio with the established connection authority", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response("RIFFaudio", {
          headers: { "Content-Type": "audio/wav" },
          status: 200
        })
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = createFervisHttpClient({
      authToken: "token-123",
      baseUrl: "http://127.0.0.1:8000/fervis/"
    });

    const question = new Blob(["RIFFquestion"], { type: "audio/wav" });
    const result = await client.askAboutAnswer("q/one", "run:one", question);

    expect(result.type).toBe("audio/wav");
    expect(result.size).toBe(9);
    const request = fetchMock.mock.calls[0];
    expect(request?.[0]).toBe(
      "http://127.0.0.1:8000/fervis/questions/q%2Fone/runs/run%3Aone/ask/"
    );
    expect(request?.[1]?.method).toBe("POST");
    const headers = new Headers(request?.[1]?.headers);
    expect(headers.get("Accept")).toBe("audio/wav");
    expect(headers.get("Content-Type")).toBe("audio/wav");
    expect(headers.get("Authorization")).toBe("Bearer token-123");
    expect(headers.has("Idempotency-Key")).toBe(false);
    expect(request?.[1]?.body).toBe(question);
  });

  it("surfaces Ask API errors without treating JSON as audio", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              error: { message: "Explanation audio is not configured." }
            }),
            {
              headers: { "Content-Type": "application/json" },
              status: 503
            }
          )
      )
    );
    const client = createFervisHttpClient({
      authToken: "",
      baseUrl: "http://127.0.0.1:8000/fervis"
    });

    await expect(
      client.askAboutAnswer("q", "run", new Blob(["RIFFquestion"], { type: "audio/wav" }))
    ).rejects.toMatchObject({
      message: "Explanation audio is not configured.",
      name: "FervisApiError",
      status: 503
    } satisfies Partial<FervisApiError>);
  });

  it("submits typed deterministic reruns without model request fields", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(questionStateFixture), {
        headers: { "Content-Type": "application/json" },
        status: 202
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = createFervisHttpClient({
      authToken: "",
      baseUrl: "http://127.0.0.1:8000/fervis"
    });

    await client.rerunQuestion("q_sales", {
      triggerKind: "rerun",
      baseRunId: "run_sales",
      patch: {
        operations: [
          {
            kind: "set",
            parameterId: "population.sale_states",
            value: {
              kind: "string_set",
              values: ["COMPLETED", "PLACED"]
            }
          }
        ]
      }
    });

    const request = fetchMock.mock.calls[0];
    expect(request?.[0]).toBe(
      "http://127.0.0.1:8000/fervis/questions/q_sales/runs/"
    );
    expect(JSON.parse(String(request?.[1]?.body))).toEqual({
      triggerKind: "rerun",
      baseRunId: "run_sales",
      patch: {
        operations: [
          {
            kind: "set",
            parameterId: "population.sale_states",
            value: {
              kind: "string_set",
              values: ["COMPLETED", "PLACED"]
            }
          }
        ]
      }
    });
  });

  it("submits same-binding reruns without inventing an empty patch", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(questionStateFixture), {
        headers: { "Content-Type": "application/json" },
        status: 202
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = createFervisHttpClient({
      authToken: "",
      baseUrl: "http://127.0.0.1:8000/fervis"
    });

    await client.rerunQuestion("q_sales", {
      triggerKind: "rerun",
      baseRunId: "run_sales"
    });

    const request = fetchMock.mock.calls[0];
    expect(JSON.parse(String(request?.[1]?.body))).toEqual({
      triggerKind: "rerun",
      baseRunId: "run_sales"
    });
  });
});
