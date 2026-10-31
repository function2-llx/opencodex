import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { handleResponses } from "../../../src/server/responses/core";
import { flushThoughtSignatureReplayForTests, resetThoughtSignatureReplayForTests } from "../../../src/responses/thought-signature-replay";
import type { OcxConfig } from "../../../src/types";
import { acquireOwnedSpendHome } from "../../helpers/owned-spend-home";
import { removeTreeWithRetry } from "../../helpers/remove-tree";

const SIGNATURE = "synthetic-opaque-signature-for-dispatch-test";
// Codex selects `provider/vendor-model`; routing decodes it to the gateway's native slash id.
const SELECTOR = "gemini-gateway/google-gemini-test";
const WIRE_MODEL = "google/gemini-test";
const config = { providers: { "gemini-gateway": {
  adapter: "openai-chat", baseUrl: "https://example.test/v1", apiKey: "test-key", models: [WIRE_MODEL],
} } } as OcxConfig;
const user = { role: "user", content: "Call exec once, then report its result." };
const tool = (custom: boolean) => custom
  ? { type: "custom", name: "exec", description: "Run a command." }
  : { type: "function", name: "exec", parameters: { type: "object", properties: { command: { type: "string" } } } };

function chatReply(stream: boolean, message: Record<string, unknown>, finish: string): Response {
  if (!stream) return Response.json({ choices: [{ message: { role: "assistant", ...message }, finish_reason: finish }] });
  const calls = message.tool_calls as Record<string, unknown>[] | undefined;
  const delta = calls ? { tool_calls: calls.map((call, index) => ({ index, ...call })) } : message;
  return new Response(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: finish }] })}\n\ndata: [DONE]\n\n`,
    { headers: { "content-type": "text/event-stream" } });
}

describe("Gemini Chat signatures through Responses dispatch", () => {
  const originalFetch = globalThis.fetch;
  let previousHome: string | undefined;
  let testDir: string;
  let releaseSpendHome: (() => void) | undefined;
  beforeEach(() => {
    previousHome = process.env.OPENCODEX_HOME;
    testDir = mkdtempSync(join(tmpdir(), "ocx-chat-signature-dispatch-"));
    process.env.OPENCODEX_HOME = testDir;
    resetThoughtSignatureReplayForTests();
    releaseSpendHome = acquireOwnedSpendHome();
  });
  afterEach(async () => {
    releaseSpendHome?.();
    releaseSpendHome = undefined;
    globalThis.fetch = originalFetch;
    await flushThoughtSignatureReplayForTests();
    resetThoughtSignatureReplayForTests();
    if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
    else process.env.OPENCODEX_HOME = previousHome;
    removeTreeWithRetry(testDir);
  });

  for (const stream of [false, true]) for (const custom of [false, true]) {
    test(`${stream ? "SSE" : "JSON"} ${custom ? "custom" : "function"} call is re-signed after the client strips its metadata`, async () => {
      const argumentsText = JSON.stringify(custom ? { input: "printf OK" } : { command: "printf OK" });
      const requests: any[] = [];
      globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
        requests.push(JSON.parse(String(init?.body)));
        return requests.length === 1
          ? chatReply(stream, { tool_calls: [{ id: "call_exec", type: "function", function: { name: "exec", arguments: argumentsText }, extra_content: { google: { thought_signature: SIGNATURE } } }] }, "tool_calls")
          : chatReply(stream, { content: "Done" }, "stop");
      }) as typeof fetch;
      /** Dispatch one Codex-shaped turn and return the completed Responses object from JSON or SSE. */
      const send = async (input: unknown[], session: string) => {
        const response = await handleResponses(new Request("http://localhost/v1/responses", {
          method: "POST", headers: { "content-type": "application/json", session_id: session },
          body: JSON.stringify({ model: SELECTOR, input, stream, store: false, tools: [tool(custom)] }),
        }), config, { model: "", provider: "" });
        expect(response.status).toBe(200);
        const text = await response.text();
        if (!stream) return JSON.parse(text);
        return text.split("\n").filter(line => line.startsWith("data: {")).map(line => JSON.parse(line.slice(6)))
          .find(event => event.type === "response.completed")?.response;
      };

      const first = await send([user], "dispatch-session");
      const item = first.output.find((entry: any) => entry.type === (custom ? "custom_tool_call" : "function_call"));
      expect(item?.call_id).toBe("call_exec");
      // Function items expose the signature for cold replay; freeform items cannot carry it.
      expect(item.extra_content?.google?.thought_signature).toBe(custom ? undefined : SIGNATURE);
      // Codex drops extension fields, so recovery must come from the scoped replay store.
      const stripped = { ...item };
      delete stripped.extra_content;
      const history = [user, stripped, { type: custom ? "custom_tool_call_output" : "function_call_output", call_id: "call_exec", output: "OK" }];
      await send(history, "dispatch-session");
      await send(history, "another-session");

      expect(requests).toHaveLength(3);
      expect(requests[0].model).toBe(WIRE_MODEL);
      const replayed = (body: any) => body.messages.find((message: any) => message.role === "assistant" && message.tool_calls).tool_calls[0];
      expect(replayed(requests[1])).toMatchObject({ id: "call_exec", extra_content: { google: { thought_signature: SIGNATURE } } });
      expect(JSON.parse(replayed(requests[1]).function.arguments)).toEqual(JSON.parse(argumentsText));
      expect(requests[1].messages.find((message: any) => message.role === "tool").content).toBe("OK");
      expect(replayed(requests[2])).not.toHaveProperty("extra_content");
    });
  }
});
