import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createOpenAIChatAdapter } from "../../../src/adapters/openai-chat";
import { buildResponseJSON } from "../../../src/bridge/response-json";
import { bridgeToResponsesSSE } from "../../../src/bridge/sse";
import { parseRequest } from "../../../src/responses/parser";
import { flushThoughtSignatureReplayForTests, resetThoughtSignatureReplayForTests } from "../../../src/responses/thought-signature-replay";
import type { AdapterEvent, OcxReasoningReplayScopeRef } from "../../../src/types";
import { withTestTranslatorBudget } from "../../helpers/translator-budget";
import { removeTreeWithRetry } from "../../helpers/remove-tree";

const MODEL = "google/gemini-test";
const SIGNATURE = "synthetic-opaque-signature-for-local-test";
const provider = { adapter: "openai-chat", baseUrl: "https://example.test/v1" };
const user = { role: "user", content: "Call exec once, then report its result." };
const tool = (custom: boolean) => custom
  ? { type: "custom", name: "exec", description: "Run a command." }
  : { type: "function", name: "exec", parameters: { type: "object", properties: { command: { type: "string" } } } };

function scope(): OcxReasoningReplayScopeRef {
  return { clientThreadId: "test-thread", current: {
    providerName: "test-provider", adapterName: "openai-chat", modelId: MODEL,
    providerDestinationIdentity: "test-destination", providerDestinationDurableIdentity: "test-destination",
    credentialIdentity: "test-credential", credentialDurableIdentity: "test-credential",
  } };
}

async function roundtrip(stream: boolean, custom: boolean) {
  const replayScope = scope();
  const adapter = withTestTranslatorBudget(createOpenAIChatAdapter(provider));
  await adapter.buildRequest(parseRequest({ model: MODEL, input: [user], tools: [tool(custom)], stream }));
  const argumentsText = JSON.stringify(custom ? { input: "printf OK" } : { command: "printf OK" });
  const call = { id: "call_exec", type: "function", function: { name: "exec", arguments: argumentsText }, extra_content: { google: { thought_signature: SIGNATURE } } };
  let output: Record<string, unknown>[];
  if (stream) {
    const wire = `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, ...call }] }, finish_reason: "tool_calls" }] })}\n\ndata: [DONE]\n\n`;
    const events = adapter.parseStream(new Response(wire));
    const response = new Response(bridgeToResponsesSSE(events, MODEL, undefined, custom ? new Set(["exec"]) : undefined, undefined, undefined, 2000, { replayCacheScope: replayScope }));
    output = (await response.text()).split("\n").filter(line => line.startsWith("data: ") && line !== "data: [DONE]").flatMap(line => {
      const event = JSON.parse(line.slice(6));
      return event.type === "response.output_item.done" ? [event.item] : [];
    });
  } else {
    const events: AdapterEvent[] = await adapter.parseResponse!(Response.json({ choices: [{ message: { role: "assistant", tool_calls: [call] }, finish_reason: "tool_calls" }] }));
    output = buildResponseJSON(events, MODEL, { replayCacheScope: replayScope, freeformToolNames: custom ? new Set(["exec"]) : undefined }).output as Record<string, unknown>[];
  }
  await flushThoughtSignatureReplayForTests();
  const item = output.find(item => item.type === (custom ? "custom_tool_call" : "function_call"));
  expect(item).toBeDefined();
  expect(item?.call_id).toBe("call_exec");
  // Codex can discard extension fields; freeform items cannot carry them at all.
  const stripped = { ...item };
  delete stripped.extra_content;
  const body = { model: MODEL, tools: [tool(custom)], input: [user, stripped, {
    type: custom ? "custom_tool_call_output" : "function_call_output", call_id: "call_exec", output: "OK",
  }] };
  return { adapter, body, replayScope, output };
}

describe("Gemini Chat tool continuation through Responses", () => {
  let previousHome: string | undefined;
  let testDir: string;
  beforeEach(() => {
    previousHome = process.env.OPENCODEX_HOME;
    testDir = mkdtempSync(join(tmpdir(), "ocx-chat-signature-"));
    process.env.OPENCODEX_HOME = testDir;
    resetThoughtSignatureReplayForTests();
  });
  afterEach(async () => {
    await flushThoughtSignatureReplayForTests();
    resetThoughtSignatureReplayForTests();
    if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
    else process.env.OPENCODEX_HOME = previousHome;
    removeTreeWithRetry(testDir);
  });

  for (const stream of [false, true]) for (const custom of [false, true]) {
    test(`${stream ? "SSE" : "JSON"} ${custom ? "custom" : "function"} call recovers stripped metadata after route binding`, async () => {
      const { adapter, body, replayScope } = await roundtrip(stream, custom);
      // Initial parsing precedes route/account selection, just as in production.
      const parsed = parseRequest(body);
      parsed._reasoningReplayScope = replayScope;
      const before = JSON.stringify(body);
      const request = await adapter.buildRequest(parsed);
      const wire = JSON.parse(request.body);
      const assistant = wire.messages.find((message: { role: string }) => message.role === "assistant");
      expect(assistant.tool_calls[0].extra_content.google.thought_signature).toBe(SIGNATURE);
      expect(assistant.tool_calls[0].id).toBe("call_exec");
      expect(JSON.parse(assistant.tool_calls[0].function.arguments)).toEqual(custom ? { input: "printf OK" } : { command: "printf OK" });
      expect(wire.messages.find((message: { role: string }) => message.role === "tool").content).toBe("OK");
      expect(JSON.stringify(body)).toBe(before);
    });
  }

  test("explicit metadata survives cold replay without a cache scope", async () => {
    const { adapter, body, output } = await roundtrip(false, false);
    const request = await adapter.buildRequest(parseRequest({ ...body, input: [user, output[0], body.input[2]] }));
    const wire = JSON.parse(request.body);
    expect(wire.messages.find((message: { role: string }) => message.role === "assistant").tool_calls[0].extra_content.google.thought_signature).toBe(SIGNATURE);
  });

  test("cache recovery does not cross thread, account, model, provider, endpoint or adapter boundaries", async () => {
    const { adapter, body, replayScope } = await roundtrip(false, true);
    const identities = [undefined,
      { ...replayScope, clientThreadId: "other-thread" },
      ...["credentialDurableIdentity", "modelId", "providerName", "providerDestinationDurableIdentity", "adapterName"].map(key => ({ ...replayScope, current: { ...replayScope.current!, [key]: "other" } })),
    ];
    for (const identity of identities) {
      const parsed = parseRequest(body);
      parsed._reasoningReplayScope = identity;
      const wire = JSON.parse((await adapter.buildRequest(parsed)).body);
      expect(wire.messages.find((message: { role: string }) => message.role === "assistant").tool_calls[0]).not.toHaveProperty("extra_content");
    }
  });

  test("switching a signed history to a non-Gemini Chat model omits Google extensions", async () => {
    const { adapter, body, output } = await roundtrip(false, false);
    const history = [user, output[0], body.input[2]];
    const before = JSON.stringify(history);
    const request = await adapter.buildRequest(parseRequest({ ...body, model: "other-model", input: history }));
    const wire = JSON.parse(request.body);
    expect(wire.messages.find((message: { role: string }) => message.role === "assistant").tool_calls[0]).not.toHaveProperty("extra_content");
    expect(JSON.stringify(history)).toBe(before);
  });
});
