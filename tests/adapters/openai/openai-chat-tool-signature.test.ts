import { describe, expect, test } from "bun:test";
import { createOpenAIChatAdapter } from "../../../src/adapters/openai-chat";
import { createTestTranslatorBudget, withTestTranslatorBudget } from "../../helpers/translator-budget";
import type { AdapterEvent } from "../../../src/types";
import { parseRequest } from "../../../src/responses/parser";
import { COMPLETED_TOOL_BLOCKS_PROTOCOL } from "../../../src/adapters/openai-chat/completed-tool-blocks";

const provider = { adapter: "openai-chat", baseUrl: "https://example.test/v1" };
const model = "google/gemini-test";
const extra = (signature: unknown) => ({ extra_content: { google: { thought_signature: signature } } });
const call = (id: string, fields = {}) => ({ id, type: "function", function: { name: "exec", arguments: "{}" }, ...fields });
const chunk = (calls: unknown[], finish?: string) => ({ choices: [{ delta: { tool_calls: calls }, ...(finish ? { finish_reason: finish } : {}) }] });
const starts = (events: AdapterEvent[]) => events.filter((e): e is Extract<AdapterEvent, { type: "tool_call_start" }> => e.type === "tool_call_start");

async function adapterFor(modelId = model, negotiated = false) {
  const adapter = withTestTranslatorBudget(createOpenAIChatAdapter({ ...provider, ...(negotiated ? { headers: { "x-opencodex-tool-completion": COMPLETED_TOOL_BLOCKS_PROTOCOL } } : {}) }));
  await adapter.buildRequest(parseRequest({ model: modelId, input: "run a command", tools: [{ type: "function", name: "exec", parameters: { type: "object" } }] }));
  return adapter;
}

async function parse(calls: unknown[], modelId = model) {
  const adapter = await adapterFor(modelId);
  return adapter.parseResponse!(Response.json({ choices: [{ message: { role: "assistant", tool_calls: calls }, finish_reason: "tool_calls" }] }));
}

async function stream(chunks: unknown[], options: { negotiated?: boolean; limit?: number; done?: boolean } = {}) {
  const adapter = await adapterFor(model, options.negotiated);
  const budget = createTestTranslatorBudget({ maxCallArgumentBytes: options.limit });
  const wire = chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join("") + (options.done === false ? "" : "data: [DONE]\n\n");
  const events: AdapterEvent[] = [];
  for await (const event of adapter.parseStream(new Response(wire, { headers: options.negotiated ? { "x-opencodex-tool-completion": COMPLETED_TOOL_BLOCKS_PROTOCOL } : {} }), budget, undefined, { completedToolBlocks: options.negotiated })) events.push(event);
  return { events, budget };
}

describe("Chat tool-call thought signatures", () => {
  test("nonstream preserves per-call metadata and the Gemini signature alias", async () => {
    const events = await parse([call("a", extra("opaque-A")), call("b", { signature: "opaque-B" }), call("c")]);
    expect(starts(events).map(e => e.providerMetadata?.google?.thoughtSignature)).toEqual(["opaque-A", "opaque-B", undefined]);
    expect(starts(events)[2]).not.toHaveProperty("providerMetadata");
  });

  test("explicit Google metadata wins and unrelated bare signatures stay unclassified", async () => {
    const events = await parse([call("a", { ...extra("original"), signature: "alias" })]);
    expect(starts(events)[0]?.providerMetadata?.google?.thoughtSignature).toBe("original");
    expect(starts(await parse([call("b", { signature: "other-vendor-signature" })], "another-model"))[0]).not.toHaveProperty("providerMetadata");
  });

  test("malformed, empty and oversized UTF-8 signatures carry nothing", async () => {
    for (const value of [null, 5, {}, [], "", "x".repeat(65537), "界".repeat(21846)]) {
      expect(starts(await parse([call("a", extra(value)), call("b", { signature: value })]))).toEqual([
        { type: "tool_call_start", id: "a", name: "exec" }, { type: "tool_call_start", id: "b", name: "exec" },
      ]);
    }
    const atLimit = "x".repeat(65536);
    expect(starts(await parse([call("a", extra(atLimit))]))[0]?.providerMetadata?.google?.thoughtSignature).toBe(atLimit);
  });

  test("late and repeated metadata follows interleaved calls without concatenation", async () => {
    const { events, budget } = await stream([
      chunk([{ index: 0, ...call("a") }, { index: 1, ...call("b", { signature: "opaque-B" }) }]),
      chunk([{ index: 0, ...extra("opaque-A") }]),
      chunk([{ index: 0, ...extra("opaque-A") }], "tool_calls"),
    ]);
    expect(starts(events).map(e => [e.id, e.providerMetadata?.google?.thoughtSignature])).toEqual([["a", "opaque-A"], ["b", "opaque-B"]]);
    expect(events.at(-1)?.type).toBe("done");
    expect(budget.snapshot().activeCalls).toBe(0);
    expect(budget.snapshot().currentBytes).toBe(0);
  });

  test("an ID-only call keeps its signature when it learns an index", async () => {
    const { events } = await stream([
      chunk([call("a", extra("opaque-A"))]),
      chunk([{ id: "a", index: 3 }]), chunk([{ index: 3, ...extra("opaque-A") }], "tool_calls"),
    ]);
    expect(starts(events)[0]?.providerMetadata?.google?.thoughtSignature).toBe("opaque-A");
  });

  test("conflicting signatures fail without exposing opaque values", async () => {
    const { events, budget } = await stream([
      chunk([{ index: 0, ...call("a", extra("first-secret-value")) }]),
      chunk([{ index: 0, ...extra("second-secret-value") }], "tool_calls"),
    ]);
    expect(starts(events)).toHaveLength(0);
    expect(events.at(-1)).toEqual({ type: "error", message: "upstream tool call contained conflicting thought signatures" });
    expect(JSON.stringify(events)).not.toContain("secret-value");
    expect(budget.snapshot().currentBytes).toBe(0);
  });

  test("metadata consumes the pending-call budget and overflow releases it", async () => {
    const { events, budget } = await stream([chunk([{ index: 0, ...call("a", extra("界".repeat(4))) }])], { limit: 10 });
    expect(starts(events)).toHaveLength(0);
    expect(events.at(-1)).toMatchObject({ type: "error", code: "translation_buffer_limit" });
    expect(budget.snapshot().activeCalls).toBe(0);
    expect(budget.snapshot().currentBytes).toBe(0);
  });

  test("truncated streams release retained metadata without dispatching calls", async () => {
    const { events, budget } = await stream([chunk([{ index: 0, ...call("a", extra("opaque-A")) }])], { done: false });
    expect(starts(events)).toHaveLength(0);
    expect(events.at(-1)?.type).toBe("error");
    expect(budget.snapshot().currentBytes).toBe(0);
  });

  test("negotiated early completion carries metadata supplied on the completion marker", async () => {
    const { events } = await stream([
      chunk([{ index: 0, ...call("a") }]),
      chunk([{ index: 0, id: "a", x_opencodex_completed: COMPLETED_TOOL_BLOCKS_PROTOCOL, ...extra("opaque-A") }]),
      chunk([], "tool_calls"),
    ], { negotiated: true });
    expect(starts(events)).toHaveLength(1);
    expect(starts(events)[0]?.providerMetadata?.google?.thoughtSignature).toBe("opaque-A");
  });
});
