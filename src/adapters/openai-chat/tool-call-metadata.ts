import type { OcxParsedRequest, OcxProviderOpaqueToolCallMetadata, OcxToolCall } from "../../types";
import type { TranslatorBudget } from "../../lib/translator-budget";
import { isCarryableSignature, providerMetadataFromResponsesFunctionCall, responsesExtraContentFromProviderMetadata } from "../../responses/provider-opaque-metadata";
import { lookupReplayThoughtSignature } from "../../responses/thought-signature-replay";

function isGeminiModel(modelId: string | undefined): boolean {
  return /(?:^|\/)gemini(?:[-.:]|$)/i.test(modelId ?? "");
}

/** Explicit Google metadata is unambiguous; a bare signature needs a Gemini destination. */
export function chatToolCallMetadata(call: Record<string, unknown>, modelId?: string): OcxProviderOpaqueToolCallMetadata | undefined {
  const metadata = providerMetadataFromResponsesFunctionCall({ extra_content: call.extra_content });
  if (metadata) return metadata;
  return isGeminiModel(modelId) && isCarryableSignature(call.signature)
    ? { google: { thoughtSignature: call.signature } } : undefined;
}

/** Stream signatures are complete opaque values, never string deltas to concatenate. */
export function retainChatToolCallMetadata(
  call: { key: string; providerMetadata?: OcxProviderOpaqueToolCallMetadata },
  raw: Record<string, unknown>,
  modelId: string | undefined,
  budget: TranslatorBudget,
): boolean {
  const metadata = chatToolCallMetadata(raw, modelId);
  const signature = metadata?.google?.thoughtSignature;
  if (!signature) return true;
  const previous = call.providerMetadata?.google?.thoughtSignature;
  if (previous !== undefined) return previous === signature;
  budget.chargeRetained(Buffer.byteLength(signature, "utf8"), { kind: "tool_args", callId: call.key });
  call.providerMetadata = metadata;
  return true;
}

/** Recover only after routing has bound the existing thread/destination/credential scope. */
export function chatToolCallExtraContent(call: OcxToolCall, parsed: OcxParsedRequest) {
  if (!isGeminiModel(parsed._wireModelOverride ?? parsed.modelId)) return undefined;
  const explicit = responsesExtraContentFromProviderMetadata(call.providerMetadata);
  if (explicit) return explicit;
  const signature = lookupReplayThoughtSignature(call.id, parsed._reasoningReplayScope);
  return signature ? responsesExtraContentFromProviderMetadata({ google: { thoughtSignature: signature } }) : undefined;
}
