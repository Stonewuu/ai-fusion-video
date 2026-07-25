/**
 * Durable AI Pipeline API.
 *
 * Pipeline events are journal-backed and identified by (runId, sequence).
 * Transport completion is never treated as business completion.
 */

import { API_BASE_URL, http } from "./client";
import {
  authenticatedFetch,
  type AiChatReq,
  type AiChatStreamEvent as BaseAiChatStreamEvent,
} from "./ai-assistant";

export type { AiChatReq };

// ========== Pipeline types ==========

/** Pipeline agent types whose user input is hidden in the conversation UI. */
export const PIPELINE_AGENT_TYPES = [
  "story_to_script",
  "script_full_parse",
  "script_to_storyboard",
  "script_episode_parse",
  "episode_scene_writer",
  "episode_script_creator",
  "episode_storyboard_writer",
  "storyboard_asset_preprocessor",
  "asset_image_gen",
  "asset_image_executor",
  "storyboard_frame_gen",
  "storyboard_frame_executor",
  "storyboard_video_gen",
  "storyboard_video_executor",
] as const;

export interface PendingToolCallInfo {
  toolCallId: string;
  toolName: string;
  argumentsPreview: string;
}

/** A strictly identified event from the durable Pipeline journal. */
export interface AiChatStreamEvent extends BaseAiChatStreamEvent {
  schemaVersion: 1;
  runId: string;
  sequence: number;
  source?: string;
  replyId?: string;
  blockId?: string;
  rawEventId?: string;
  rawEventType?: string;
  createdAt?: string;
  controlType?: string;
  pendingToolCalls?: PendingToolCallInfo[];
  expiresAt?: string;
}

export interface StreamCallbacks {
  onEvent: (event: AiChatStreamEvent) => void;
  onError?: (error: Error) => void;
  /** Called only after the stream has delivered a root terminal event. */
  onComplete?: () => void;
}

export type PipelineRunStatus =
  | "RUNNING"
  | "WAITING_CONFIRMATION"
  | "WAITING_EXTERNAL"
  | "CANCEL_REQUESTED"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export interface PipelineRunStatusResponse {
  runId: string;
  status: PipelineRunStatus;
  lastSequence: number;
  waitingReplyId?: string;
  terminalEvent?: AiChatStreamEvent;
}

export interface RunningPipelineRun {
  runId: string;
  conversationId: string;
  projectId: number;
  title: string;
  category: string;
  status: Exclude<PipelineRunStatus, "COMPLETED" | "FAILED" | "CANCELLED">;
  lastSequence: number;
  waitingReplyId?: string;
  startedAt: string;
}

export type PipelineRunTarget =
  | { runId: string; conversationId?: never }
  | { conversationId: string; runId?: never };

interface StreamCursor {
  runId?: string;
  lastSequence: number;
  terminalSeen: boolean;
}

const OUTPUT_TYPES = new Set([
  "REASONING",
  "CONTENT",
  "TOOL_CALL",
  "TOOL_FINISHED",
  "SUB_AGENT_STARTED",
  "SUB_AGENT_FINISHED",
  "USER_CONFIRMATION_REQUIRED",
  "EXTERNAL_EXECUTION_REQUIRED",
  "USER_CONFIRM_RESULT",
  "EXTERNAL_EXECUTION_RESULT",
  "DONE",
  "ERROR",
  "CANCELLED",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseEventId(value: string): { runId: string; sequence: number } {
  const separator = value.lastIndexOf(":");
  if (separator <= 0 || separator === value.length - 1) {
    throw new Error("Pipeline SSE id is invalid");
  }
  const runId = value.slice(0, separator);
  const encodedSequence = value.slice(separator + 1);
  if (!/^\d+$/.test(encodedSequence)) {
    throw new Error("Pipeline SSE id sequence is invalid");
  }
  const sequence = Number(encodedSequence);
  if (!Number.isSafeInteger(sequence) || sequence <= 0) {
    throw new Error("Pipeline SSE id sequence is outside the safe range");
  }
  return { runId, sequence };
}

function parsePipelineEvent(jsonText: string): AiChatStreamEvent {
  let value: unknown;
  try {
    value = JSON.parse(jsonText);
  } catch (cause) {
    throw new Error("Pipeline SSE data is not valid JSON", { cause });
  }
  if (!isRecord(value)) {
    throw new Error("Pipeline SSE data must be an object");
  }
  if (value.schemaVersion !== 1) {
    throw new Error("Unsupported Pipeline SSE schema version");
  }
  if (typeof value.runId !== "string" || value.runId.trim() !== value.runId || !value.runId) {
    throw new Error("Pipeline SSE runId is invalid");
  }
  if (!Number.isSafeInteger(value.sequence) || (value.sequence as number) <= 0) {
    throw new Error("Pipeline SSE sequence is invalid");
  }
  if (typeof value.outputType !== "string" || !OUTPUT_TYPES.has(value.outputType)) {
    throw new Error("Pipeline SSE outputType is invalid");
  }
  return value as unknown as AiChatStreamEvent;
}

function isRootTerminalEvent(event: AiChatStreamEvent): boolean {
  return (
    !event.parentToolCallId &&
    !event.agentName &&
    (event.outputType === "DONE" ||
      event.outputType === "ERROR" ||
      event.outputType === "CANCELLED")
  );
}

function parseSseEventBlock(
  eventBlock: string,
  callbacks: StreamCallbacks,
  cursor: StreamCursor
) {
  const dataLines: string[] = [];
  const idLines: string[] = [];

  for (const rawLine of eventBlock.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    } else if (line.startsWith("id:")) {
      idLines.push(line.slice(3).trimStart());
    }
  }

  const jsonText = dataLines.join("\n").trim();
  if (!jsonText) return;
  if (idLines.length !== 1) {
    throw new Error("Pipeline SSE event must contain exactly one id field");
  }

  const eventId = parseEventId(idLines[0]);
  const event = parsePipelineEvent(jsonText);
  if (event.runId !== eventId.runId || event.sequence !== eventId.sequence) {
    throw new Error("Pipeline SSE id does not match its data identity");
  }
  if (cursor.runId && event.runId !== cursor.runId) {
    throw new Error("Pipeline SSE switched to a different run");
  }
  if (event.sequence <= cursor.lastSequence) {
    return;
  }

  callbacks.onEvent(event);
  cursor.runId = event.runId;
  cursor.lastSequence = event.sequence;
  if (isRootTerminalEvent(event)) {
    cursor.terminalSeen = true;
  }
}

function consumeSseBuffer(
  buffer: string,
  callbacks: StreamCallbacks,
  cursor: StreamCursor
) {
  const normalizedBuffer = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const eventBlocks = normalizedBuffer.split("\n\n");
  const remaining = eventBlocks.pop() || "";

  for (const eventBlock of eventBlocks) {
    parseSseEventBlock(eventBlock, callbacks, cursor);
  }
  return remaining;
}

async function consumePipelineResponse(
  response: Response,
  callbacks: StreamCallbacks,
  cursor: StreamCursor
) {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json() as { msg?: unknown; message?: unknown };
      if (typeof payload.msg === "string" && payload.msg.trim()) detail = payload.msg.trim();
      else if (typeof payload.message === "string" && payload.message.trim()) detail = payload.message.trim();
    } catch {
      // Some proxies return an empty or non-JSON error body. The status still
      // provides a deterministic fallback for the connection state machine.
    }
    throw new Error(detail ? `HTTP ${response.status}: ${detail}` : `HTTP ${response.status}`);
  }
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("无法获取 Pipeline 响应流");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = consumeSseBuffer(buffer, callbacks, cursor);
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    parseSseEventBlock(buffer.replace(/\r\n/g, "\n"), callbacks, cursor);
  }
  if (!cursor.terminalSeen) {
    throw new Error("Pipeline SSE ended before a terminal journal event");
  }
  callbacks.onComplete?.();
}

function runStreamRequest(
  request: () => Promise<Response>,
  callbacks: StreamCallbacks,
  cursor: StreamCursor,
  controller: AbortController
) {
  void (async () => {
    try {
      await consumePipelineResponse(await request(), callbacks, cursor);
    } catch (error) {
      if (controller.signal.aborted) return;
      callbacks.onError?.(
        error instanceof Error ? error : new Error(String(error))
      );
    }
  })();
}

// ========== SSE API ==========

export function pipelineStream(
  req: AiChatReq,
  callbacks: StreamCallbacks
): AbortController {
  const controller = new AbortController();
  const cursor: StreamCursor = { lastSequence: 0, terminalSeen: false };
  runStreamRequest(
    () =>
      authenticatedFetch(`${API_BASE_URL}/api/ai/pipeline/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
        signal: controller.signal,
      }),
    callbacks,
    cursor,
    controller
  );
  return controller;
}

export function reconnectPipelineStream(
  runId: string,
  afterSequence: number,
  callbacks: StreamCallbacks
): AbortController {
  if (!runId || runId.trim() !== runId) {
    throw new Error("runId is required for Pipeline reconnect");
  }
  if (!Number.isSafeInteger(afterSequence) || afterSequence < 0) {
    throw new Error("afterSequence must be a non-negative safe integer");
  }

  const controller = new AbortController();
  const cursor: StreamCursor = {
    runId,
    lastSequence: afterSequence,
    terminalSeen: false,
  };
  const eventId = `${runId}:${afterSequence}`;
  const query = new URLSearchParams({
    runId,
    afterSequence: String(afterSequence),
  });
  runStreamRequest(
    () =>
      authenticatedFetch(
        `${API_BASE_URL}/api/ai/pipeline/reconnect?${query.toString()}`,
        {
          headers: { "Last-Event-ID": eventId },
          signal: controller.signal,
        }
      ),
    callbacks,
    cursor,
    controller
  );
  return controller;
}

// ========== Query API ==========

function targetQuery(target: PipelineRunTarget): string {
  const params = new URLSearchParams();
  if (target.runId) params.set("runId", target.runId);
  if (target.conversationId) params.set("conversationId", target.conversationId);
  return params.toString();
}

export async function cancelPipeline(target: PipelineRunTarget): Promise<void> {
  await http.post(`/api/ai/pipeline/cancel?${targetQuery(target)}`);
}

export async function getPipelineStatus(
  target: PipelineRunTarget
): Promise<PipelineRunStatusResponse> {
  return http.get(`/api/ai/pipeline/status?${targetQuery(target)}`);
}

export async function listRunningPipelines(): Promise<RunningPipelineRun[]> {
  return http.get("/api/ai/pipeline/running");
}
