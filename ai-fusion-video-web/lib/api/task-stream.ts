import { API_BASE_URL, http } from "./client";
import { readApiResponseError } from "./api-error";
import type { AgentConversation } from "./ai-assistant";
import {
  authenticatedFetch,
  type AiChatStreamEvent,
  type StreamCallbacks,
} from "./ai-assistant";

export type { AiChatStreamEvent, StreamCallbacks };

function parseSseEventBlock(eventBlock: string, callbacks: StreamCallbacks) {
  const dataLines: string[] = [];

  for (const rawLine of eventBlock.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  const jsonStr = dataLines.join("\n").trim();
  if (!jsonStr) {
    return;
  }

  try {
    const event: AiChatStreamEvent = JSON.parse(jsonStr);
    callbacks.onEvent(event);
  } catch {
    console.warn("Task SSE 解析失败:", jsonStr);
  }
}

function consumeSseBuffer(buffer: string, callbacks: StreamCallbacks) {
  const normalizedBuffer = buffer.replace(/\r\n/g, "\n");
  const eventBlocks = normalizedBuffer.split("\n\n");
  const remaining = eventBlocks.pop() || "";

  for (const eventBlock of eventBlocks) {
    parseSseEventBlock(eventBlock, callbacks);
  }

  return remaining;
}

export function reconnectTaskStream(
  taskId: string,
  callbacks: StreamCallbacks
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await authenticatedFetch(
        `${API_BASE_URL}/api/task-stream/reconnect?taskId=${encodeURIComponent(taskId)}`,
        {
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        throw new Error(await readApiResponseError(response));
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("无法获取响应流");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        buffer = consumeSseBuffer(buffer, callbacks);
      }

      if (buffer.trim()) {
        parseSseEventBlock(buffer, callbacks);
      }

      callbacks.onComplete?.();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      callbacks.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  })();

  return controller;
}

export async function getTaskStreamStatus(taskId: string): Promise<string> {
  return http.get(
    `/api/task-stream/status?taskId=${encodeURIComponent(taskId)}`
  );
}

export async function listRunningTaskStreams(): Promise<AgentConversation[]> {
  return http.get("/api/task-stream/running");
}
