import { getApiBaseUrl } from "@/lib/api/client";
import { authenticatedFetch } from "@/lib/api/ai-assistant";
import type { CanvasEvent } from "./types";

export class CanvasStreamError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

export function parseCanvasEvent(block: string): CanvasEvent | null {
  let eventType = "";
  let eventId = "";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    if (line.startsWith("id:")) eventId = line.slice(3).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  const parsed = JSON.parse(data.join("\n"));
  if (eventType === "canvas.error")
    throw new CanvasStreamError(parsed.message, parsed.errorCode);
  if (
    !parsed ||
    typeof parsed !== "object" ||
    parsed.type !== eventType ||
    !Number.isSafeInteger(parsed.sequence) ||
    parsed.sequence <= 0 ||
    String(parsed.sequence) !== eventId ||
    typeof parsed.canvasId !== "string" ||
    typeof parsed.payload !== "object" ||
    !parsed.payload
  ) {
    throw new CanvasStreamError(
      "画布事件格式无效，需重新同步",
      "INVALID_EVENT",
    );
  }
  return parsed as CanvasEvent;
}

export async function streamCanvas(
  canvasId: string,
  afterSequence: number,
  signal: AbortSignal,
  consume: (event: CanvasEvent) => Promise<void>,
  connected: () => void,
): Promise<void> {
  const response = await authenticatedFetch(
    `${getApiBaseUrl()}/api/canvases/${canvasId}/stream?afterSequence=${afterSequence}`,
    { signal },
  );
  if (!response.ok) {
    throw new CanvasStreamError(
      `画布同步请求失败（${response.status}）`,
      "HTTP_ERROR",
      response.status,
    );
  }
  if (
    !response.body ||
    !response.headers.get("content-type")?.includes("text/event-stream")
  ) {
    throw new CanvasStreamError("服务器未返回画布事件流", "INVALID_STREAM");
  }
  connected();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (!signal.aborted) {
      const part = await reader.read();
      if (part.done) break;
      buffer += decoder.decode(part.value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      if (buffer.length > 16 * 1024 * 1024)
        throw new CanvasStreamError("画布事件超过可处理大小", "INVALID_STREAM");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const event = parseCanvasEvent(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (event) await consume(event);
      }
    }
    if (!signal.aborted)
      throw new CanvasStreamError("画布同步连接已关闭", "DISCONNECTED");
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
