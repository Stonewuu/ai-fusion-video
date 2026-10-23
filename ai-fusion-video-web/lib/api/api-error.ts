import axios from "axios";
import { userFacingError } from "@/lib/user-facing-error";

export class ApiResultError extends Error {
  constructor(readonly code: number, message: string) {
    super(userFacingError(message));
    this.name = "ApiResultError";
  }
}

export function getApiPayloadMessage(payload: unknown): string | null {
  if (!isRecord(payload)) return null;
  const message = payload.msg;
  return typeof message === "string" && message.trim() ? message.trim() : null;
}

export function getApiErrorMessage(error: unknown): string {
  return userFacingError(rawApiErrorMessage(error));
}

function rawApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      const responseMessage = getApiPayloadMessage(error.response.data);
      if (responseMessage) return responseMessage;
      return "请求失败";
    }
    const transportMessage = nonEmpty(error.message);
    return transportMessage || error.name;
  }

  const directMessage = getApiPayloadMessage(error);
  if (directMessage) return directMessage;
  if (error instanceof Error) return nonEmpty(error.message) || error.name;
  return error == null ? "请求失败" : String(error);
}

export function toApiError(error: unknown): Error {
  const message = getApiErrorMessage(error);
  if (error instanceof Error && error.message === message) return error;
  return new Error(message, { cause: error });
}

/** 保留包装错误中的 HTTP 状态，用于撤权后清除已失效的数据投影。 */
export function getApiErrorStatus(error: unknown): number | null {
  const seen = new Set<unknown>();
  let current = error;
  while (current != null && !seen.has(current)) {
    seen.add(current);
    if (current instanceof ApiResultError && current.code >= 400 && current.code <= 599) return current.code;
    if (axios.isAxiosError(current) && current.response) return current.response.status;
    current = current instanceof Error ? current.cause : null;
  }
  return null;
}

export async function readApiResponseError(response: Response): Promise<string> {
  try {
    const body = await response.text();
    if (body.trim()) {
      try {
        const message = getApiPayloadMessage(JSON.parse(body));
        if (message) return userFacingError(message);
      } catch {
        // Error responses must use the CommonResult structure.
      }
    }
  } catch {
    // The contract error below also covers unreadable response bodies.
  }
  return "暂时无法完成请求，请稍后重试。";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmpty(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
