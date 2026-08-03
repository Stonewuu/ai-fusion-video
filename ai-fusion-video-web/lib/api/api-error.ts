import axios from "axios";

export function getApiPayloadMessage(payload: unknown): string | null {
  if (!isRecord(payload)) return null;
  const message = payload.msg;
  return typeof message === "string" && message.trim() ? message.trim() : null;
}

export function getApiErrorMessage(error: unknown): string {
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

export async function readApiResponseError(response: Response): Promise<string> {
  try {
    const body = await response.text();
    if (body.trim()) {
      try {
        const message = getApiPayloadMessage(JSON.parse(body));
        if (message) return message;
      } catch {
        // Error responses must use the CommonResult structure.
      }
    }
  } catch {
    // The contract error below also covers unreadable response bodies.
  }
  return "请求失败";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmpty(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
