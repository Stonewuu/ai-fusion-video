import {
  AUTH_REFRESH_FAILED_ERROR_CODE,
  AUTH_SESSION_EXPIRED_ERROR_CODE,
} from "@/lib/auth-flow-error";

export function getClientErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message.trim();
  }
  if (typeof error === "string" && error.trim()) {
    return error.trim();
  }
  return "发生未知客户端异常";
}

export function isIgnoredClientError(error: unknown): boolean {
  const message = getClientErrorMessage(error);
  const errorName = error instanceof Error ? error.name : undefined;
  const errorCode = getStringProperty(error, "code");
  const errorDigest = getStringProperty(error, "digest");

  return errorName === "AbortError"
    || errorName === "CanceledError"
    || errorCode === "ERR_CANCELED"
    || errorCode === AUTH_SESSION_EXPIRED_ERROR_CODE
    || errorCode === AUTH_REFRESH_FAILED_ERROR_CODE
    || message.includes("NEXT_REDIRECT")
    || message.includes("NEXT_NOT_FOUND")
    || errorDigest?.includes("NEXT_REDIRECT") === true
    || errorDigest?.includes("NEXT_NOT_FOUND") === true
    || message === "登录已过期，请重新登录"
    || message === "刷新令牌失败";
}

function getStringProperty(error: unknown, key: string): string | undefined {
  if (typeof error !== "object" || error === null || !(key in error)) {
    return undefined;
  }
  return String((error as Record<string, unknown>)[key]);
}
