import { getApiBaseUrl } from "@/lib/runtime-config";
import { AUTH_REFRESH_FAILED_ERROR_CODE, AuthFlowError } from "@/lib/auth-flow-error";
import { getApiPayloadMessage, readApiResponseError, toApiError } from "./api-error";
import { createSessionRefresher, RefreshRejectedError, type RefreshSession } from "./session-refresh-coordinator";

const AUTH_STORAGE = "auth-storage";
const REFRESH_TIMEOUT_MS = 30_000;

export function readAuthSession(): RefreshSession | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = localStorage.getItem(AUTH_STORAGE);
  if (!raw) {
    return null;
  }
  let stored;
  try {
    stored = JSON.parse(raw);
  } catch {
    return null;
  }
  const state = stored?.state;
  if (!state || typeof state.token !== "string" || !state.token
      || typeof state.refreshToken !== "string" || !state.refreshToken
      || !Number.isSafeInteger(state.user?.id)) {
    return null;
  }
  return { userId: state.user.id, accessToken: state.token, refreshToken: state.refreshToken };
}

/** 仅清理发生失败的原会话，不能把另一标签页刚完成的登录清掉。 */
export function handleAuthFailure(expectedToken: string | null) {
  if (typeof window === "undefined") {
    return;
  }
  const current = readAuthSession();
  if (current && current.accessToken !== expectedToken) {
    return;
  }
  localStorage.removeItem(AUTH_STORAGE);
  document.cookie = "auth-token=; path=/; max-age=0";
  if (window.location.pathname !== "/login") {
    window.location.href = "/login";
  }
}

async function publish(previous: RefreshSession, next: RefreshSession) {
  const { useAuthStore } = await import("@/lib/store/auth-store");
  const current = readAuthSession();
  if (!current || current.userId !== previous.userId || current.refreshToken !== previous.refreshToken) {
    throw new Error("登录会话已变化，已放弃过期的刷新结果");
  }
  const stored = JSON.parse(localStorage.getItem(AUTH_STORAGE)!);
  useAuthStore.setState({ token: next.accessToken, refreshToken: next.refreshToken, user: stored.state.user });
  document.cookie = `auth-token=${next.accessToken}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax`;
}

function waitForRotation(previous: RefreshSession): Promise<RefreshSession> {
  return new Promise((resolve, reject) => {
    const finish = (session: RefreshSession | null) => {
      window.clearTimeout(timer);
      window.removeEventListener("storage", changed);
      if (session) {
        resolve(session);
      } else {
        reject(new Error("刷新令牌已失效，请重新登录"));
      }
    };
    const check = () => {
      const current = readAuthSession();
      if (!current || current.userId !== previous.userId) {
        finish(null);
      } else if (current.refreshToken !== previous.refreshToken) {
        finish(current);
      }
    };
    const changed = (event: StorageEvent) => {
      if (event.key === AUTH_STORAGE || event.key === null) {
        check();
      }
    };
    const timer = window.setTimeout(() => {
      const current = readAuthSession();
      finish(current && current.userId === previous.userId && current.refreshToken !== previous.refreshToken
        ? current : null);
    }, REFRESH_TIMEOUT_MS);
    window.addEventListener("storage", changed);
    check();
  });
}

const refresh = createSessionRefresher({
  read: readAuthSession,
  publish,
  waitForRotation,
  exchange: async (session) => {
    const response = await fetch(`${getApiBaseUrl()}/api/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshToken: session.refreshToken }),
      signal: AbortSignal.timeout(REFRESH_TIMEOUT_MS),
    });
    if (!response.ok) {
      throw new RefreshRejectedError(response.status, await readApiResponseError(response));
    }
    const result = await response.json();
    if (!result || typeof result.code !== "number") {
      throw new Error("刷新接口返回了无效的响应");
    }
    if (result.code !== 0) {
      throw new RefreshRejectedError(result.code, getApiPayloadMessage(result) ?? "刷新令牌失败");
    }
    const data = result.data;
    if (!data || !Number.isSafeInteger(data.userId) || typeof data.accessToken !== "string" || !data.accessToken
        || typeof data.refreshToken !== "string" || !data.refreshToken) {
      throw new Error("刷新接口返回了无效的会话");
    }
    return { userId: data.userId, accessToken: data.accessToken, refreshToken: data.refreshToken };
  },
});

export async function refreshAuthSession(failedToken: string | null, userId: number | null): Promise<string> {
  try {
    const next = await refresh(failedToken, userId);
    const { useAuthStore } = await import("@/lib/store/auth-store");
    const current = readAuthSession();
    if (!current || current.userId !== next.userId || current.refreshToken !== next.refreshToken) {
      throw new Error("登录会话已变化，已放弃过期的刷新结果");
    }
    if (useAuthStore.getState().token !== next.accessToken) {
      const stored = JSON.parse(localStorage.getItem(AUTH_STORAGE)!);
      useAuthStore.setState({ token: next.accessToken, refreshToken: next.refreshToken, user: stored.state.user });
    }
    return next.accessToken;
  } catch (error) {
    const normalized = toApiError(error);
    throw new AuthFlowError(AUTH_REFRESH_FAILED_ERROR_CODE, normalized.message, normalized);
  }
}
