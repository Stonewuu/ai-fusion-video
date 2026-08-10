<<<<<<< Updated upstream
import axios from "axios";
import type { CommonResult } from "./types";
import { getApiPayloadMessage, toApiError } from "./api-error";
import { getApiBaseUrl } from "@/lib/runtime-config";
import {
  AUTH_REFRESH_FAILED_ERROR_CODE,
  AUTH_SESSION_EXPIRED_ERROR_CODE,
  AuthFlowError,
} from "@/lib/auth-flow-error";

// 默认同源；Docker 可在容器启动时通过 PUBLIC_API_URL 注入分域后端地址。
const API_BASE_URL = getApiBaseUrl();

const PUBLIC_AUTH_PREFIXES = [
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/refresh",
  "/api/system/init/",
];

// 创建 axios 实例
const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * 从 localStorage 读取 auth-storage（zustand persist）
 */
function getAuthStorage() {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem("auth-storage");
    if (stored) return JSON.parse(stored);
  } catch {
    // 忽略
  }
  return null;
}

/**
 * 更新 localStorage 中的 token
 */
function updateAuthStorage(accessToken: string, refreshToken: string) {
  if (typeof window === "undefined") return;
  try {
    const stored = localStorage.getItem("auth-storage");
    if (stored) {
      const parsed = JSON.parse(stored);
      parsed.state.token = accessToken;
      parsed.state.refreshToken = refreshToken;
      localStorage.setItem("auth-storage", JSON.stringify(parsed));
    }
  } catch {
    // 忽略
  }
}

/**
 * 清除认证状态并跳转登录页
 */
function handleAuthFailure() {
  if (typeof window === "undefined") return;
  localStorage.removeItem("auth-storage");
  // 清除 auth-token cookie
  document.cookie = "auth-token=; path=/; max-age=0";
  if (window.location.pathname !== "/login") {
    window.location.href = "/login";
  }
}

function getRequestPath(url?: string, baseURL?: string) {
  if (!url) return "";
  try {
    if (url.startsWith("http://") || url.startsWith("https://")) {
      return new URL(url).pathname;
    }
    return new URL(url, baseURL ?? API_BASE_URL).pathname;
  } catch {
    return url.startsWith("/") ? url : `/${url}`;
  }
}

function isPublicAuthRequest(config?: { url?: string; baseURL?: string }) {
  const path = getRequestPath(config?.url, config?.baseURL);
  return PUBLIC_AUTH_PREFIXES.some((prefix) => path.startsWith(prefix));
}

// 请求拦截器：注入 access_token
http.interceptors.request.use((config) => {
  const parsed = getAuthStorage();
  const token = parsed?.state?.token;
  if (token && !isPublicAuthRequest(config)) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ---- 刷新令牌队列机制 ----
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (token: string) => void;
  reject: (err: Error) => void;
}> = [];

function processQueue(error: Error | null, token: string | null) {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token!);
    }
  });
  failedQueue = [];
}

// 响应拦截器：统一解包 CommonResult + 自动刷新令牌
http.interceptors.response.use(
  (response) => {
    const result = response.data as CommonResult<unknown>;
    if (result.code !== 0) {
      const message = getApiPayloadMessage(result);
      if (!message) return Promise.reject(new Error("请求失败"));
      return Promise.reject(new Error(message));
    }
    return result.data as never;
  },
  async (error) => {
    if (axios.isCancel(error)) {
      return Promise.reject(error);
    }

    const originalRequest = error.config;

    // 如果没有 config（例如网络错误），直接走通用错误处理
    if (!originalRequest || error.response?.status !== 401) {
      return Promise.reject(toApiError(error));
    }

    if (isPublicAuthRequest(originalRequest)) {
      return Promise.reject(toApiError(error));
    }

    // 已经是重试请求 → 不再重复刷新
    if (originalRequest._retry) {
      handleAuthFailure();
      return Promise.reject(
        new AuthFlowError(
          AUTH_SESSION_EXPIRED_ERROR_CODE,
          "登录已过期，请重新登录"
        )
      );
    }

    // 如果正在刷新中，将请求排队等待
    if (isRefreshing) {
      return new Promise<string>((resolve, reject) => {
        failedQueue.push({ resolve, reject });
      }).then((newToken) => {
        originalRequest.headers = {
          ...originalRequest.headers,
          Authorization: `Bearer ${newToken}`,
        };
        return http(originalRequest);
      });
    }

    const parsed = getAuthStorage();
    const storedRefreshToken = parsed?.state?.refreshToken;

    // 没有 refresh_token → 直接清除状态跳登录页
    if (!storedRefreshToken) {
      handleAuthFailure();
      return Promise.reject(
        new AuthFlowError(
          AUTH_SESSION_EXPIRED_ERROR_CODE,
          "登录已过期，请重新登录"
        )
      );
    }

    originalRequest._retry = true;
    isRefreshing = true;

    try {
      // 调用刷新接口（直接用 axios 避免走拦截器死循环）
      const refreshResp = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {
        refreshToken: storedRefreshToken,
      });

      const result = refreshResp.data as CommonResult<{
        accessToken: string;
        refreshToken: string;
      }>;

      if (result.code !== 0 || !result.data) {
        throw new Error(result.msg || "刷新令牌失败");
      }

      const { accessToken, refreshToken } = result.data;

      // 更新 localStorage
      updateAuthStorage(accessToken, refreshToken);

      // 同步更新 cookie（供 Next.js middleware 路由守卫使用）
      document.cookie = `auth-token=${accessToken}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax`;

      // 尝试更新 zustand store（如果已初始化）
      try {
        const { useAuthStore } = await import("@/lib/store/auth-store");
        useAuthStore.getState().setTokens(accessToken, refreshToken);
      } catch {
        // store 可能未初始化，localStorage 已更新所以问题不大
      }

      // 处理等待队列
      processQueue(null, accessToken);

      // 用新 token 重试原始请求
      originalRequest.headers = {
        ...originalRequest.headers,
        Authorization: `Bearer ${accessToken}`,
      };
      return http(originalRequest);
    } catch (refreshError) {
      // 刷新失败 → 清除认证状态，跳登录页
      const normalizedError = toApiError(refreshError);
      const authError = new AuthFlowError(
        AUTH_REFRESH_FAILED_ERROR_CODE,
        normalizedError.message,
        normalizedError
      );
      processQueue(authError, null);
      handleAuthFailure();
      return Promise.reject(authError);
    } finally {
      isRefreshing = false;
    }
  }
);

export { http, API_BASE_URL };

/**
 * 解析媒体资源 URL
 *
 * - 以 http:// 或 https:// 开头的完整 URL（如 OSS 直链）=> 原样返回
 * - 以 / 开头的相对路径（如 /media/images/xxx.png）=> 拼接后端 API_BASE_URL
 * - 空值 => 返回 null
 */
export function resolveMediaUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  if (url.startsWith("data:")) return url;
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${API_BASE_URL}${url}`;
  return url;
}
=======
import axios from "axios";
import type { CommonResult } from "./types";
import { getApiPayloadMessage, toApiError } from "./api-error";
import { getApiBaseUrl } from "@/lib/runtime-config";

// 默认同源；Docker 可在容器启动时通过 PUBLIC_API_URL 注入分域后端地址。
const API_BASE_URL = getApiBaseUrl();

const PUBLIC_AUTH_PREFIXES = [
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/refresh",
  "/api/system/init/",
];

// 创建 axios 实例
const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * 从 localStorage 读取 auth-storage（zustand persist）
 */
function getAuthStorage() {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem("auth-storage");
    if (stored) return JSON.parse(stored);
  } catch {
    // 忽略
  }
  return null;
}

/**
 * 更新 localStorage 中的 token
 */
function updateAuthStorage(accessToken: string, refreshToken: string) {
  if (typeof window === "undefined") return;
  try {
    const stored = localStorage.getItem("auth-storage");
    if (stored) {
      const parsed = JSON.parse(stored);
      parsed.state.token = accessToken;
      parsed.state.refreshToken = refreshToken;
      localStorage.setItem("auth-storage", JSON.stringify(parsed));
    }
  } catch {
    // 忽略
  }
}

/**
 * 清除认证状态并跳转登录页
 */
function handleAuthFailure() {
  if (typeof window === "undefined") return;
  localStorage.removeItem("auth-storage");
  // 清除 auth-token cookie
  document.cookie = "auth-token=; path=/; max-age=0";
  if (window.location.pathname !== "/login") {
    window.location.href = "/login";
  }
}

function getRequestPath(url?: string, baseURL?: string) {
  if (!url) return "";
  try {
    if (url.startsWith("http://") || url.startsWith("https://")) {
      return new URL(url).pathname;
    }
    return new URL(url, baseURL ?? API_BASE_URL).pathname;
  } catch {
    return url.startsWith("/") ? url : `/${url}`;
  }
}

function isPublicAuthRequest(config?: { url?: string; baseURL?: string }) {
  const path = getRequestPath(config?.url, config?.baseURL);
  return PUBLIC_AUTH_PREFIXES.some((prefix) => path.startsWith(prefix));
}

// 请求拦截器：注入 access_token
http.interceptors.request.use((config) => {
  const parsed = getAuthStorage();
  const token = parsed?.state?.token;
  if (token && !isPublicAuthRequest(config)) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ---- 刷新令牌队列机制 ----
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (token: string) => void;
  reject: (err: Error) => void;
}> = [];

function processQueue(error: Error | null, token: string | null) {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token!);
    }
  });
  failedQueue = [];
}

// 响应拦截器：统一解包 CommonResult + 自动刷新令牌
http.interceptors.response.use(
  (response) => {
    const result = response.data as CommonResult<unknown>;
    if (result.code !== 0) {
      const message = getApiPayloadMessage(result);
      if (!message) return Promise.reject(new Error("请求失败"));
      return Promise.reject(new Error(message));
    }
    return result.data as never;
  },
  async (error) => {
    if (axios.isCancel(error)) {
      return Promise.reject(error);
    }

    const originalRequest = error.config;

    // 如果没有 config（例如网络错误），直接走通用错误处理
    if (!originalRequest || error.response?.status !== 401) {
      return Promise.reject(toApiError(error));
    }

    if (isPublicAuthRequest(originalRequest)) {
      return Promise.reject(toApiError(error));
    }

    // 已经是重试请求 → 不再重复刷新
    if (originalRequest._retry) {
      handleAuthFailure();
      return Promise.reject(new Error("登录已过期，请重新登录"));
    }

    // 如果正在刷新中，将请求排队等待
    if (isRefreshing) {
      return new Promise<string>((resolve, reject) => {
        failedQueue.push({ resolve, reject });
      }).then((newToken) => {
        originalRequest.headers = {
          ...originalRequest.headers,
          Authorization: `Bearer ${newToken}`,
        };
        return http(originalRequest);
      });
    }

    const parsed = getAuthStorage();
    const storedRefreshToken = parsed?.state?.refreshToken;

    // 没有 refresh_token → 直接清除状态跳登录页
    if (!storedRefreshToken) {
      handleAuthFailure();
      return Promise.reject(new Error("登录已过期，请重新登录"));
    }

    originalRequest._retry = true;
    isRefreshing = true;

    try {
      // 调用刷新接口（直接用 axios 避免走拦截器死循环）
      const refreshResp = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {
        refreshToken: storedRefreshToken,
      });

      const result = refreshResp.data as CommonResult<{
        accessToken: string;
        refreshToken: string;
      }>;

      if (result.code !== 0 || !result.data) {
        throw new Error(result.msg || "刷新令牌失败");
      }

      const { accessToken, refreshToken } = result.data;

      // 更新 localStorage
      updateAuthStorage(accessToken, refreshToken);

      // 同步更新 cookie（供 Next.js middleware 路由守卫使用）
      document.cookie = `auth-token=${accessToken}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax`;

      // 尝试更新 zustand store（如果已初始化）
      try {
        const { useAuthStore } = await import("@/lib/store/auth-store");
        useAuthStore.getState().setTokens(accessToken, refreshToken);
      } catch {
        // store 可能未初始化，localStorage 已更新所以问题不大
      }

      // 处理等待队列
      processQueue(null, accessToken);

      // 用新 token 重试原始请求
      originalRequest.headers = {
        ...originalRequest.headers,
        Authorization: `Bearer ${accessToken}`,
      };
      return http(originalRequest);
    } catch (refreshError) {
      // 刷新失败 → 清除认证状态，跳登录页
      const normalizedError = toApiError(refreshError);
      processQueue(normalizedError, null);
      handleAuthFailure();
      return Promise.reject(normalizedError);
    } finally {
      isRefreshing = false;
    }
  }
);

export { http, API_BASE_URL };

/**
 * 解析媒体资源 URL
 *
 * - 以 http:// 或 https:// 开头的完整 URL（如 OSS 直链）=> 原样返回
 * - 以 / 开头的相对路径（如 /media/images/xxx.png）=> 拼接后端 API_BASE_URL
 * - 空值 => 返回 null
 */
export function resolveMediaUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  if (url.startsWith("data:")) return url;
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${API_BASE_URL}${url}`;
  return url;
}
>>>>>>> Stashed changes
