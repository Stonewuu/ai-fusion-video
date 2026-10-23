import axios from "axios";
import type { CommonResult } from "./types";
import { ApiResultError, getApiPayloadMessage, toApiError } from "./api-error";
import { getApiBaseUrl } from "@/lib/runtime-config";
import { handleAuthFailure, readAuthSession, refreshAuthSession } from "./refresh-session";
import {
  AUTH_SESSION_EXPIRED_ERROR_CODE,
  AuthFlowError,
} from "@/lib/auth-flow-error";

declare module "axios" {
  interface InternalAxiosRequestConfig {
    _authUserId?: number;
    _retry?: boolean;
  }
}

const PUBLIC_AUTH_PREFIXES = [
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/refresh",
  "/api/system/init/",
];

// 创建 axios 实例
const http = axios.create({
  timeout: 30000,
});

function getRequestPath(url?: string, baseURL?: string) {
  if (!url) {
    return "";
  }
  try {
    if (url.startsWith("http://") || url.startsWith("https://")) {
      return new URL(url).pathname;
    }
    return new URL(url, baseURL ?? getApiBaseUrl()).pathname;
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
  // 客户端模块可能先于 beforeInteractive 配置脚本求值，不能在导入时缓存空地址。
  config.baseURL = getApiBaseUrl();
  const session = readAuthSession();
  const token = session?.accessToken;
  config._authUserId = session?.userId;
  if (token && !isPublicAuthRequest(config)) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 响应拦截器：统一解包 CommonResult + 自动刷新令牌
http.interceptors.response.use(
  (response) => {
    const result = response.data as CommonResult<unknown>;
    if (result.code !== 0) {
      const message = getApiPayloadMessage(result);
      return Promise.reject(new ApiResultError(result.code, message ?? "请求失败"));
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

    const attemptedToken = String(originalRequest.headers?.Authorization ?? "").replace(/^Bearer /, "") || null;
    if (originalRequest._retry) {
      handleAuthFailure(attemptedToken);
      return Promise.reject(
        new AuthFlowError(AUTH_SESSION_EXPIRED_ERROR_CODE, "登录已过期，请重新登录")
      );
    }

    originalRequest._retry = true;
    try {
      const accessToken = await refreshAuthSession(attemptedToken, originalRequest._authUserId ?? null);
      originalRequest.headers = {
        ...originalRequest.headers,
        Authorization: "Bearer " + accessToken,
      };
      return http(originalRequest);
    } catch (refreshError) {
      handleAuthFailure(attemptedToken);
      return Promise.reject(refreshError);
    }
  }
);

export { http, getApiBaseUrl };

/**
 * 解析媒体资源 URL
 *
 * - 以 http:// 或 https:// 开头的完整 URL（如 OSS 直链）=> 原样返回
 * - 以 / 开头的相对路径（如 /media/images/xxx.png）=> 拼接当前后端地址
 * - 空值 => 返回 null
 */
export function resolveMediaUrl(url: string | null | undefined): string | null {
  if (!url) {
    return null;
  }
  if (url.startsWith("data:")) {
    return url;
  }
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  if (url.startsWith("/")) {
    return `${getApiBaseUrl()}${url}`;
  }
  return url;
}
