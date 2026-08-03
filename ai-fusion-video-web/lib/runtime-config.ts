export interface FusionRuntimeConfig {
  apiBaseUrl?: string;
}

declare global {
  interface Window {
    __FUSION_RUNTIME_CONFIG__?: FusionRuntimeConfig;
  }
}

function normalizeBaseUrl(value: string | undefined): string {
  return (value ?? "").trim().replace(/\/+$/, "");
}

/**
 * 浏览器优先读取容器启动时生成的运行时配置。
 * 未配置 API 地址时表示明确采用同源部署模式。
 */
export function getApiBaseUrl(): string {
  const runtimeValue =
    typeof window === "undefined"
      ? undefined
      : window.__FUSION_RUNTIME_CONFIG__?.apiBaseUrl;

  return normalizeBaseUrl(runtimeValue);
}
