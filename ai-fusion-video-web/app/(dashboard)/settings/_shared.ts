// 设置页面共享的常量、工具函数和动画变量

// 容器动画
export const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.1,
    },
  },
};

// 子元素动画
export const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number],
    },
  },
};

// ---------- 平台图标颜色映射 ----------

export const platformIconColors: Record<string, { color: string; bg: string }> = {
  openai_compatible: { color: "text-emerald-400", bg: "bg-emerald-500/10" },
  volcengine: { color: "text-sky-400", bg: "bg-sky-500/10" },
  vertex_ai: { color: "text-blue-400", bg: "bg-blue-500/10" },
  gemini: { color: "text-teal-400", bg: "bg-teal-500/10" },
  GoogleFlowReverseApi: { color: "text-cyan-400", bg: "bg-cyan-500/10" },
  dashscope: { color: "text-orange-400", bg: "bg-orange-500/10" },
  anthropic: { color: "text-amber-400", bg: "bg-amber-500/10" },
  ollama: { color: "text-violet-400", bg: "bg-violet-500/10" },
};

// ---------- 密钥脱敏 ----------

export function maskSecret(value: string | null | undefined): string {
  if (!value) return "未设置";
  if (value.length <= 8) return "••••••••";
  return value.slice(0, 4) + "••••" + value.slice(-4);
}

// ---------- 平台动态字段描述 ----------

export interface PlatformField {
  key: string;
  label: string;
  placeholder: string;
  type?: "text" | "password";
  required?: boolean;
  multiline?: boolean;
  helperText?: string;
}

export function getPlatformFields(platform: string): PlatformField[] {
  switch (platform) {
    case "openai_compatible":
    case "volcengine":
      return [
        { key: "apiUrl", label: "API 地址", placeholder: "https://api.openai.com（只填根域名）", type: "text" },
        { key: "apiKey", label: "API 密钥", placeholder: "sk-...", type: "password", required: true },
      ];
    case "GoogleFlowReverseApi":
      return [
        { key: "apiUrl", label: "Base URL", placeholder: "http://localhost:8000（只填服务根地址）", type: "text" },
        { key: "apiKey", label: "API 密钥", placeholder: "your-flow2api-key", type: "password", required: true },
      ];
    case "vertex_ai":
      return [
        { key: "appId", label: "项目 ID (Project ID)", placeholder: "my-gcp-project", type: "text", required: true },
        { key: "apiUrl", label: "区域 (Location)", placeholder: "us-central1", type: "text" },
        {
          key: "appSecret",
          label: "服务账号 JSON Key",
          placeholder: '{"type":"service_account",...}',
          type: "text",
          multiline: true,
          helperText: "Vertex AI 还需要认证凭证。留空时依赖服务器上的 ADC；若需显式凭证或用于 Imagen 文生图，可在此填写服务账号 JSON Key。",
        },
      ];
    case "gemini":
      return [
        {
          key: "apiKey",
          label: "Gemini API Key",
          placeholder: "AIza...",
          type: "password",
          required: true,
          helperText: "从 Google AI Studio 获取，用于 Gemini Developer API。这个平台不需要 Project ID 和区域。",
        },
      ];
    case "dashscope":
      return [
        { key: "apiKey", label: "API 密钥", placeholder: "sk-...", type: "password", required: true },
      ];
    case "anthropic":
      return [
        { key: "apiUrl", label: "API 地址", placeholder: "https://api.anthropic.com（只填根域名）", type: "text" },
        { key: "apiKey", label: "API 密钥", placeholder: "sk-ant-...", type: "password", required: true },
      ];
    case "ollama":
      return [
        { key: "apiUrl", label: "服务地址", placeholder: "http://localhost:11434", type: "text" },
      ];
    default:
      return [
        { key: "apiUrl", label: "API 地址", placeholder: "https://...", type: "text" },
        { key: "apiKey", label: "API 密钥", placeholder: "sk-...", type: "password" },
      ];
  }
}
