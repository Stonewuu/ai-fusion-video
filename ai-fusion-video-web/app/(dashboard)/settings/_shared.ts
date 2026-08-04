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

// 设置模块统一的文字层级。状态色仍由具体业务语义决定，常规文字只使用
// foreground / muted-foreground，避免页面之间出现不同的灰阶与字号体系。
export const settingsTypography = {
  pageTitle: "text-2xl font-bold tracking-tight text-foreground",
  pageDescription: "mt-1 text-sm leading-relaxed text-muted-foreground",
  sectionTitle: "text-sm font-semibold text-foreground",
  sectionDescription: "mt-1 text-xs leading-relaxed text-muted-foreground",
  fieldLabel: "text-xs font-medium text-muted-foreground",
  body: "text-sm leading-relaxed text-foreground",
  metadata: "text-xs leading-relaxed text-muted-foreground",
  metric: "text-lg font-semibold tracking-tight text-foreground tabular-nums",
} as const;

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
    case "deepseek":
      return [
        { key: "apiUrl", label: "API 地址", placeholder: "https://api.deepseek.com", type: "text", helperText: "留空时使用 DeepSeek 官方地址。" },
        { key: "apiKey", label: "API 密钥", placeholder: "sk-...", type: "password", required: true },
      ];
    case "openai_compatible":
      return [
        {
          key: "apiUrl",
          label: "API 地址",
          placeholder: "https://api.openai.com（只填服务根地址）",
          type: "text",
          helperText: "这里配置连接与鉴权。文本、图片、视频可在 API 配置中分别设置默认协议，特殊模型再单独覆盖。",
        },
        { key: "apiKey", label: "API 密钥", placeholder: "sk-...", type: "password", required: true },
      ];
    case "agnes":
      return [
        {
          key: "apiUrl",
          label: "API 地址",
          placeholder: "https://apihub.agnes-ai.com（只填服务根地址）",
          type: "text",
          helperText: "Agnes 开放接口根地址。文本走 OpenAI 兼容协议，图片/视频默认使用 Agnes 协议。",
        },
        { key: "apiKey", label: "API 密钥", placeholder: "sk-...", type: "password", required: true },
      ];
    case "volcengine":
      return [
        { key: "apiUrl", label: "API 地址", placeholder: "https://ark.cn-beijing.volces.com", type: "text" },
        { key: "apiKey", label: "API 密钥", placeholder: "sk-...", type: "password", required: true },
      ];
    case "newapi":
      return [
        {
          key: "apiUrl",
          label: "API 地址",
          placeholder: "https://docs.newapi.ai（或你的 new api 服务根地址）",
          type: "text",
          required: true,
          helperText: "只填服务根地址即可；系统会自动调用 /v1/models 和 /v1/video/generations。",
        },
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
        {
          key: "apiUrl",
          label: "API 地址",
          placeholder: "https://dashscope.aliyuncs.com（北京，留空使用默认）",
          type: "text",
          helperText: "只填地域根地址即可：文本模型会自动使用 /compatible-mode/v1，图片/视频生成会自动使用 /api/v1。",
        },
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
