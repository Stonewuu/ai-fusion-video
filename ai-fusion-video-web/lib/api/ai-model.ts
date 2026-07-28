import { http } from "./client";

export type MultimodalInputType = "image" | "video" | "audio" | "file";
export type MultimodalInputTransport = "url" | "base64";
export type MultimodalInputTransports = Partial<Record<MultimodalInputType, MultimodalInputTransport[]>>;

// ========== 类型定义 ==========

/** AI 模型 */
export interface AiModel {
  id: number;
  name: string;
  code: string;
  modelProtocol: string | null;
  capabilityPresetCode: string | null;
  modelType: number;
  icon: string | null;
  description: string | null;
  sort: number;
  status: number;
  config: string | null;
  maxConcurrency: number | null;
  defaultModel: boolean;
  supportVision: boolean;
  multimodalInputTypes: MultimodalInputType[];
  multimodalInputTransports: MultimodalInputTransports;
  supportReasoning: boolean;
  contextWindow: number | null;
  apiConfigId: number | null;
  createTime: string;
  updateTime: string;
}

/** 文本模型连通性检测结果 */
export interface AiModelConnectivityResult {
  modelId: number;
  modelName: string;
  responseText: string;
  durationMs: number;
  testedAt: string;
}

/** 创建 AI 模型请求 */
export interface AiModelCreateReq {
  name: string;
  code: string;
  modelProtocol?: string;
  capabilityPresetCode?: string;
  modelType: number;
  icon?: string;
  description?: string;
  sort?: number;
  config?: string;
  maxConcurrency?: number;
  defaultModel?: boolean;
  supportVision?: boolean;
  multimodalInputTypes: MultimodalInputType[];
  multimodalInputTransports: MultimodalInputTransports;
  supportReasoning?: boolean;
  contextWindow?: number;
  apiConfigId?: number;
}

/** 更新 AI 模型请求 */
export interface AiModelUpdateReq {
  id: number;
  name?: string;
  code?: string;
  modelProtocol?: string;
  capabilityPresetCode?: string;
  modelType?: number;
  icon?: string;
  description?: string;
  sort?: number;
  status?: number;
  config?: string;
  maxConcurrency?: number;
  defaultModel?: boolean;
  supportVision?: boolean;
  multimodalInputTypes?: MultimodalInputType[];
  multimodalInputTransports?: MultimodalInputTransports;
  supportReasoning?: boolean;
  contextWindow?: number;
  apiConfigId?: number;
}

/** 模型能力预设（从后端 JSON 加载） */
export interface ModelPreset {
  code: string;
  modelCode?: string;
  name: string;
  platform: string;
  modelProtocol?: string | null;
  modelType: number;
  description: string;
  supportReasoning?: boolean;
  contextWindow?: number;
  multimodalInputTypes?: MultimodalInputType[];
  multimodalInputTransports?: MultimodalInputTransports;
  config: Record<string, unknown>;
}

/** 远程 API 返回的可用模型 */
export interface RemoteModel {
  id: string;
  displayName?: string | null;
  ownedBy: string;
  providerPlatform?: string | null;
  modelType?: number | null;
  modelProtocol?: string | null;
  capabilityPresetCode?: string | null;
  inferredMetadata?: boolean | null;
}

/** 分页结果 */
export interface PageResult<T> {
  list: T[];
  total: number;
}

/** AI 模型分页请求 */
export interface AiModelPageReq {
  name?: string;
  code?: string;
  modelType?: number;
  status?: number;
  pageNo?: number;
  pageSize?: number;
}

// ========== API 配置 ==========

/** API 配置 */
export interface ApiConfig {
  id: number;
  name: string;
  platform: string | null;
  textProtocol: string | null;
  imageProtocol: string | null;
  videoProtocol: string | null;
  apiUrl: string | null;
  autoAppendV1Path: boolean;
  proxyType: string | null;
  proxyHost: string | null;
  proxyPort: number | null;
  proxyUsername: string | null;
  proxyPassword: string | null;
  apiKey: string | null;
  appId: string | null;
  appSecret: string | null;
  modelId: number | null;
  status: number;
  remark: string | null;
  createTime: string;
  updateTime: string;
}

/** 保存 API 配置请求（创建/更新共用） */
export interface ApiConfigSaveReq {
  id?: number;
  name: string;
  platform?: string;
  textProtocol?: string;
  imageProtocol?: string;
  videoProtocol?: string;
  apiUrl?: string;
  autoAppendV1Path?: boolean;
  proxyType?: string;
  proxyHost?: string;
  proxyPort?: number;
  proxyUsername?: string;
  proxyPassword?: string;
  apiKey?: string;
  appId?: string;
  appSecret?: string;
  modelId?: number;
  status?: number;
  remark?: string;
}

/** API 配置分页请求 */
export interface ApiConfigPageReq {
  name?: string;
  platform?: string;
  status?: number;
  pageNo?: number;
  pageSize?: number;
}

// ========== 常量 ==========

/** 接入与鉴权类型选项 */
export const PLATFORM_OPTIONS = [
  { value: "openai_compatible", label: "OpenAI / 兼容接入", description: "API Key / Bearer 鉴权；文本、图片、视频协议可分别配置" },
  { value: "deepseek", label: "DeepSeek", description: "DeepSeek 官方 API，支持聊天与推理模型" },
  { value: "newapi", label: "New API", description: "New API 聚合网关，支持远程模型发现与视频任务接口" },
  { value: "volcengine", label: "火山引擎（豆包）", description: "字节跳动火山引擎豆包大模型" },
  { value: "vertex_ai", label: "Google Vertex AI", description: "Google Cloud Vertex AI Gemini" },
  { value: "gemini", label: "Google Gemini API", description: "Google AI Studio / Gemini Developer API" },
  { value: "GoogleFlowReverseApi", label: "Google Flow Reverse API", description: "Flow2API 反向代理，图片/视频 alias 模型" },
  { value: "dashscope", label: "阿里 DashScope", description: "阿里云百炼 / DashScope（通义千问、万相、Qwen-Image）" },
  { value: "anthropic", label: "Anthropic", description: "Claude 系列模型" },
  { value: "ollama", label: "Ollama", description: "本地部署的开源模型" },
] as const;

/** 模型类型选项 */
export const MODEL_TYPE_OPTIONS = [
  { value: 1, label: "对话" },
  { value: 2, label: "图像生成" },
  { value: 3, label: "视频生成" },
  { value: 4, label: "语音合成" },
  { value: 5, label: "语音识别" },
] as const;

/** 模型类型标签映射 */
export const MODEL_TYPE_LABELS: Record<number, string> = {
  1: "对话",
  2: "图像生成",
  3: "视频生成",
  4: "语音合成",
  5: "语音识别",
};

/** 接入与鉴权类型标签映射 */
export const PLATFORM_LABELS: Record<string, string> = {
  openai_compatible: "OpenAI / 兼容接入",
  openai: "OpenAI / 兼容接入",
  newapi: "New API",
  deepseek: "DeepSeek",
  volcengine: "火山引擎",
  zhipu: "智谱",
  moonshot: "Moonshot",
  siliconflow: "硅基流动",
  vertex_ai: "Vertex AI",
  vertexai: "Vertex AI",
  GoogleFlowReverseApi: "Google Flow Reverse API",
  gemini: "Gemini",
  dashscope: "DashScope",
  anthropic: "Anthropic",
  ollama: "Ollama",
};

// ========== API ==========

export const aiModelApi = {
  /** 获取 AI 模型详情 */
  get: (id: number) => http.get<never, AiModel>(`/api/ai/model/get?id=${id}`),

  /** 获取启用的 AI 模型列表 */
  list: () => http.get<never, AiModel[]>("/api/ai/model/list"),

  /** 按类型获取 AI 模型列表 */
  listByType: (type: number) =>
    http.get<never, AiModel[]>(`/api/ai/model/list-by-type?type=${type}`),

  /** AI 模型分页列表 */
  page: (params: AiModelPageReq) => {
    const query = new URLSearchParams();
    if (params.name) query.set("name", params.name);
    if (params.code) query.set("code", params.code);
    if (params.modelType !== undefined) query.set("modelType", String(params.modelType));
    if (params.status !== undefined) query.set("status", String(params.status));
    query.set("pageNo", String(params.pageNo ?? 1));
    query.set("pageSize", String(params.pageSize ?? 10));
    return http.get<never, PageResult<AiModel>>(`/api/ai/model/page?${query.toString()}`);
  },

  /** 创建 AI 模型 */
  create: (data: AiModelCreateReq) =>
    http.post<never, number>("/api/ai/model/create", data),

  /** 更新 AI 模型 */
  update: (data: AiModelUpdateReq) =>
    http.put<never, boolean>("/api/ai/model/update", data),

  /** 删除 AI 模型 */
  delete: (id: number) =>
    http.delete<never, boolean>(`/api/ai/model/delete?id=${id}`),

  /** 检测文本模型连通性 */
  testTextConnectivity: (id: number) =>
    http.post<never, AiModelConnectivityResult>(`/api/ai/model/test-text-connectivity?id=${id}`),

  /** 获取模型能力预设列表 */
  presets: (type?: number) =>
    http.get<never, ModelPreset[]>(
      type !== undefined ? `/api/ai/model/presets?type=${type}` : `/api/ai/model/presets`
    ),
};

export const apiConfigApi = {
  /** 获取 API 配置详情 */
  get: (id: number) => http.get<never, ApiConfig>(`/api/ai/api-config/get?id=${id}`),

  /** 获取启用的 API 配置列表 */
  list: () => http.get<never, ApiConfig[]>("/api/ai/api-config/list"),

  /** API 配置分页列表 */
  page: (params: ApiConfigPageReq) => {
    const query = new URLSearchParams();
    if (params.name) query.set("name", params.name);
    if (params.platform) query.set("platform", params.platform);
    if (params.status !== undefined) query.set("status", String(params.status));
    query.set("pageNo", String(params.pageNo ?? 1));
    query.set("pageSize", String(params.pageSize ?? 10));
    return http.get<never, PageResult<ApiConfig>>(`/api/ai/api-config/page?${query.toString()}`);
  },

  /** 创建 API 配置 */
  create: (data: ApiConfigSaveReq) =>
    http.post<never, number>("/api/ai/api-config/create", data),

  /** 更新 API 配置 */
  update: (data: ApiConfigSaveReq) =>
    http.put<never, boolean>("/api/ai/api-config/update", data),

  /** 删除 API 配置 */
  delete: (id: number) =>
    http.delete<never, boolean>(`/api/ai/api-config/delete?id=${id}`),

  /** 获取远程可用模型列表 */
  remoteModels: (id: number) =>
    http.get<never, RemoteModel[]>(`/api/ai/api-config/remote-models?id=${id}`),
};
