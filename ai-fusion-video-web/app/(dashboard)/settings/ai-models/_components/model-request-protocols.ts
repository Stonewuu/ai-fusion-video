export const MODEL_PROTOCOL_LABELS: Record<string, string> = {
  openai_compatible: "OpenAI 兼容文本协议",
  openai: "OpenAI 官方协议",
  deepseek: "OpenAI 兼容文本协议",
  anthropic: "Anthropic 协议",
  gemini: "Gemini 协议",
  newapi: "New API 协议",
  dashscope: "DashScope 协议",
  dashscope_wan3: "万相 3.0 官方协议",
  minimax_h3: "MiniMax H3 官方协议",
  metaso_video: "秘塔视频协议",
  infistar_video: "无限星河视频协议",
  volcengine: "火山引擎协议",
  vertex_ai: "Vertex AI 协议",
  google_flow: "Flow 协议",
  agnes: "Agnes 协议",
  agnes_25: "Agnes 视频 2.5 协议",
  jimeng: "即梦协议",
  kling: "可灵协议",
  sora: "Sora 协议",
  comfyui: "ComfyUI Native API",
};

export const TEXT_PROTOCOL_OPTIONS = [
  { value: "openai_compatible", label: "OpenAI 兼容文本协议" },
  { value: "anthropic", label: "Anthropic 协议" },
  { value: "gemini", label: "Gemini 协议" },
  { value: "dashscope", label: "DashScope 协议" },
  { value: "volcengine", label: "火山引擎协议" },
  { value: "vertex_ai", label: "Vertex AI 协议" },
  { value: "ollama", label: "Ollama 协议" },
] as const;

export const IMAGE_PROTOCOL_OPTIONS = [
  { value: "gemini", label: "Gemini 协议" },
  { value: "openai", label: "OpenAI 官方协议" },
  { value: "newapi", label: "New API 协议" },
  { value: "agnes", label: "Agnes 协议" },
  { value: "dashscope", label: "DashScope 协议" },
  { value: "volcengine", label: "火山引擎协议" },
  { value: "vertex_ai", label: "Vertex AI 协议" },
  { value: "google_flow", label: "Flow 协议" },
  { value: "comfyui", label: "ComfyUI Native API" },
] as const;

export const VIDEO_PROTOCOL_OPTIONS = [
  { value: "gemini", label: "Gemini 协议" },
  { value: "vertex_ai", label: "Vertex AI 协议" },
  { value: "minimax_h3", label: "MiniMax H3 官方协议" },
  { value: "metaso_video", label: "秘塔视频协议", providerOnly: true },
  { value: "infistar_video", label: "无限星河视频协议", providerOnly: true },
  { value: "dashscope_wan3", label: "万相 3.0 官方协议" },
  { value: "openai", label: "OpenAI 官方协议" },
  { value: "newapi", label: "New API 通用协议" },
  { value: "agnes_25", label: "Agnes 视频 2.5 协议" },
  { value: "agnes", label: "Agnes 视频 2.0 协议" },
  { value: "jimeng", label: "即梦协议" },
  { value: "kling", label: "可灵协议" },
  { value: "sora", label: "Sora 协议" },
  { value: "dashscope", label: "DashScope 协议" },
  { value: "volcengine", label: "火山引擎协议" },
  { value: "google_flow", label: "Flow 协议" },
  { value: "comfyui", label: "ComfyUI Native API" },
] as const;

export const CUSTOM_CAPABILITY_PRESET_VALUE = "__custom_capabilities__";
export const INHERIT_PROVIDER_PROTOCOL_VALUE = "__provider_default__";

export function getProtocolOptionsForModelType(
  modelType: number,
  providerId?: string | null,
  providers: ReadonlyArray<{ id: string; restrictedProtocols: readonly string[] }> = [],
): ReadonlyArray<{ value: string; label: string }> {
  if (providers.length === 0) {
    return [];
  }
  const restrictedOwners = new Map(providers.flatMap(provider =>
    provider.restrictedProtocols.map(protocol => [protocol, provider.id] as const)));
  const allowed = (option: { value: string; providerOnly?: boolean }) =>
    (!option.providerOnly && !restrictedOwners.has(option.value)) ||
    (!!providerId && restrictedOwners.get(option.value) === providerId);
  switch (modelType) {
    case 1:
      return TEXT_PROTOCOL_OPTIONS.filter(allowed);
    case 2:
      return IMAGE_PROTOCOL_OPTIONS.filter(allowed);
    case 3:
      return VIDEO_PROTOCOL_OPTIONS.filter(allowed);
    case 4:
      return [
        { value: "gemini", label: "Gemini 协议" },
        { value: "vertex_ai", label: "Vertex AI 协议" },
      ];
    default:
      return [];
  }
}
