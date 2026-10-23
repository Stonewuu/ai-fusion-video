"use client";

import { cn } from "@/lib/utils";
import type { AiModel, ApiConfig, ModelPreset } from "@/lib/api/ai-model";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// ============================================================
// 模型配置可视化表单
// ============================================================

/** 解析 config JSON 字符串为对象，解析失败返回空对象 */
export function parseConfigJson(
  json: string | undefined | null,
): Record<string, unknown> {
  if (!json) {
    return {};
  }
  try {
    const parsed = JSON.parse(json);
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

export function isConfigRecord(
  value: unknown,
): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function mergeConfigObjects(
  baseConfig: Record<string, unknown>,
  overrideConfig: Record<string, unknown>,
): Record<string, unknown> {
  // 与后端 JSONObject 合并语义保持一致：模型配置按顶层字段覆盖预设。
  return { ...baseConfig, ...overrideConfig };
}

export function stringifyConfigObject(config: Record<string, unknown>): string {
  return Object.keys(config).length > 0 ? JSON.stringify(config) : "";
}

export { COMMON_ASPECT_RATIOS, COMMON_TIERS } from "./model-config-options";

export const OPENAI_REASONING_PLATFORMS = new Set([
  "openai_compatible",
  "newapi",
  "deepseek",
  "zhipu",
  "moonshot",
  "volcengine",
  "siliconflow",
]);

const GEMINI_REASONING_PLATFORMS = new Set(["gemini", "vertex_ai", "vertexai"]);

export function normalizePlatform(platform: string | null | undefined): string {
  const normalized = (platform || "").toLowerCase();
  return normalized === "openai" ? "openai_compatible" : normalized;
}

export function normalizeProtocol(protocol: string | null | undefined): string {
  return (protocol || "")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
}

export function isOpenAiReasoningPlatform(
  platform: string | null | undefined,
): boolean {
  return OPENAI_REASONING_PLATFORMS.has(normalizePlatform(platform));
}

export function isAnthropicReasoningPlatform(
  platform: string | null | undefined,
): boolean {
  return normalizePlatform(platform) === "anthropic";
}

export function isDashScopeReasoningPlatform(
  platform: string | null | undefined,
): boolean {
  return normalizePlatform(platform) === "dashscope";
}

export function isGeminiReasoningPlatform(
  platform: string | null | undefined,
): boolean {
  return GEMINI_REASONING_PLATFORMS.has(normalizePlatform(platform));
}

export function formatSemanticLabel(
  value: string | null | undefined,
  labels: Record<string, string>,
): string {
  if (!value) {
    return "未声明";
  }
  return labels[value] || value;
}

export const REASONING_CONFIG_KEYS = [
  "includeReasoning",
  "include_reasoning",
  "reasoningEffort",
  "reasoning_effort",
  "thinkingBudget",
  "thinking_budget",
  "thinking",
];

export const OPENAI_RESPONSE_MODE_CONFIG_KEYS = [
  "apiMode",
  "api_mode",
  "openaiApiMode",
  "openai_api_mode",
  "responseApi",
  "responsesApi",
  "useResponsesApi",
  "useResponses",
];

export const OPENAI_IMAGE_MODE_CONFIG_KEYS = [
  "openaiImageApiMode",
  "openai_image_api_mode",
  "imageApiMode",
  "image_api_mode",
  "useResponsesImageApi",
  "useResponseImageApi",
  "responsesImageApi",
];
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
  { value: "minimax_h3", label: "MiniMax H3 官方协议" },
  { value: "metaso_video", label: "秘塔视频协议" },
  { value: "infistar_video", label: "无限星河视频协议" },
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
): ReadonlyArray<{ value: string; label: string }> {
  switch (modelType) {
    case 1:
      return TEXT_PROTOCOL_OPTIONS;
    case 2:
      return IMAGE_PROTOCOL_OPTIONS;
    case 3:
      return VIDEO_PROTOCOL_OPTIONS;
    default:
      return [];
  }
}

export function getApiConfigDefaultProtocol(
  config: ApiConfig | null | undefined,
  modelType: number,
): string {
  if (!config) {
    return "";
  }
  switch (modelType) {
    case 1:
      return config.textProtocol || "";
    case 2:
      return config.imageProtocol || "";
    case 3:
      return config.videoProtocol || "";
    default:
      return "";
  }
}

export function supportsReasoningConfig(
  platform: string | null | undefined,
): boolean {
  return (
    isOpenAiReasoningPlatform(platform) ||
    isAnthropicReasoningPlatform(platform) ||
    isDashScopeReasoningPlatform(platform) ||
    isGeminiReasoningPlatform(platform)
  );
}

export function supportsOpenAiResponsesMode(
  platform: string | null | undefined,
): boolean {
  return isOpenAiReasoningPlatform(platform);
}

export function stripConfigKeys(
  configJson: string | undefined,
  keys: readonly string[],
): string {
  const next = { ...parseConfigJson(configJson) };
  keys.forEach((key) => {
    delete next[key];
  });
  return Object.keys(next).length > 0 ? JSON.stringify(next) : "";
}

export function stripReasoningConfig(configJson: string | undefined): string {
  return stripConfigKeys(configJson, REASONING_CONFIG_KEYS);
}

export function stripOpenAiResponsesModeConfig(
  configJson: string | undefined,
): string {
  return stripConfigKeys(configJson, OPENAI_RESPONSE_MODE_CONFIG_KEYS);
}

export function stripOpenAiImageModeConfig(
  configJson: string | undefined,
): string {
  return stripConfigKeys(configJson, OPENAI_IMAGE_MODE_CONFIG_KEYS);
}

export function sanitizeModelConfigJson({
  configJson,
  modelType,
  platform,
  supportReasoning,
}: {
  configJson: string | undefined;
  modelType: number;
  platform: string | null | undefined;
  supportReasoning: boolean;
}): string {
  let next = configJson || "";

  if (
    !(modelType === 1 && supportReasoning && supportsReasoningConfig(platform))
  ) {
    next = stripReasoningConfig(next);
  }
  if (!(modelType === 1 && supportsOpenAiResponsesMode(platform))) {
    next = stripOpenAiResponsesModeConfig(next);
  }
  next = stripOpenAiImageModeConfig(next);

  return next;
}

export function getPositiveNumberValue(value: number | undefined): number | "" {
  return value !== undefined && value > 0 ? value : "";
}

export function getConfigBooleanValue(value: unknown): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    return value === "true";
  }
  return false;
}

export function getConfigNumberValue(value: unknown): number | "" {
  if (typeof value === "number" && !Number.isNaN(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? "" : parsed;
  }
  return "";
}

export function getOptionalConfigNumber(value: unknown): number | undefined {
  const numericValue = getConfigNumberValue(value);
  return typeof numericValue === "number" ? numericValue : undefined;
}

export function getConfigStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is string =>
          typeof item === "string" && item.trim().length > 0,
      )
    : [];
}

export function getFirstConfigString(
  value: Record<string, unknown>,
  keys: readonly string[],
): string | undefined {
  for (const key of keys) {
    const current = value[key];
    if (typeof current === "string" && current.trim().length > 0) {
      return current;
    }
  }
  return undefined;
}

export interface CapabilityChipDef {
  label: string;
  tone: "positive" | "muted" | "info";
}

export interface GenerationCapabilityView {
  chips: CapabilityChipDef[];
  summary: string;
}

export function isEquivalentPresetPlatform(
  presetPlatform: string | null | undefined,
  selectedPlatform: string | null | undefined,
): boolean {
  if (!selectedPlatform) {
    return true;
  }
  if (!presetPlatform) {
    return false;
  }
  return (
    normalizePlatform(presetPlatform) === normalizePlatform(selectedPlatform)
  );
}

export function isCapabilityPresetCompatible({
  preset,
  modelType,
}: {
  preset: ModelPreset;
  modelType: number;
  platform: string | null | undefined;
  effectiveProtocol: string | null | undefined;
}): boolean {
  return preset.modelType === modelType;
}

export function findCapabilityPreset(
  model: AiModel,
  presets: ModelPreset[],
): ModelPreset | null {
  if (!model.capabilityPresetCode) {
    return null;
  }
  return (
    presets.find(
      (preset) =>
        preset.code === model.capabilityPresetCode &&
        preset.modelType === model.modelType,
    ) || null
  );
}

export function getCapabilityChipClassName(
  tone: CapabilityChipDef["tone"],
): string {
  switch (tone) {
    case "positive":
      return "bg-emerald-500/10 text-emerald-500 border-emerald-500/20";
    case "info":
      return "bg-sky-500/10 text-sky-500 border-sky-500/20";
    default:
      return "bg-muted/60 text-muted-foreground border-border/40";
  }
}

export function buildImageCapabilityView(
  config: Record<string, unknown>,
): GenerationCapabilityView {
  const supportsReferenceImages = getConfigBooleanValue(
    config.supportReferenceImages,
  );
  const asyncMode = getConfigBooleanValue(config.asyncMode);
  const maxReferenceImages = getOptionalConfigNumber(config.maxReferenceImages);
  const aspectRatioCount = getConfigStringArray(
    config.supportedAspectRatios,
  ).length;
  const supportedSizes = isConfigRecord(config.supportedSizes)
    ? Object.keys(config.supportedSizes).length
    : 0;

  const chips: CapabilityChipDef[] = [
    supportsReferenceImages
      ? {
          label:
            maxReferenceImages && maxReferenceImages > 0
              ? `参考图 ≤${maxReferenceImages}`
              : "支持参考图",
          tone: "positive",
        }
      : { label: "仅文生图", tone: "muted" },
  ];

  if (aspectRatioCount > 0) {
    chips.push({ label: `${aspectRatioCount} 种比例`, tone: "info" });
  }
  if (supportedSizes > 0) {
    chips.push({ label: `${supportedSizes} 个尺寸档`, tone: "info" });
  }
  if (asyncMode) {
    chips.push({ label: "异步任务", tone: "positive" });
  }

  return {
    chips,
    summary: supportsReferenceImages
      ? `支持参考图输入${maxReferenceImages && maxReferenceImages > 0 ? `，最多 ${maxReferenceImages} 张` : ""}${asyncMode ? "；已开启异步任务模式" : ""}`
      : `不支持参考图输入，当前模型只适合文生图。${asyncMode ? "已开启异步任务模式。" : ""}`,
  };
}

export function buildVideoCapabilityView(
  config: Record<string, unknown>,
): GenerationCapabilityView {
  const supportsFirstFrame = getConfigBooleanValue(config.supportFirstFrame);
  const supportsLastFrame = getConfigBooleanValue(config.supportLastFrame);
  const supportsReferenceImages = getConfigBooleanValue(
    config.supportReferenceImages,
  );
  const supportsReferenceVideos = getConfigBooleanValue(
    config.supportReferenceVideos,
  );
  const supportsReferenceAudios = getConfigBooleanValue(
    config.supportReferenceAudios,
  );
  const maxImageInputs = getOptionalConfigNumber(config.maxImageInputs);
  const maxReferenceImages = getOptionalConfigNumber(config.maxReferenceImages);
  const maxReferenceVideos = getOptionalConfigNumber(config.maxReferenceVideos);
  const maxReferenceAudios = getOptionalConfigNumber(config.maxReferenceAudios);

  const summaryParts = [
    supportsFirstFrame ? "首帧" : null,
    supportsLastFrame ? "尾帧" : null,
    supportsReferenceImages
      ? `参考图${maxReferenceImages && maxReferenceImages > 0 ? `（最多 ${maxReferenceImages} 张）` : ""}`
      : null,
    supportsReferenceVideos
      ? `参考视频${maxReferenceVideos && maxReferenceVideos > 0 ? `（最多 ${maxReferenceVideos} 个）` : ""}`
      : null,
    supportsReferenceAudios
      ? `参考音频${maxReferenceAudios && maxReferenceAudios > 0 ? `（最多 ${maxReferenceAudios} 个）` : ""}`
      : null,
  ].filter((part) => part !== null);
  const imageLimit = maxImageInputs && maxImageInputs > 0
    ? `；图片输入最多 ${maxImageInputs} 张`
    : "";

  return {
    chips: [],
    summary: `该渠道模型能力仅支持${summaryParts.length ? summaryParts.join("、") : "文生视频"}${imageLimit}。`,
  };
}

export function buildGenerationCapabilityView(
  model: AiModel,
  config: Record<string, unknown>,
): GenerationCapabilityView | null {
  if (model.modelType === 2) {
    return buildImageCapabilityView(config);
  }
  if (model.modelType === 3) {
    return buildVideoCapabilityView(config);
  }
  return null;
}

export function getChatSamplingFields(
  platform: string | null | undefined,
): ConfigFieldDef[] {
  const normalized = normalizePlatform(platform);
  const fields: ConfigFieldDef[] = [
    {
      key: "temperature",
      label: "Temperature",
      type: "range",
      min: 0,
      max: 2,
      step: 0.1,
      defaultValue: 0.7,
      hint: "控制输出随机性，值越高越随机",
    },
  ];

  if (
    normalized === "vertex_ai" ||
    normalized === "vertexai" ||
    normalized === "gemini"
  ) {
    return fields;
  }

  fields.push({
    key: "topP",
    label: "Top P",
    type: "range",
    min: 0,
    max: 1,
    step: 0.05,
    defaultValue: 1,
    hint: "核心采样概率阈值",
  });

  if (normalized === "ollama") {
    return fields;
  }

  fields.push({
    key: "maxTokens",
    label: "Max Tokens",
    type: "number",
    min: 1,
    max: 1000000,
    step: 1,
    placeholder: "例如：4096",
    hint: "单次请求最大输出 token 数",
  });
  return fields;
}

// ---------- supportedSizes 编辑器 ----------

export { SupportedSizesEditor, type SizesMap } from "./model-config-sizes-editor";
export { AspectRatiosEditor } from "./model-config-aspect-ratios-editor";

// ---------- ModelConfigForm 主组件 ----------

export interface ConfigFieldDef {
  key: string;
  label: string;
  type: "number" | "range" | "supported-sizes" | "aspect-ratios";
  min?: number;
  max?: number;
  step?: number;
  defaultValue?: number;
  placeholder?: string;
  hint?: string;
  presetOptions?: string[];
}

export function getConfigFieldsByModelType(
  modelType: number,
  platform?: string | null,
): ConfigFieldDef[] {
  switch (modelType) {
    case 1:
      return getChatSamplingFields(platform);
    case 2:
      return [
        {
          key: "defaultWidth",
          label: "默认宽度",
          type: "number",
          min: 256,
          max: 8192,
          step: 64,
          placeholder: "例如：1024",
        },
        {
          key: "defaultHeight",
          label: "默认高度",
          type: "number",
          min: 256,
          max: 8192,
          step: 64,
          placeholder: "例如：1024",
        },
        {
          key: "minPixels",
          label: "最小像素数",
          type: "number",
          min: 0,
          step: 1,
          placeholder: "例如：921600",
          hint: "生成图像的最小总像素数限制",
        },
        {
          key: "maxPixels",
          label: "最大像素数",
          type: "number",
          min: 0,
          step: 1,
          placeholder: "例如：16777216",
          hint: "生成图像的最大总像素数限制",
        },
        { key: "supportedSizes", label: "支持的尺寸", type: "supported-sizes" },
        {
          key: "supportedAspectRatios",
          label: "支持的比例",
          type: "aspect-ratios",
        },
      ];
    case 3:
      return [
        {
          key: "supportedResolutions",
          label: "支持的分辨率",
          type: "aspect-ratios",
          hint: "如 480p, 720p, 1080p",
          presetOptions: ["480p", "720p", "1080p", "2K", "4K"],
        },
        {
          key: "supportedAspectRatios",
          label: "支持的宽高比",
          type: "aspect-ratios",
          hint: "如 16:9, 9:16, 1:1",
        },
        {
          key: "minDuration",
          label: "最短时长（秒）",
          type: "number",
          min: 1,
          max: 60,
          step: 1,
          placeholder: "例如：4",
        },
        {
          key: "maxDuration",
          label: "最长时长（秒）",
          type: "number",
          min: 1,
          max: 60,
          step: 1,
          placeholder: "例如：15",
        },
        {
          key: "defaultDuration",
          label: "默认时长（秒）",
          type: "number",
          min: 1,
          max: 60,
          step: 1,
          placeholder: "例如：5",
        },
      ];
    default:
      return [];
  }
}

export function ToggleSettingCard({
  checked,
  title,
  description,
  onToggle,
  disabled = false,
}: {
  checked: boolean;
  title: string;
  description: string;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border/30 bg-background/70 px-3 py-2.5",
        disabled && "opacity-50",
      )}
    >
      <div className="flex items-start gap-3">
        <button
          type="button"
          onClick={onToggle}
          disabled={disabled}
          className={cn(
            "relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors duration-200",
            checked ? "bg-primary" : "bg-muted-foreground/30",
          )}
        >
          <span
            className={cn(
              "absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
              checked && "translate-x-4",
            )}
          />
        </button>
        <div className="min-w-0">
          <Label
            className={cn(
              "text-xs text-muted-foreground",
              !disabled && "cursor-pointer",
            )}
            onClick={() => {
              if (!disabled) {
                onToggle();
              }
            }}
          >
            {title}
          </Label>
          <p className="mt-1 text-[10px] leading-5 text-muted-foreground/70">
            {description}
          </p>
        </div>
      </div>
    </div>
  );
}

export function CapabilityNumberField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  min,
  max,
  step = 1,
  disabled = false,
}: {
  label: string;
  value: number | "";
  onChange: (value: string) => void;
  placeholder?: string;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <div className={cn("space-y-1.5", disabled && "opacity-55")}>
      <Label className="text-[11px] text-muted-foreground">{label}</Label>
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-xs font-mono h-8"
      />
      {hint && <p className="text-[10px] text-muted-foreground/70">{hint}</p>}
    </div>
  );
}
