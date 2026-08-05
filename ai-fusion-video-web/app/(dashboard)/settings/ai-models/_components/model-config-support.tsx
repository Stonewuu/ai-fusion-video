"use client";

import { useState } from "react";
import { Check, ChevronRight, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AiModel, ApiConfig, ModelPreset } from "@/lib/api/ai-model";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// ============================================================
// 模型配置可视化表单
// ============================================================

/** 解析 config JSON 字符串为对象，解析失败返回空对象 */
export function parseConfigJson(json: string | undefined | null): Record<string, unknown> {
  if (!json) return {};
  try {
    const parsed = JSON.parse(json);
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

export function isConfigRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function mergeConfigObjects(baseConfig: Record<string, unknown>, overrideConfig: Record<string, unknown>): Record<string, unknown> {
  // 与后端 JSONObject 合并语义保持一致：模型配置按顶层字段覆盖预设。
  return { ...baseConfig, ...overrideConfig };
}

export function stringifyConfigObject(config: Record<string, unknown>): string {
  return Object.keys(config).length > 0 ? JSON.stringify(config) : "";
}

/** 常见宽高比 */
export const COMMON_ASPECT_RATIOS = ["1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2", "21:9"];

/** 常见分辨率档位 */
export const COMMON_TIERS = ["1K", "2K", "3K", "4K"];

export const OPENAI_REASONING_PLATFORMS = new Set([
  "openai_compatible",
  "agnes",
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
  return (protocol || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
}

export function isOpenAiReasoningPlatform(platform: string | null | undefined): boolean {
  return OPENAI_REASONING_PLATFORMS.has(normalizePlatform(platform));
}

export function isAnthropicReasoningPlatform(platform: string | null | undefined): boolean {
  return normalizePlatform(platform) === "anthropic";
}

export function isDashScopeReasoningPlatform(platform: string | null | undefined): boolean {
  return normalizePlatform(platform) === "dashscope";
}

export function isGeminiReasoningPlatform(platform: string | null | undefined): boolean {
  return GEMINI_REASONING_PLATFORMS.has(normalizePlatform(platform));
}

export function formatSemanticLabel(value: string | null | undefined, labels: Record<string, string>): string {
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
  volcengine: "火山引擎协议",
  vertex_ai: "Vertex AI 协议",
  google_flow: "Flow 协议",
  agnes: "Agnes 协议",
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
  { value: "openai", label: "OpenAI 官方协议" },
  { value: "newapi", label: "New API 通用协议" },
  { value: "agnes", label: "Agnes 协议" },
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

export function getProtocolOptionsForModelType(modelType: number): ReadonlyArray<{ value: string; label: string }> {
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

export function getApiConfigDefaultProtocol(config: ApiConfig | null | undefined, modelType: number): string {
  if (!config) return "";
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

export function supportsReasoningConfig(platform: string | null | undefined): boolean {
  return (
    isOpenAiReasoningPlatform(platform) ||
    isAnthropicReasoningPlatform(platform) ||
    isDashScopeReasoningPlatform(platform) ||
    isGeminiReasoningPlatform(platform)
  );
}
export function supportsOpenAiResponsesMode(platform: string | null | undefined): boolean {
  return isOpenAiReasoningPlatform(platform);
}

export function stripConfigKeys(configJson: string | undefined, keys: readonly string[]): string {
  const next = { ...parseConfigJson(configJson) };
  keys.forEach((key) => {
    delete next[key];
  });
  return Object.keys(next).length > 0 ? JSON.stringify(next) : "";
}

export function stripReasoningConfig(configJson: string | undefined): string {
  return stripConfigKeys(configJson, REASONING_CONFIG_KEYS);
}

export function stripOpenAiResponsesModeConfig(configJson: string | undefined): string {
  return stripConfigKeys(configJson, OPENAI_RESPONSE_MODE_CONFIG_KEYS);
}

export function stripOpenAiImageModeConfig(configJson: string | undefined): string {
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

  if (!(modelType === 1 && supportReasoning && supportsReasoningConfig(platform))) {
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
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return value === "true";
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
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0) : [];
}

export function getFirstConfigString(value: Record<string, unknown>, keys: readonly string[]): string | undefined {
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
  selectedPlatform: string | null | undefined
): boolean {
  if (!selectedPlatform) {
    return true;
  }
  if (!presetPlatform) {
    return false;
  }
  return normalizePlatform(presetPlatform) === normalizePlatform(selectedPlatform);
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

export function findCapabilityPreset(model: AiModel, presets: ModelPreset[]): ModelPreset | null {
  if (!model.capabilityPresetCode) {
    return null;
  }
  return presets.find(preset =>
    preset.code === model.capabilityPresetCode && preset.modelType === model.modelType
  ) || null;
}

export function getCapabilityChipClassName(tone: CapabilityChipDef["tone"]): string {
  switch (tone) {
    case "positive":
      return "bg-emerald-500/10 text-emerald-500 border-emerald-500/20";
    case "info":
      return "bg-sky-500/10 text-sky-500 border-sky-500/20";
    default:
      return "bg-muted/60 text-muted-foreground border-border/40";
  }
}

export function buildImageCapabilityView(config: Record<string, unknown>): GenerationCapabilityView {
  const supportsReferenceImages = getConfigBooleanValue(config.supportReferenceImages);
  const asyncMode = getConfigBooleanValue(config.asyncMode);
  const maxReferenceImages = getOptionalConfigNumber(config.maxReferenceImages);
  const aspectRatioCount = getConfigStringArray(config.supportedAspectRatios).length;
  const supportedSizes = isConfigRecord(config.supportedSizes) ? Object.keys(config.supportedSizes).length : 0;

  const chips: CapabilityChipDef[] = [supportsReferenceImages
    ? { label: maxReferenceImages && maxReferenceImages > 0 ? `参考图 ≤${maxReferenceImages}` : "支持参考图", tone: "positive" }
    : { label: "仅文生图", tone: "muted" }];

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

export function buildVideoCapabilityView(config: Record<string, unknown>): GenerationCapabilityView {
  const supportsFirstFrame = getConfigBooleanValue(config.supportFirstFrame);
  const supportsLastFrame = getConfigBooleanValue(config.supportLastFrame);
  const supportsReferenceImages = getConfigBooleanValue(config.supportReferenceImages);
  const supportsReferenceVideos = getConfigBooleanValue(config.supportReferenceVideos);
  const supportsReferenceAudios = getConfigBooleanValue(config.supportReferenceAudios);
  const minImageInputs = getOptionalConfigNumber(config.minImageInputs);
  const maxImageInputs = getOptionalConfigNumber(config.maxImageInputs);
  const maxReferenceImages = getOptionalConfigNumber(config.maxReferenceImages);
  const maxReferenceVideos = getOptionalConfigNumber(config.maxReferenceVideos);
  const maxReferenceAudios = getOptionalConfigNumber(config.maxReferenceAudios);

  const chips: CapabilityChipDef[] = [
    { label: supportsFirstFrame ? "首帧" : "无首帧", tone: supportsFirstFrame ? "positive" : "muted" },
    { label: supportsLastFrame ? "尾帧" : "无尾帧", tone: supportsLastFrame ? "positive" : "muted" },
    {
      label: supportsReferenceImages
        ? maxReferenceImages && maxReferenceImages > 0 ? `参考图 ≤${maxReferenceImages}` : "参考图"
        : "无参考图",
      tone: supportsReferenceImages ? "positive" : "muted",
    },
    {
      label: supportsReferenceVideos
        ? maxReferenceVideos && maxReferenceVideos > 0 ? `参考视频 ≤${maxReferenceVideos}` : "参考视频"
        : "无参考视频",
      tone: supportsReferenceVideos ? "positive" : "muted",
    },
    {
      label: supportsReferenceAudios
        ? maxReferenceAudios && maxReferenceAudios > 0 ? `参考音频 ≤${maxReferenceAudios}` : "参考音频"
        : "无参考音频",
      tone: supportsReferenceAudios ? "positive" : "muted",
    },
  ];

  if (minImageInputs !== undefined || maxImageInputs !== undefined) {
    const imageInputLabel = minImageInputs !== undefined && maxImageInputs !== undefined
      ? `图输 ${minImageInputs}-${maxImageInputs} 张`
      : minImageInputs !== undefined
        ? `图输 ≥${minImageInputs} 张`
        : `图输 ≤${maxImageInputs} 张`;
    chips.push({ label: imageInputLabel, tone: "info" });
  }

  const summaryParts = [
    supportsFirstFrame ? "支持首帧" : "不支持首帧",
    supportsLastFrame ? "支持尾帧" : "不支持尾帧",
    supportsReferenceImages ? "支持参考图" : "不支持参考图",
    supportsReferenceVideos ? "支持参考视频" : "不支持参考视频",
    supportsReferenceAudios ? "支持参考音频" : "不支持参考音频",
  ];

  return {
    chips,
    summary: summaryParts.join("，") + "。",
  };
}

export function buildGenerationCapabilityView(model: AiModel, config: Record<string, unknown>): GenerationCapabilityView | null {
  if (model.modelType === 2) {
    return buildImageCapabilityView(config);
  }
  if (model.modelType === 3) {
    return buildVideoCapabilityView(config);
  }
  return null;
}

export function getChatSamplingFields(platform: string | null | undefined): ConfigFieldDef[] {
  const normalized = normalizePlatform(platform);
  const fields: ConfigFieldDef[] = [
    { key: "temperature", label: "Temperature", type: "range", min: 0, max: 2, step: 0.1, defaultValue: 0.7, hint: "控制输出随机性，值越高越随机" },
  ];

  if (normalized === "vertex_ai" || normalized === "vertexai" || normalized === "gemini") {
    return fields;
  }

  fields.push({ key: "topP", label: "Top P", type: "range", min: 0, max: 1, step: 0.05, defaultValue: 1, hint: "核心采样概率阈值" });

  if (normalized === "ollama") {
    return fields;
  }

  fields.push({ key: "maxTokens", label: "Max Tokens", type: "number", min: 1, max: 1000000, step: 1, placeholder: "例如：4096", hint: "单次请求最大输出 token 数" });
  return fields;
}

// ---------- supportedSizes 编辑器 ----------

export type SizesMap = Record<string, Record<string, string>>;

export function SupportedSizesEditor({
  value,
  onChange,
}: {
  value: SizesMap | undefined;
  onChange: (v: SizesMap | undefined) => void;
}) {
  const [collapsedTiers, setCollapsedTiers] = useState<Set<string>>(
    () => new Set(Object.keys(value || {}).slice(1))
  );
  const [newTierName, setNewTierName] = useState("");

  const sizes = value || {};
  const tierNames = Object.keys(sizes);

  const toggleCollapse = (tier: string) => {
    setCollapsedTiers(prev => {
      const next = new Set(prev);
      if (next.has(tier)) {
        next.delete(tier);
      } else {
        next.add(tier);
      }
      return next;
    });
  };

  const addTier = (tierName: string) => {
    const name = tierName.trim();
    if (!name || sizes[name]) return;
    const next = { ...sizes, [name]: {} };
    onChange(next);
    setCollapsedTiers(prev => {
      const collapsed = new Set(prev);
      collapsed.delete(name);
      return collapsed;
    });
    setNewTierName("");
  };

  const removeTier = (tier: string) => {
    const next = { ...sizes };
    delete next[tier];
    onChange(Object.keys(next).length > 0 ? next : undefined);
  };

  const addRatio = (tier: string, ratio: string) => {
    if (!ratio.trim()) return;
    const next = { ...sizes, [tier]: { ...sizes[tier], [ratio.trim()]: "" } };
    onChange(next);
  };

  const updateResolution = (tier: string, ratio: string, resolution: string) => {
    const next = { ...sizes, [tier]: { ...sizes[tier], [ratio]: resolution } };
    onChange(next);
  };

  const removeRatio = (tier: string, ratio: string) => {
    const tierData = { ...sizes[tier] };
    delete tierData[ratio];
    const next = { ...sizes, [tier]: tierData };
    onChange(next);
  };

  const fillCommonRatios = (tier: string) => {
    const tierData = { ...sizes[tier] };
    COMMON_ASPECT_RATIOS.forEach(r => {
      if (!(r in tierData)) tierData[r] = "";
    });
    onChange({ ...sizes, [tier]: tierData });
  };

  const availableTierChips = COMMON_TIERS.filter(t => !sizes[t]);

  return (
    <div className="space-y-3">
      <label className="text-xs text-muted-foreground">支持的尺寸 (supportedSizes)</label>

      {tierNames.map(tier => {
        const isCollapsed = collapsedTiers.has(tier);
        const ratios = sizes[tier] || {};
        const ratioEntries = Object.entries(ratios);

        return (
          <div key={tier} className="overflow-hidden rounded-lg border border-border/20 bg-background/70">
            <div className={cn(
              "flex min-h-10 items-center justify-between gap-3 px-3",
              !isCollapsed && "border-b border-border/20"
            )}>
              <button
                type="button"
                onClick={() => toggleCollapse(tier)}
                className="flex min-w-0 items-center gap-2 rounded-md text-sm font-medium text-foreground/80 outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
              >
                <ChevronRight className={cn("size-4 shrink-0 transition-transform duration-200", !isCollapsed && "rotate-90")} />
                <span className="truncate">{tier}</span>
                <span className="shrink-0 text-xs font-normal text-muted-foreground">{ratioEntries.length} 项</span>
              </button>
              <div className="flex shrink-0 items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  onClick={() => fillCommonRatios(tier)}
                  title="填充常见比例"
                >
                  补齐常见比例
                </Button>
                <Button
                  type="button"
                  variant="destructive-ghost"
                  size="icon-xs"
                  onClick={() => removeTier(tier)}
                  aria-label={`删除 ${tier} 档位`}
                  title="删除档位"
                >
                  <Trash2 />
                </Button>
              </div>
            </div>

            {!isCollapsed && (
              <div className="space-y-2 p-3">
                {ratioEntries.length > 0 ? (
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                    {ratioEntries.map(([ratio, resolution]) => (
                      <div
                        key={ratio}
                        className="group flex h-9 min-w-0 items-center gap-2 rounded-lg border border-border/20 bg-background/70 pl-2 pr-1 transition-colors focus-within:border-border/50"
                      >
                        <span className="w-10 shrink-0 text-right font-mono text-xs text-muted-foreground">{ratio}</span>
                        <Input
                          aria-label={`${tier} ${ratio} 尺寸`}
                          placeholder="1024x1024"
                          value={resolution}
                          onChange={e => updateResolution(tier, ratio, e.target.value)}
                          className="h-7 min-w-0 flex-1 border-0 bg-transparent px-0 font-mono text-xs shadow-none focus-visible:ring-0"
                        />
                        <Button
                          type="button"
                          variant="destructive-ghost"
                          size="icon-xs"
                          onClick={() => removeRatio(tier, ratio)}
                          aria-label={`删除 ${ratio} 比例`}
                          title={`删除 ${ratio}`}
                        >
                          <X />
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg border border-dashed border-border/40 py-4 text-center text-xs text-muted-foreground">
                    暂无尺寸比例
                  </div>
                )}

                {(() => {
                  const existingRatios = new Set(Object.keys(ratios));
                  const available = COMMON_ASPECT_RATIOS.filter(r => !existingRatios.has(r));
                  if (available.length === 0) return null;
                  return (
                    <div className="flex flex-wrap items-center gap-2 border-t border-border/20 pt-2">
                      <span className="text-xs text-muted-foreground">添加比例</span>
                      {available.map(r => (
                        <Button
                          key={r}
                          type="button"
                          variant="ghost"
                          size="xs"
                          onClick={() => addRatio(tier, r)}
                        >
                          + {r}
                        </Button>
                      ))}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        );
      })}

      <div className="flex flex-wrap items-center gap-2">
        {availableTierChips.map(t => (
          <Button
            key={t}
            type="button"
            variant="outline"
            size="xs"
            onClick={() => addTier(t)}
          >
            + {t}
          </Button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <Input
            placeholder="自定义档位"
            value={newTierName}
            onChange={e => setNewTierName(e.target.value)}
            onKeyDown={e => e.key === "Enter" && addTier(newTierName)}
            className="h-8 w-28 font-mono text-xs"
          />
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => addTier(newTierName)}
            disabled={!newTierName.trim()}
          >
            添加
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------- supportedAspectRatios 编辑器 ----------

export function AspectRatiosEditor({
  value,
  onChange,
  label,
  hint,
  presetOptions,
}: {
  value: string[] | undefined;
  onChange: (v: string[] | undefined) => void;
  label?: string;
  hint?: string;
  presetOptions?: string[];
}) {
  const [customRatio, setCustomRatio] = useState("");
  const selected = new Set(value || []);
  const commonOptions = presetOptions || COMMON_ASPECT_RATIOS;
  const selectedValues = Array.from(selected);
  const availableOptions = commonOptions.filter(ratio => !selected.has(ratio));

  const toggle = (ratio: string) => {
    const next = new Set(selected);
    if (next.has(ratio)) {
      next.delete(ratio);
    } else {
      next.add(ratio);
    }
    const arr = Array.from(next);
    onChange(arr.length > 0 ? arr : undefined);
  };

  const addCustom = () => {
    const r = customRatio.trim();
    if (!r || selected.has(r)) return;
    onChange([...Array.from(selected), r]);
    setCustomRatio("");
  };

  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground">{label || "支持的比例 (supportedAspectRatios)"}</label>
        {hint && <p className="text-xs text-muted-foreground/70">{hint}</p>}
      </div>

      {selectedValues.length > 0 ? (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 2xl:grid-cols-7">
          {selectedValues.map(ratio => (
            <div
              key={ratio}
              className="group flex h-9 min-w-0 items-center gap-2 rounded-lg border border-border/20 bg-background/70 pl-2 pr-1 transition-colors focus-within:border-border/50"
            >
              <Check className="size-4 shrink-0 text-primary" />
              <span className="min-w-0 flex-1 truncate font-mono text-xs text-foreground/80">{ratio}</span>
              <Button
                type="button"
                variant="destructive-ghost"
                size="icon-xs"
                onClick={() => toggle(ratio)}
                aria-label={`删除 ${ratio}`}
                title={`删除 ${ratio}`}
              >
                <X />
              </Button>
            </div>
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-border/40 py-4 text-center text-xs text-muted-foreground">
          暂无已选项
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 border-t border-border/20 pt-2">
        {availableOptions.length > 0 && (
          <>
            <span className="text-xs text-muted-foreground">添加选项</span>
            {availableOptions.map(ratio => (
              <Button
                key={ratio}
                type="button"
                variant="ghost"
                size="xs"
                onClick={() => toggle(ratio)}
              >
                + {ratio}
              </Button>
            ))}
          </>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Input
            aria-label="自定义选项"
            placeholder="自定义值，如：16:9"
            value={customRatio}
            onChange={e => setCustomRatio(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter") {
                e.preventDefault();
                addCustom();
              }
            }}
            className="h-8 w-40 font-mono text-xs"
          />
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={addCustom}
            disabled={!customRatio.trim() || selected.has(customRatio.trim())}
          >
            添加
          </Button>
        </div>
      </div>
    </div>
  );
}

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

export function getConfigFieldsByModelType(modelType: number, platform?: string | null): ConfigFieldDef[] {
  switch (modelType) {
    case 1:
      return getChatSamplingFields(platform);
    case 2:
      return [
        { key: "defaultWidth", label: "默认宽度", type: "number", min: 256, max: 8192, step: 64, placeholder: "例如：1024" },
        { key: "defaultHeight", label: "默认高度", type: "number", min: 256, max: 8192, step: 64, placeholder: "例如：1024" },
        { key: "minPixels", label: "最小像素数", type: "number", min: 0, step: 1, placeholder: "例如：921600", hint: "生成图像的最小总像素数限制" },
        { key: "maxPixels", label: "最大像素数", type: "number", min: 0, step: 1, placeholder: "例如：16777216", hint: "生成图像的最大总像素数限制" },
        { key: "supportedSizes", label: "支持的尺寸", type: "supported-sizes" },
        { key: "supportedAspectRatios", label: "支持的比例", type: "aspect-ratios" },
      ];
    case 3:
      return [
        { key: "supportedResolutions", label: "支持的分辨率", type: "aspect-ratios", hint: "如 480p, 720p, 1080p", presetOptions: ["480p", "720p", "1080p", "2K", "4K"] },
        { key: "supportedAspectRatios", label: "支持的宽高比", type: "aspect-ratios", hint: "如 16:9, 9:16, 1:1" },
        { key: "minDuration", label: "最短时长（秒）", type: "number", min: 1, max: 60, step: 1, placeholder: "例如：4" },
        { key: "maxDuration", label: "最长时长（秒）", type: "number", min: 1, max: 60, step: 1, placeholder: "例如：15" },
        { key: "defaultDuration", label: "默认时长（秒）", type: "number", min: 1, max: 60, step: 1, placeholder: "例如：5" },
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
    <div className={cn(
      "rounded-lg border border-border/30 bg-background/70 px-3 py-2.5",
      disabled && "opacity-50",
    )}>
      <div className="flex items-start gap-3">
        <button
          type="button"
          onClick={onToggle}
          disabled={disabled}
          className={cn(
            "relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors duration-200",
            checked ? "bg-primary" : "bg-muted-foreground/30"
          )}
        >
          <span
            className={cn(
              "absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
              checked && "translate-x-4"
            )}
          />
        </button>
        <div className="min-w-0">
          <Label
            className={cn("text-xs text-muted-foreground", !disabled && "cursor-pointer")}
            onClick={() => {
              if (!disabled) onToggle();
            }}
          >
            {title}
          </Label>
          <p className="mt-1 text-[10px] leading-5 text-muted-foreground/70">{description}</p>
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
        onChange={e => onChange(e.target.value)}
        className="text-xs font-mono h-8"
      />
      {hint && <p className="text-[10px] text-muted-foreground/70">{hint}</p>}
    </div>
  );
}
