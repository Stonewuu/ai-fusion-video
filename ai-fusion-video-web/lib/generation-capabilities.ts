import type { AiModel, ComfyUiInputCapability, ModelPreset } from "@/lib/api/ai-model";
import { videoRulesOf, type VideoModelRules } from "@/lib/video-model-rules";

type JsonRecord = Record<string, unknown>;

export interface GenerationCapabilities {
  unavailable?: string | null;
  workflowInputs?: Record<string, ComfyUiInputCapability>;
  supportsReferenceImages: boolean;
  referenceImageInputFormats: string[];
  supportsReferenceImageUrlInput: boolean;
  supportsReferenceImageBase64Input: boolean;
  minReferenceImages: number;
  maxReferenceImages: number;
  supportsFirstFrame: boolean;
  supportsLastFrame: boolean;
  supportsReferenceVideos: boolean;
  supportsReferenceAudios: boolean;
  minImageInputs: number;
  maxImageInputs: number;
  maxReferenceVideos: number;
  maxReferenceAudios: number;
  supportedAspectRatios: string[];
  supportedResolutions: string[];
  minDuration: number;
  maxDuration: number;
  maxDurationByResolution?: Record<string, number>;
  supportedDurations?: number[];
  supportedDurationsByResolution?: Record<string, number[]>;
  defaultDuration: number;
  supportCameraFixed: boolean;
  supportGenerateAudio: boolean;
  supportWatermark: boolean;
  supportSeed?: boolean;
  defaultAspectRatio?: string;
  defaultResolution?: string;
  videoRules?: VideoModelRules;
  exclusiveInputModes?: boolean;
  framesWithReferenceVideos?: boolean;
  referenceAudioRequiresVisualInput?: boolean;
  maxCount: number;
  defaultWidth?: number;
  defaultHeight?: number;
}

export interface ReferenceImageUploadAvailability {
  supported: boolean;
  reason: string;
}

function parseConfig(config?: string | null): JsonRecord {
  if (!config) {
    return {};
  }
  try {
    const parsed = JSON.parse(config);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : {};
  } catch {
    return {};
  }
}

function bool(config: JsonRecord, fallback: boolean, ...keys: string[]) {
  for (const key of keys) {
    if (typeof config[key] === "boolean") {
      return config[key] as boolean;
    }
  }
  return fallback;
}

function num(config: JsonRecord, fallback: number, ...keys: string[]) {
  for (const key of keys) {
    const value = Number(config[key]);
    if (Number.isFinite(value)) {
      return Math.max(0, value);
    }
  }
  return fallback;
}

function inputMaximum(config: JsonRecord, ...keys: string[]) {
  if (config.comfyuiWorkflowVersionId && Object.hasOwn(config, keys[0]) && config[keys[0]] === null) {
    return Number.POSITIVE_INFINITY;
  }
  return num(config, 0, ...keys);
}

function strings(config: JsonRecord, fallback: string[], ...keys: string[]) {
  for (const key of keys) {
    const value = config[key];
    if (Array.isArray(value)) {
      const result = value
        .map(String)
        .map((item) => item.trim())
        .filter(Boolean);
      if (result.length) {
        return result;
      }
    }
  }
  return fallback;
}

function sizeStrings(config: JsonRecord) {
  const value = config.supportedSizes;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }
  const result = new Set<string>();
  for (const sizeMap of Object.values(value as JsonRecord)) {
    if (!sizeMap || typeof sizeMap !== "object" || Array.isArray(sizeMap)) {
      continue;
    }
    for (const item of Object.values(sizeMap as JsonRecord)) {
      if (typeof item === "string" && item.trim()) {
        result.add(item.trim());
      }
    }
  }
  return [...result];
}

export function generationCapabilityError(model: AiModel): string | null {
  if (model.capabilityError) {
    return model.capabilityError;
  }
  if (model.comfyuiWorkflowId && !model.generationConfig) {
    return "工作流能力尚未加载，请刷新模型列表";
  }
  return null;
}

export function resolvedGenerationConfig(model: AiModel, presets: ModelPreset[]): JsonRecord {
  if (model.generationConfig) {
    return model.generationConfig;
  }
  if (model.comfyuiWorkflowId) {
    return {};
  }
  const preset = presets.find((item) => item.code === model.capabilityPresetCode && item.modelType === model.modelType);
  return { ...(preset?.config || {}), ...parseConfig(model.config) };
}

export function resolveGenerationCapabilities(
  model: AiModel,
  presets: ModelPreset[],
): GenerationCapabilities {
  const config = resolvedGenerationConfig(model, presets);
  const imageModel = model.modelType === 2;
  const videoRules = videoRulesOf(config);
  const referenceImageInputFormats = strings(
    config,
    [],
    "referenceImageInputFormats",
  )
    .map((item) => item.toLowerCase())
    .map((item) =>
      item === "base64" || item === "data-uri" ? "data_uri" : item,
    )
    .filter((item) => item === "url" || item === "data_uri");
  if (
    bool(config, false, "supportDataUriInput") &&
    !referenceImageInputFormats.includes("data_uri")
  ) {
    referenceImageInputFormats.push("data_uri");
  }

  return {
    unavailable: generationCapabilityError(model),
    workflowInputs: config.comfyuiInputs as Record<string, ComfyUiInputCapability> | undefined,
    supportsReferenceImages: bool(
      config,
      false,
      "supportReferenceImages",
      "supportsReferenceImages",
    ),
    referenceImageInputFormats,
    supportsReferenceImageUrlInput: referenceImageInputFormats.includes("url"),
    supportsReferenceImageBase64Input:
      referenceImageInputFormats.includes("data_uri"),
    minReferenceImages: num(config, 0, "minReferenceImages", "minRefImages"),
    maxReferenceImages: inputMaximum(config, "maxReferenceImages", "maxRefImages"),
    supportsFirstFrame: bool(
      config,
      false,
      "supportFirstFrame",
      "supportsFirstFrame",
    ),
    supportsLastFrame: bool(
      config,
      false,
      "supportLastFrame",
      "supportsLastFrame",
    ),
    supportsReferenceVideos: bool(
      config,
      false,
      "supportReferenceVideos",
      "supportsReferenceVideos",
    ),
    supportsReferenceAudios: bool(
      config,
      false,
      "supportReferenceAudios",
      "supportsReferenceAudios",
    ),
    minImageInputs: num(config, 0, "minImageInputs", "minImages"),
    maxImageInputs: inputMaximum(config, "maxImageInputs", "maxImages"),
    maxReferenceVideos: inputMaximum(config, "maxReferenceVideos", "maxRefVideos"),
    maxReferenceAudios: inputMaximum(config, "maxReferenceAudios", "maxRefAudios"),
    supportedAspectRatios: strings(
      config,
      ["1:1", "4:3", "3:4", "16:9", "9:16"],
      "supportedAspectRatios",
    ),
    supportedResolutions: strings(
      config,
      imageModel
        ? sizeStrings(config).slice(0, 12).length
          ? sizeStrings(config).slice(0, 12)
          : ["1024x1024"]
        : ["720p", "1080p"],
      "supportedResolutions",
      "resolutions",
    ),
    minDuration: num(config, 3, "minDuration"),
    maxDuration: num(config, 10, "maxDuration"),
    maxDurationByResolution: config.maxDurationByResolution as Record<string, number> | undefined,
    supportedDurations: config.supportedDurations as number[] | undefined,
    supportedDurationsByResolution: config.supportedDurationsByResolution as Record<string, number[]> | undefined,
    framesWithReferenceVideos: config.framesWithReferenceVideos as boolean | undefined,
    defaultDuration:
      config.defaultDuration === -1 && videoRules?.automaticDuration
        ? -1
        : num(config, 5, "defaultDuration"),
    defaultAspectRatio: videoRules?.defaultRatio,
    defaultResolution:
      typeof config.defaultResolution === "string"
        ? config.defaultResolution
        : undefined,
    videoRules,
    exclusiveInputModes: bool(config, false, "exclusiveInputModes"),
    referenceAudioRequiresVisualInput: bool(
      config,
      false,
      "referenceAudioRequiresVisualInput",
    ),
    supportCameraFixed: bool(config, true, "supportCameraFixed"),
    supportGenerateAudio: bool(
      config,
      true,
      "supportGenerateAudio",
      "generateAudioSupported",
    ),
    supportWatermark: bool(config, true, "supportWatermark"),
    supportSeed: bool(config, true, "supportSeed"),
    maxCount: num(config, 4, "maxCount", "maxOutputCount"),
    defaultWidth: num(config, 0, "defaultWidth") || undefined,
    defaultHeight: num(config, 0, "defaultHeight") || undefined,
  };
}

export function workflowInputIssue(capabilities: GenerationCapabilities, counts: Record<string, number>): string | null {
  if (!capabilities.workflowInputs) {
    return null;
  }
  const labels: Record<string, string> = { referenceImages: "参考图", firstFrame: "首帧", lastFrame: "尾帧",
    referenceVideos: "参考视频", referenceAudios: "参考音频" };
  for (const [field, count] of Object.entries(counts)) {
    if (count === 0) {
      continue;
    }
    const input = capabilities.workflowInputs[field];
    const name = labels[field] ?? field;
    if (!input || input.maximum === 0) {
      return `当前工作流不支持${name}输入`;
    }
    if (input.maximum !== null && count > input.maximum) {
      return `${name}最多允许 ${input.maximum} 项`;
    }
    if (count < input.minimumWhenProvided) {
      return `${name}需传入至少 ${input.minimumWhenProvided} 项`;
    }
  }
  const images = (counts.referenceImages ?? 0) + (counts.firstFrame ?? 0) + (counts.lastFrame ?? 0);
  return images > capabilities.maxImageInputs ? "图片输入总数超过使用上限" : null;
}

export function resolveReferenceImageUploadAvailability(
  capabilities: GenerationCapabilities | null,
  publicMediaAccessConfigured: boolean,
  runtimeConfigLoaded: boolean,
): ReferenceImageUploadAvailability {
  if (
    !capabilities ||
    (!capabilities.supportsReferenceImages &&
      !capabilities.supportsFirstFrame &&
      !capabilities.supportsLastFrame)
  ) {
    return { supported: false, reason: "模型不支持图片输入" };
  }

  if (capabilities.referenceImageInputFormats.length === 0) {
    return {
      supported: false,
      reason: "未配置 URL 或 Data URI 传递模式",
    };
  }

  if (capabilities.supportsReferenceImageBase64Input) {
    return { supported: true, reason: "" };
  }

  if (
    capabilities.supportsReferenceImageUrlInput &&
    publicMediaAccessConfigured
  ) {
    return { supported: true, reason: "" };
  }

  if (capabilities.supportsReferenceImageUrlInput && !runtimeConfigLoaded) {
    return { supported: false, reason: "正在检查公网访问配置" };
  }

  if (capabilities.supportsReferenceImageUrlInput) {
    return {
      supported: false,
      reason: "仅支持公网 URL，且未配置公网访问",
    };
  }

  return {
    supported: false,
    reason: "未声明 URL 或 Data URI 输入格式",
  };
}
