import type { AiModel, ModelPreset } from "@/lib/api/ai-model";

type JsonRecord = Record<string, unknown>;

export interface GenerationCapabilities {
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
  defaultDuration: number;
  supportCameraFixed: boolean;
  supportGenerateAudio: boolean;
  supportWatermark: boolean;
  maxCount: number;
  defaultWidth?: number;
  defaultHeight?: number;
}

export interface ReferenceImageUploadAvailability {
  supported: boolean;
  reason: string;
}

function parseConfig(config?: string | null): JsonRecord {
  if (!config) return {};
  try {
    const parsed = JSON.parse(config);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function bool(config: JsonRecord, fallback: boolean, ...keys: string[]) {
  for (const key of keys) {
    if (typeof config[key] === "boolean") return config[key] as boolean;
  }
  return fallback;
}

function num(config: JsonRecord, fallback: number, ...keys: string[]) {
  for (const key of keys) {
    const value = Number(config[key]);
    if (Number.isFinite(value)) return Math.max(0, value);
  }
  return fallback;
}

function strings(config: JsonRecord, fallback: string[], ...keys: string[]) {
  for (const key of keys) {
    const value = config[key];
    if (Array.isArray(value)) {
      const result = value.map(String).map(item => item.trim()).filter(Boolean);
      if (result.length) return result;
    }
  }
  return fallback;
}

function sizeStrings(config: JsonRecord) {
  const value = config.supportedSizes;
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const result = new Set<string>();
  for (const sizeMap of Object.values(value as JsonRecord)) {
    if (!sizeMap || typeof sizeMap !== "object" || Array.isArray(sizeMap)) continue;
    for (const item of Object.values(sizeMap as JsonRecord)) {
      if (typeof item === "string" && item.trim()) result.add(item.trim());
    }
  }
  return [...result];
}

export function resolveGenerationCapabilities(model: AiModel, presets: ModelPreset[]): GenerationCapabilities {
  const preset = model.capabilityPresetCode
    ? presets.find(item => item.code === model.capabilityPresetCode && item.modelType === model.modelType)
    : undefined;
  const config = { ...(preset?.config || {}), ...parseConfig(model.config) };
  const imageModel = model.modelType === 2;
  const referenceImageInputFormats = strings(config, [], "referenceImageInputFormats")
    .map((item) => item.toLowerCase())
    .map((item) => item === "base64" || item === "data-uri" ? "data_uri" : item)
    .filter((item) => item === "url" || item === "data_uri");
  if (
    bool(config, false, "supportDataUriInput") &&
    !referenceImageInputFormats.includes("data_uri")
  ) {
    referenceImageInputFormats.push("data_uri");
  }

  return {
    supportsReferenceImages: bool(config, false, "supportReferenceImages", "supportsReferenceImages"),
    referenceImageInputFormats,
    supportsReferenceImageUrlInput: referenceImageInputFormats.includes("url"),
    supportsReferenceImageBase64Input: referenceImageInputFormats.includes("data_uri"),
    minReferenceImages: num(config, 0, "minReferenceImages", "minRefImages"),
    maxReferenceImages: num(config, 0, "maxReferenceImages", "maxRefImages"),
    supportsFirstFrame: bool(config, false, "supportFirstFrame", "supportsFirstFrame"),
    supportsLastFrame: bool(config, false, "supportLastFrame", "supportsLastFrame"),
    supportsReferenceVideos: bool(config, false, "supportReferenceVideos", "supportsReferenceVideos"),
    supportsReferenceAudios: bool(config, false, "supportReferenceAudios", "supportsReferenceAudios"),
    minImageInputs: num(config, 0, "minImageInputs", "minImages"),
    maxImageInputs: num(config, 0, "maxImageInputs", "maxImages"),
    maxReferenceVideos: num(config, 0, "maxReferenceVideos", "maxRefVideos"),
    maxReferenceAudios: num(config, 0, "maxReferenceAudios", "maxRefAudios"),
    supportedAspectRatios: strings(config, ["1:1", "4:3", "3:4", "16:9", "9:16"], "supportedAspectRatios"),
    supportedResolutions: strings(config, imageModel ? (sizeStrings(config).slice(0, 12).length ? sizeStrings(config).slice(0, 12) : ["1024x1024"]) : ["720p", "1080p"], "supportedResolutions", "resolutions"),
    minDuration: num(config, 3, "minDuration"),
    maxDuration: num(config, 10, "maxDuration"),
    defaultDuration: num(config, 5, "defaultDuration"),
    supportCameraFixed: bool(config, true, "supportCameraFixed"),
    supportGenerateAudio: bool(config, true, "supportGenerateAudio", "generateAudioSupported"),
    supportWatermark: bool(config, true, "supportWatermark"),
    maxCount: num(config, 4, "maxCount", "maxOutputCount"),
    defaultWidth: num(config, 0, "defaultWidth") || undefined,
    defaultHeight: num(config, 0, "defaultHeight") || undefined,
  };
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

  if (capabilities.supportsReferenceImageUrlInput && publicMediaAccessConfigured) {
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
