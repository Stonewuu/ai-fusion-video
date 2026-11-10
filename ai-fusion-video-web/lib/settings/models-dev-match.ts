import type { AiModelCreateReq, ModelPreset, MultimodalInputType, MultimodalInputTransports } from "@/lib/api/ai-model";
import type { ModelsDevEntry } from "@/lib/api/models-dev";

const MODALITIES: Record<string, MultimodalInputType> = { image: "image", video: "video", audio: "audio", pdf: "file", file: "file" };

/** 仅在用户应用候选项时填入能力，保留当前模型的调用标识和渠道配置。 */
export function applyModelsDevMatch<T extends AiModelCreateReq>(
  form: T, entry: ModelsDevEntry, fetchedAt: string, officialPreset?: ModelPreset,
): T {
  const next = { ...form };
  if (entry.contextWindow != null) {
    next.contextWindow = entry.contextWindow;
  }
  if (entry.supportReasoning != null) {
    next.supportReasoning = entry.supportReasoning;
    if (!entry.supportReasoning) {
      next.reasoningEffortLevels = [];
    }
  }
  if (entry.supportReasoning !== false && entry.reasoningEffortLevels != null) {
    next.reasoningEffortLevels = entry.reasoningEffortLevels;
  }
  if (entry.inputModalities != null && form.modelType === 1) {
    next.multimodalInputTypes = [...new Set(entry.inputModalities.map(value => MODALITIES[value]).filter(Boolean))];
    const transports: MultimodalInputTransports = {};
    for (const type of next.multimodalInputTypes) {
      // 与手动启用多模态输入的表单默认值一致；已有传输设置优先保留。
      transports[type] = form.multimodalInputTransports[type]?.length ? form.multimodalInputTransports[type] : ["base64"];
    }
    next.multimodalInputTransports = transports;
    next.supportVision = next.multimodalInputTypes.includes("image");
  }
  let config: Record<string, unknown> = {};
  if (form.config?.trim()) {
    try {
      config = JSON.parse(form.config);
      if (!config || typeof config !== "object" || Array.isArray(config)) {
        throw new Error();
      }
    } catch {
      throw new Error("请先修正高级配置中的 JSON 格式");
    }
  }
  if (officialPreset) {
    next.capabilityPresetCode = officialPreset.code;
    next.contextWindow = officialPreset.contextWindow ?? next.contextWindow;
    next.supportReasoning = officialPreset.supportReasoning ?? next.supportReasoning;
    next.reasoningEffortLevels = officialPreset.reasoningEffortLevels ?? next.reasoningEffortLevels;
    next.multimodalInputTypes = officialPreset.multimodalInputTypes ?? next.multimodalInputTypes;
    next.multimodalInputTransports = officialPreset.multimodalInputTransports ?? next.multimodalInputTransports;
    next.supportVision = next.multimodalInputTypes.includes("image");
  }
  config.catalogMatch = {
    source: "models.dev", providerId: entry.providerId, modelId: entry.modelId,
    canonicalModelId: entry.canonicalModelId, fetchedAt, updatedAt: entry.updatedAt,
    outputTokenLimit: entry.outputTokenLimit,
  };
  next.config = JSON.stringify(config);
  return next;
}

export function findOfficialGooglePreset(entry: ModelsDevEntry, presets: ModelPreset[], platform?: string | null) {
  if (!["gemini", "vertex_ai", "vertex_ai_api_key"].includes(platform || "")) {
    return undefined;
  }
  const canonical = entry.canonicalModelId;
  const modelId = canonical?.startsWith("google/") ? canonical.slice("google/".length)
    : entry.providerId === "google" || entry.providerId === "google-vertex" ? entry.modelId : undefined;
  return presets.find(preset => preset.official && preset.modelType === entry.modelType &&
    (preset.modelCode || preset.code) === modelId && preset.matchPlatforms?.includes(platform || ""));
}
