import type { ApiConfig, ModelPreset } from "@/lib/api/ai-model";
import type { AiModelFormState } from "./ai-model-advanced-section";
import {
  getApiConfigDefaultProtocol,
  isCapabilityPresetCompatible,
  isConfigRecord,
  mergeConfigObjects,
  parseConfigJson,
  stringifyConfigObject,
} from "./model-config-support";

const MULTIMODAL_LABELS: Record<string, string> = {
  image: "图片",
  video: "视频",
  audio: "音频",
  file: "文件",
};

export function materializeEffectiveModelConfig(
  source: AiModelFormState,
  presets: readonly ModelPreset[],
): string {
  const preset = source.capabilityPresetCode
    ? presets.find((item) => item.code === source.capabilityPresetCode)
    : null;
  return stringifyConfigObject(
    mergeConfigObjects(preset?.config || {}, parseConfigJson(source.config)),
  );
}

export function detachIncompatiblePreset(
  previous: AiModelFormState,
  next: AiModelFormState,
  nextApiConfig: ApiConfig | undefined,
  presets: readonly ModelPreset[],
): AiModelFormState {
  if (!previous.capabilityPresetCode) {
    return next;
  }
  const preset = presets.find(
    (item) => item.code === previous.capabilityPresetCode,
  );
  if (!preset) {
    return next;
  }
  const nextProtocol =
    next.modelProtocol?.trim()
    || getApiConfigDefaultProtocol(nextApiConfig, next.modelType);
  if (
    (!preset.providerId || preset.providerId === nextApiConfig?.providerId)
    && isCapabilityPresetCompatible({
      preset,
      modelType: next.modelType,
      platform: nextApiConfig?.platform,
      effectiveProtocol: nextProtocol,
    })
  ) {
    return next;
  }
  return {
    ...next,
    capabilityPresetCode: "",
    config: materializeEffectiveModelConfig(previous, presets),
  };
}

export function parseModelRawJsonDraft(
  draft: string,
): { ok: true; value: string } | { ok: false; error: string } {
  if (!draft.trim()) {
    return { ok: true, value: "" };
  }
  try {
    const parsed = JSON.parse(draft);
    if (!isConfigRecord(parsed)) {
      return {
        ok: false,
        error: 'JSON 必须为对象格式（如 {"temperature": 0.7}）。',
      };
    }
    const cleaned = Object.fromEntries(
      Object.entries(parsed).filter(
        ([, value]) => value !== "" && value !== undefined && value !== null,
      ),
    );
    return {
      ok: true,
      value: Object.keys(cleaned).length > 0 ? JSON.stringify(cleaned) : "",
    };
  } catch {
    return { ok: false, error: "JSON 语法错误，请检查括号、引号或逗号。" };
  }
}

export function buildModelCapabilityHighlights(
  form: AiModelFormState,
  effectiveModelConfig: Record<string, unknown>,
): string[] {
  const highlights: string[] = [];
  if (form.modelType === 1) {
    if (form.contextWindow && form.contextWindow > 0) {
      highlights.push(`上下文 ${form.contextWindow.toLocaleString()}`);
    }
    if (form.supportReasoning) {
      highlights.push("支持思考");
    }
    if (form.multimodalInputTypes.length > 0) {
      highlights.push(
        `输入：${form.multimodalInputTypes.map((item) => MULTIMODAL_LABELS[item] || item).join(" / ")}`,
      );
    }
    return highlights;
  }
  if (form.modelType === 2 && isConfigRecord(effectiveModelConfig.supportedSizes)) {
    const tiers = Object.keys(effectiveModelConfig.supportedSizes);
    if (tiers.length > 0) {
      highlights.push(`尺寸档位：${tiers.join(" / ")}`);
    }
    return highlights;
  }
  if (form.modelType === 3 && Array.isArray(effectiveModelConfig.supportedResolutions)) {
    const resolutions = effectiveModelConfig.supportedResolutions.filter(
      (item): item is string => typeof item === "string",
    );
    if (resolutions.length > 0) {
      highlights.push(`分辨率：${resolutions.join(" / ")}`);
    }
  }
  return highlights;
}
