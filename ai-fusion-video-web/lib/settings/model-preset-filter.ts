import type { ModelPreset } from "@/lib/api/ai-model";

const NATIVE_PLATFORMS = new Set(["gemini", "vertex_ai", "vertex_ai_api_key", "anthropic", "dashscope", "volcengine", "deepseek", "ollama", "GoogleFlowReverseApi"]);

/** 官方渠道只列对应预设；兼容渠道可选择多个厂商，专属渠道保留自己的能力。 */
export function filterModelPresets(presets: ModelPreset[], modelType: number, platform?: string | null,
  provider?: { id: string; quickSetup: boolean }) {
  return presets.filter(preset => {
    if (preset.modelType !== modelType) {
      return false;
    }
    if (preset.providerId) {
      return preset.providerId === provider?.id;
    }
    if (provider?.quickSetup && modelType !== 1) {
      return false;
    }
    if (!platform) {
      return true;
    }
    if (preset.matchPlatforms) {
      return preset.matchPlatforms.includes(platform);
    }
    return !NATIVE_PLATFORMS.has(platform) || preset.platform === platform;
  });
}
