import type {
  CanvasConfig,
  CanvasConfigField,
  CanvasModelOption,
} from "./types";
import { canvasChoiceLabels } from "./labels";
import {
  automaticVideoDuration,
  videoDurationOptions,
  videoRatioOptions,
  videoRulesOf,
  type VideoInputState,
} from "../video-model-rules";

export interface FieldChoice {
  value: string;
  label: string;
}

export const configFieldOrder = [
  "content",
  "prompt",
  "instruction",
  "apiConfigId",
  "operation",
  "frameType",
  "valueType",
  "format",
  "ratio",
  "resolution",
  "duration",
  "count",
  "width",
  "height",
  "fit",
  "x",
  "y",
  "frameRate",
  "originalVolume",
  "musicVolume",
  "sfxVolume",
  "separator",
  "generateAudio",
  "cameraFixed",
  "watermark",
  "seed",
  "negativePrompt",
  "fps",
];
export const advancedConfigFields = new Set([
  "cameraFixed",
  "watermark",
  "seed",
  "negativePrompt",
  "fps",
]);
export const fieldHints: Record<string, string> = {
  prompt: "描述主体、场景、动作和画面风格。",
  instruction: "补充这一步的创作要求、重点和需要避免的内容。",
  negativePrompt: "填写不希望出现在画面里的内容。",
  count: "生成多个候选，方便比较后选用。",
  cameraFixed: "开启后尽量保持机位不动。",
  generateAudio: "是否同时生成视频声音。",
  seed: "固定种子便于比较不同参数下的效果。",
  x: "从原图左边缘开始，单位为像素。",
  y: "从原图上边缘开始，单位为像素。",
};

const strings = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];

export function configFieldChoices(
  kind: string,
  key: string,
  field: CanvasConfigField,
  model?: CanvasModelOption,
  inputs?: VideoInputState,
): FieldChoice[] {
  let values = field.enumValues;
  if (key === "ratio") {
    values = strings(model?.parameters.supportedAspectRatios);
  }
  if (key === "ratio" && inputs) {
    values = videoRatioOptions(values, videoRulesOf(model?.parameters), inputs);
  }
  if (key === "resolution" && model) {
    values = strings(model.parameters.supportedResolutions);
    const sizes = model.parameters.supportedSizes;
    if (
      !values.length &&
      sizes &&
      typeof sizes === "object" &&
      !Array.isArray(sizes)
    ) {
      values = Object.keys(sizes);
    }
  }
  if (key === "separator") {
    return [
      { value: "\n\n", label: "空行" },
      { value: "\n", label: "换行" },
      { value: " ", label: "空格" },
      { value: "", label: "直接拼接" },
    ];
  }
  if (kind === "videoCompose" && key === "resolution") {
    return [
      { value: "1280x720", label: "横屏 720P" },
      { value: "1920x1080", label: "横屏 1080P" },
      { value: "720x1280", label: "竖屏 720P" },
      { value: "1080x1920", label: "竖屏 1080P" },
      { value: "1080x1080", label: "方形 1080P" },
    ];
  }
  return values.map((value) => ({
    value,
    label: canvasChoiceLabels[value] ?? value,
  }));
}

export function configNumberBounds(
  key: string,
  field: CanvasConfigField,
  model?: CanvasModelOption,
  config?: CanvasConfig,
) {
  const minDuration = model?.parameters.minDuration;
  let maxDuration = model?.parameters.maxDuration;
  const resolutionLimits = model?.parameters.maxDurationByResolution;
  if (resolutionLimits && typeof resolutionLimits === "object" && !Array.isArray(resolutionLimits)) {
    const resolution = String(config?.resolution ?? model?.parameters.defaultResolution ?? "");
    const maximum = (resolutionLimits as Record<string, unknown>)[resolution];
    if (typeof maximum === "number" && typeof maxDuration === "number") {
      maxDuration = Math.min(maxDuration, maximum);
    }
  }
  return {
    min:
      key === "duration" && typeof minDuration === "number"
        ? Math.max(field.minimum ?? minDuration, minDuration)
        : (field.minimum ?? undefined),
    max:
      key === "duration" && typeof maxDuration === "number"
        ? Math.min(field.maximum ?? maxDuration, maxDuration)
        : (field.maximum ?? undefined),
  };
}

export function configuredImageSize(
  config: CanvasConfig,
  model?: CanvasModelOption,
): string | null {
  const sizes = model?.parameters.supportedSizes;
  if (!sizes || typeof sizes !== "object" || Array.isArray(sizes)) {
    return null;
  }
  const tier = (sizes as Record<string, unknown>)[String(config.resolution)];
  if (!tier || typeof tier !== "object" || Array.isArray(tier)) {
    return null;
  }
  const size = (tier as Record<string, unknown>)[String(config.ratio)];
  return typeof size === "string" ? size : null;
}

export function configFieldError(
  key: string,
  field: CanvasConfigField,
  value: unknown,
  choices: FieldChoice[],
  model?: CanvasModelOption,
  config?: CanvasConfig,
): string | null {
  if (value === undefined) {
    return null;
  }
  const promptLimit = videoRulesOf(model?.parameters)?.promptMaxLength;
  if (
    key === "prompt" &&
    promptLimit &&
    [...String(value)].length > promptLimit
  ) {
    return `创作要求最多允许 ${promptLimit} 个字符。`;
  }
  if (
    key === "duration" &&
    automaticVideoDuration(value, videoRulesOf(model?.parameters))
  ) {
    return null;
  }
  const supported = {
    seed: "supportSeed",
    watermark: "supportWatermark",
    generateAudio: "supportGenerateAudio",
  }[key];
  const fixedAudio = videoRulesOf(model?.parameters)?.fixedGenerateAudio;
  if (key === "generateAudio" && fixedAudio !== undefined) {
    return value === fixedAudio
      ? null
      : "此模型固定生成带声音的视频，请恢复默认设置。";
  }
  if (supported && model?.parameters[supported] === false && value !== false) {
    return "此模型不支持这项设置，请恢复默认或更换模型。";
  }
  if (
    key === "cameraFixed" &&
    value === true &&
    model?.parameters.supportCameraFixed === false
  ) {
    return "当前模型不支持固定镜头，请关闭或更换模型。";
  }
  if (
    ["ratio", "resolution"].includes(key) &&
    model &&
    choices.length &&
    !choices.some((choice) => choice.value === String(value))
  ) {
    return "当前值不在所选模型的支持范围内，请重新选择。";
  }
  if (typeof value === "number") {
    const durations = videoDurationOptions(model?.parameters, String(config?.resolution ?? model?.parameters.defaultResolution ?? ""));
    if (key === "duration" && durations?.length && !durations.includes(value)) {
      return `当前分辨率支持的时长为 ${durations.join("、")} 秒。`;
    }
    const { min, max } = configNumberBounds(key, field, model, config);
    if (
      !Number.isFinite(value) ||
      (field.type === "integer" && !Number.isSafeInteger(value))
    ) {
      return "请输入有效数字。";
    }
    if (
      (min !== undefined && value < min) ||
      (max !== undefined && value > max)
    ) {
      return `允许范围：${min ?? "不限"}–${max ?? "不限"}。`;
    }
  }
  return null;
}
