import type { GenerationCapabilities } from "./generation-capabilities";

/** 与服务端 videoRules v1 对应；界面只消费规则，不按模型名称猜测能力。 */
export interface VideoModelRules {
  version: 1;
  profile: string;
  promptRequired?: boolean;
  promptMaxLength?: number;
  automaticDuration?: boolean;
  defaultRatio?: string;
  frameRatio?: string;
  frameRatioMode?: "required" | "inferred";
  textRequiresFixedRatio?: boolean;
  fixedGenerateAudio?: boolean;
  defaultGenerateAudio?: boolean;
  lastFrameRequiresFirst?: boolean;
  maxReferenceInputs?: number;
  maxInputOutputVideoSeconds?: number;
  referenceVideoLimits?: VideoMediaLimits;
  referenceAudioLimits?: VideoMediaLimits;
  imageLimits?: VideoMediaLimits;
  promptExpansionModes?: string[];
  referenceTaskTypes?: string[];
  outputFormats?: string[];
  supportPromptExtend?: boolean;
}

export interface VideoMediaLimits {
  minSeconds?: number;
  maxSeconds?: number;
  maxTotalSeconds?: number;
  singleItemMaxTotalSeconds?: number;
  maxBytes?: number;
  minDimension?: number;
  maxDimension?: number;
  minAspect?: number;
  maxAspect?: number;
  minPixels?: number;
  maxPixels?: number;
  mimeTypes?: string[];
}

export interface VideoInputState {
  firstFrame: boolean;
  lastFrame: boolean;
  referenceImages: number;
  referenceVideos: number;
  referenceAudios: number;
  references: boolean;
  any: boolean;
}

export function videoRulesOf(
  parameters?: Record<string, unknown>,
): VideoModelRules | undefined {
  return parameters?.videoRules as VideoModelRules | undefined;
}

export function videoDurationOptions(
  parameters: { supportedDurations?: unknown; supportedDurationsByResolution?: unknown } | undefined,
  resolution: string,
): number[] | undefined {
  const mapping = parameters?.supportedDurationsByResolution;
  const specific = mapping && typeof mapping === "object" && !Array.isArray(mapping)
    ? (mapping as Record<string, unknown>)[resolution]
    : undefined;
  const values = specific ?? parameters?.supportedDurations;
  if (!Array.isArray(values)) {
    return undefined;
  }
  return values.filter((value): value is number => Number.isInteger(value) && value > 0);
}

export function videoDurationForResolution(
  value: number,
  capabilities: GenerationCapabilities,
  resolution: string,
): number {
  if (automaticVideoDuration(value, capabilities.videoRules)) {
    return value;
  }
  const options = videoDurationOptions(capabilities, resolution);
  if (options?.length && !options.includes(value)) {
    return options.includes(capabilities.defaultDuration) ? capabilities.defaultDuration : options[0];
  }
  return Math.min(videoMaxDuration(capabilities, resolution), Math.max(capabilities.minDuration, value));
}

type VideoInputKey =
  | "firstFrame"
  | "lastFrame"
  | "referenceImages"
  | "referenceVideos"
  | "referenceAudios";

export function videoInputState(
  input: Partial<Record<VideoInputKey, unknown>>,
  connected: string[] = [],
): VideoInputState {
  const count = (key: VideoInputKey) => {
    const value = input[key];
    const present = (item: unknown) =>
      typeof item === "string" ? !!item.trim() : item != null;
    return Array.isArray(value)
      ? value.filter(present).length
      : present(value)
        ? 1
        : 0;
  };
  const firstFrame =
    count("firstFrame") > 0 || connected.includes("firstFrame");
  const lastFrame = count("lastFrame") > 0 || connected.includes("lastFrame");
  const referenceImages = count("referenceImages");
  const referenceVideos = count("referenceVideos");
  const referenceAudios = count("referenceAudios");
  const references =
    referenceImages + referenceVideos + referenceAudios > 0 ||
    ["referenceImages", "referenceVideos", "referenceAudios"].some((key) =>
      connected.includes(key),
    );
  return {
    firstFrame,
    lastFrame,
    referenceImages,
    referenceVideos,
    referenceAudios,
    references,
    any: firstFrame || lastFrame || references,
  };
}

export function videoRatioOptions(
  values: string[],
  rules: VideoModelRules | undefined,
  inputs: VideoInputState,
) {
  if (rules?.frameRatio && (inputs.firstFrame || inputs.lastFrame)) {
    return [rules.frameRatio];
  }
  if (rules?.textRequiresFixedRatio && !inputs.any) {
    return values.filter((value) => value !== "adaptive");
  }
  return values;
}

export function videoPromptReady(
  prompt: string,
  rules: VideoModelRules | undefined,
  inputs: VideoInputState,
) {
  return !!prompt.trim() || (rules?.promptRequired === false && inputs.any);
}

export function automaticVideoDuration(
  value: unknown,
  rules: VideoModelRules | undefined,
) {
  return value === -1 && rules?.automaticDuration === true;
}

export function videoMaxDuration(
  capabilities: GenerationCapabilities,
  resolution: string,
): number {
  return capabilities.maxDurationByResolution?.[resolution] ?? capabilities.maxDuration;
}

export function videoFormIssue(
  capabilities: GenerationCapabilities,
  form: Partial<Record<VideoInputKey, unknown>> & {
    prompt: string;
    ratio: string;
    duration: number;
    resolution?: string;
    seed: string;
  },
): string | null {
  const rules = capabilities.videoRules;
  const inputs = videoInputState(form);
  if (!videoPromptReady(form.prompt, rules, inputs)) {
    return rules?.promptRequired === false
      ? "请填写创作要求或添加参考素材"
      : "请填写创作要求";
  }
  if (
    rules?.promptMaxLength &&
    [...form.prompt].length > rules.promptMaxLength
  ) {
    return `创作要求最多允许 ${rules.promptMaxLength} 个字符`;
  }
  if (rules?.lastFrameRequiresFirst && inputs.lastFrame && !inputs.firstFrame) {
    return "使用尾帧时，请同时添加首帧图片";
  }
  if (
    capabilities.exclusiveInputModes &&
    (inputs.firstFrame || inputs.lastFrame) &&
    inputs.references
  ) {
    return "首尾帧与参考素材不能混用，请选择一种输入方式";
  }
  if (
    capabilities.framesWithReferenceVideos === false &&
    (inputs.firstFrame || inputs.lastFrame) &&
    inputs.referenceVideos > 0
  ) {
    return "当前模型的首尾帧不能与参考视频混用";
  }
  if (
    capabilities.referenceAudioRequiresVisualInput &&
    inputs.referenceAudios > 0 &&
    !inputs.firstFrame &&
    !inputs.lastFrame &&
    !inputs.referenceImages &&
    !inputs.referenceVideos
  ) {
    return "参考音频需要搭配图片或视频";
  }
  const total =
    Number(inputs.firstFrame) +
    Number(inputs.lastFrame) +
    inputs.referenceImages +
    inputs.referenceVideos +
    inputs.referenceAudios;
  if (rules?.maxReferenceInputs && total > rules.maxReferenceInputs) {
    return `参考素材总数不能超过 ${rules.maxReferenceInputs} 项`;
  }
  if (
    form.ratio &&
    !videoRatioOptions(
      capabilities.supportedAspectRatios,
      rules,
      inputs,
    ).includes(form.ratio)
  ) {
    return "请按当前输入重新选择画面比例";
  }
  if (
    !automaticVideoDuration(form.duration, rules) &&
    (form.duration < capabilities.minDuration ||
      form.duration > videoMaxDuration(capabilities, form.resolution ?? ""))
  ) {
    return "视频时长超出模型允许范围";
  }
  const durations = videoDurationOptions(capabilities, form.resolution ?? "");
  if (!automaticVideoDuration(form.duration, rules) && durations?.length && !durations.includes(form.duration)) {
    return `当前分辨率支持的时长为 ${durations.join("、")} 秒`;
  }
  if (form.seed.trim()) {
    if (capabilities.supportSeed === false) {
      return "此模型不支持随机种子，请清空种子设置";
    }
    const seed = Number(form.seed);
    if (!Number.isSafeInteger(seed) || seed < -1 || seed > 2147483647) {
      return "随机种子须为 -1 或 0 至 2147483647 的整数";
    }
  }
  return null;
}
