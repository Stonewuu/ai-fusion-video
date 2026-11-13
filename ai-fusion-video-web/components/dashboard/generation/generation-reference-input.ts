import type { GenerationCapabilities } from "@/lib/generation-capabilities";
import type { ImageGenerationTask, VideoGenerationTask } from "@/lib/api/generation";
import type {
  GenerationFormState,
  GenerationResultItem,
  GenerationTask,
  SimpleAttachment,
  SimpleAttachmentKind,
  WorkbenchMode,
} from "./generation-types";

const REFERENCE_FIELDS = {
  firstFrame: "firstFrame",
  lastFrame: "lastFrame",
  referenceImage: "referenceImages",
  referenceVideo: "referenceVideos",
  referenceAudio: "referenceAudios",
} as const;

export function referenceValues(value: string | null | undefined): string[] {
  if (!value) {
    return [];
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("原任务的参考素材记录无法读取");
  }
  if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== "string" || !item.trim())) {
    throw new Error("原任务的参考素材记录格式无效");
  }
  return parsed;
}

export function generationFormFromTask(mode: WorkbenchMode, task: GenerationTask, defaults: GenerationFormState): GenerationFormState {
  const common = {
    ...defaults,
    prompt: task.prompt,
    ratio: task.ratio ?? defaults.ratio,
    resolution: task.resolution ?? defaults.resolution,
    count: task.count,
    firstFrame: [],
    lastFrame: [],
    referenceImages: [],
    referenceVideos: [],
    referenceAudios: [],
  };
  if (mode === "image") {
    const image = task as ImageGenerationTask;
    return { ...common, referenceImages: referenceValues(image.refImageUrls) };
  }
  const video = task as VideoGenerationTask;
  return {
    ...common,
    duration: video.duration ?? defaults.duration,
    seed: video.seed == null ? "" : String(video.seed),
    watermark: video.watermark ?? defaults.watermark,
    generateAudio: video.generateAudio ?? defaults.generateAudio,
    cameraFixed: video.cameraFixed ?? defaults.cameraFixed,
    firstFrame: video.firstFrameImageUrl ? [video.firstFrameImageUrl] : [],
    lastFrame: video.lastFrameImageUrl ? [video.lastFrameImageUrl] : [],
    referenceImages: referenceValues(video.referenceImageUrls),
    referenceVideos: referenceValues(video.referenceVideoUrls),
    referenceAudios: referenceValues(video.referenceAudioUrls),
  };
}

export function appendResultReference(
  mode: WorkbenchMode,
  form: GenerationFormState,
  capabilities: GenerationCapabilities | null,
  item: GenerationResultItem,
): { form: GenerationFormState; label: string } {
  if (!capabilities) {
    throw new Error("请先选择生成模型");
  }
  let kind: SimpleAttachmentKind;
  let source: string;
  let label: string;
  if ("videoUrl" in item) {
    if (!item.videoUrl) {
      throw new Error("视频结果尚未就绪，请稍后再添加引用");
    }
    if (mode !== "video" || !capabilities.supportsReferenceVideos) {
      throw new Error("当前模型不支持参考视频，请先选择支持参考视频的模型");
    }
    if (form.referenceVideos.includes(item.videoUrl)) {
      throw new Error("此视频已在参考素材中");
    }
    if (form.referenceVideos.length >= capabilities.maxReferenceVideos) {
      throw new Error("参考视频数量已达上限，请先移除不需要的视频");
    }
    kind = "referenceVideo";
    source = item.videoUrl;
    label = "参考视频";
  } else {
    if (!item.imageUrl) {
      throw new Error("图片结果尚未就绪，请稍后再添加引用");
    }
    source = item.imageUrl;
    const images = [...form.firstFrame, ...form.lastFrame, ...form.referenceImages];
    if (images.includes(source)) {
      throw new Error("此图片已在参考素材中");
    }
    if (mode === "video" && images.length >= capabilities.maxImageInputs) {
      throw new Error("图片输入数量已达上限");
    }
    if (mode === "video" && capabilities.supportsFirstFrame && form.firstFrame.length === 0) {
      kind = "firstFrame";
      label = "首帧";
    } else if (mode === "video" && capabilities.supportsLastFrame && form.lastFrame.length === 0) {
      kind = "lastFrame";
      label = "尾帧";
    } else if (capabilities.supportsReferenceImages && form.referenceImages.length < capabilities.maxReferenceImages) {
      kind = "referenceImage";
      label = "参考图片";
    } else {
      throw new Error("当前模型没有可用的图片输入位置，请检查模型与参考素材");
    }
  }
  const field = REFERENCE_FIELDS[kind];
  return { form: { ...form, [field]: [...form[field], source] }, label };
}

export function removeGenerationReference(form: GenerationFormState, attachment: SimpleAttachment): GenerationFormState {
  const field = REFERENCE_FIELDS[attachment.kind];
  return { ...form, [field]: form[field].filter((source) => source !== attachment.url) };
}

export function taskReferences(mode: WorkbenchMode, task: GenerationTask): SimpleAttachment[] {
  if (mode === "image") {
    const image = task as ImageGenerationTask;
    return referenceValues(image.refImageUrls).map((url) => ({ url, kind: "referenceImage", mediaType: "image", label: "参考图" }));
  }
  const video = task as VideoGenerationTask;
  const result: SimpleAttachment[] = [];
  if (video.firstFrameImageUrl) {
    result.push({ url: video.firstFrameImageUrl, kind: "firstFrame", mediaType: "image", label: "首帧" });
  }
  if (video.lastFrameImageUrl) {
    result.push({ url: video.lastFrameImageUrl, kind: "lastFrame", mediaType: "image", label: "尾帧" });
  }
  result.push(...referenceValues(video.referenceImageUrls).map((url) => ({ url, kind: "referenceImage" as const, mediaType: "image" as const, label: "参考图" })));
  result.push(...referenceValues(video.referenceVideoUrls).map((url) => ({ url, kind: "referenceVideo" as const, mediaType: "video" as const, label: "参考视频" })));
  result.push(...referenceValues(video.referenceAudioUrls).map((url) => ({ url, kind: "referenceAudio" as const, mediaType: "audio" as const, label: "参考音频" })));
  return result;
}
