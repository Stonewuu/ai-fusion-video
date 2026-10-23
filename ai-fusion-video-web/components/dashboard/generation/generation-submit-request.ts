import type {
  ImageGenerationSubmitReq,
  VideoGenerationSubmitReq,
} from "@/lib/api/generation";
import type { GenerationFormState, WorkbenchMode } from "./generation-types";
import type { GenerationCapabilities } from "@/lib/generation-capabilities";

export type GenerationInput = Omit<
  ImageGenerationSubmitReq | VideoGenerationSubmitReq,
  "requestId" | "scope"
>;
const clean = (values: string[]) =>
  values.map((value) => value.trim()).filter(Boolean);

export function generationSubmitRequest(
  mode: WorkbenchMode,
  form: GenerationFormState,
  modelId: number,
  capabilities?: GenerationCapabilities | null,
): GenerationInput {
  const referenceImages = clean(form.referenceImages);
  const common = {
    prompt: form.prompt.trim(),
    modelId,
    ratio: form.ratio || undefined,
    resolution: form.resolution || undefined,
    count: form.count,
  };
  if (mode === "image") {
    const input: Omit<ImageGenerationSubmitReq, "requestId" | "scope"> = {
      ...common,
      refImageUrls: referenceImages.length
        ? JSON.stringify(referenceImages)
        : undefined,
      aspectRatio: form.ratio || undefined,
      category: "manual_image_generation",
    };
    return input;
  }
  const firstFrame = clean(form.firstFrame);
  const lastFrame = clean(form.lastFrame);
  const referenceVideos = clean(form.referenceVideos);
  const referenceAudios = clean(form.referenceAudios);
  const input: Omit<VideoGenerationSubmitReq, "requestId" | "scope"> = {
    ...common,
    generateMode:
      referenceImages.length + firstFrame.length + lastFrame.length > 0
        ? "image2video"
        : "text2video",
    firstFrameImageUrl: firstFrame[0],
    lastFrameImageUrl: lastFrame[0],
    referenceImageUrls: referenceImages.length
      ? JSON.stringify(referenceImages)
      : undefined,
    referenceVideoUrls: referenceVideos.length
      ? JSON.stringify(referenceVideos)
      : undefined,
    referenceAudioUrls: referenceAudios.length
      ? JSON.stringify(referenceAudios)
      : undefined,
    duration: form.duration,
    watermark: capabilities?.supportWatermark === false ? undefined : form.watermark,
    generateAudio: capabilities?.supportGenerateAudio === false ? undefined : form.generateAudio,
    cameraFixed: capabilities?.supportCameraFixed === false ? undefined : form.cameraFixed,
    seed: capabilities?.supportSeed === false ? undefined : form.seed.trim() ? Number(form.seed) : undefined,
    category: "manual_video_generation",
  };
  return input;
}
