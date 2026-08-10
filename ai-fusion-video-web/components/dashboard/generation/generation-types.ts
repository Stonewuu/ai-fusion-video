import type {
  ImageGenerationItem,
  ImageGenerationTask,
  VideoGenerationItem,
  VideoGenerationTask,
} from "@/lib/api/generation";

export type WorkbenchMode = "image" | "video";
export type ComposerMode = "simple" | "advanced";
export type GenerationTask = ImageGenerationTask | VideoGenerationTask;
export type GenerationResultItem = ImageGenerationItem | VideoGenerationItem;

export interface GenerationHistoryEntry {
  task: GenerationTask;
  items: GenerationResultItem[];
}

export interface GenerationFormState {
  prompt: string;
  ratio: string;
  resolution: string;
  count: number;
  duration: number;
  seed: string;
  watermark: boolean;
  generateAudio: boolean;
  cameraFixed: boolean;
  referenceImages: string[];
  firstFrame: string[];
  lastFrame: string[];
  referenceVideos: string[];
  referenceAudios: string[];
}

export type SimpleAttachmentKind =
  | "firstFrame"
  | "lastFrame"
  | "referenceImage"
  | "referenceVideo"
  | "referenceAudio";

export type SimpleAttachmentMediaType = "image" | "video" | "audio";

export interface SimpleAttachment {
  url: string;
  label: string;
  kind: SimpleAttachmentKind;
  mediaType: SimpleAttachmentMediaType;
}

export interface SimpleAttachmentUploadOption {
  kind: SimpleAttachmentKind;
  label: string;
  mediaType: SimpleAttachmentMediaType;
  remaining: number;
  disabledReason?: string;
}

export interface GenerationAssetTarget {
  mode: WorkbenchMode;
  item: GenerationResultItem;
  prompt: string;
}

export const EMPTY_GENERATION_FORM: GenerationFormState = {
  prompt: "",
  ratio: "",
  resolution: "",
  count: 1,
  duration: 5,
  seed: "",
  watermark: false,
  generateAudio: false,
  cameraFixed: false,
  referenceImages: [],
  firstFrame: [],
  lastFrame: [],
  referenceVideos: [],
  referenceAudios: [],
};
