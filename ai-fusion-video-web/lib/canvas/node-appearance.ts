import type { CanvasNodeKind } from "./types";

type CanvasTone =
  | "text"
  | "image"
  | "video"
  | "audio"
  | "business"
  | "flow"
  | "note"
  | "neutral";
const tones: Record<CanvasNodeKind, CanvasTone> = {
  story: "text",
  text: "text",
  textFile: "text",
  textCombine: "text",
  textGeneration: "text",
  agent: "text",
  mcpTool: "flow",
  comfyui: "text",
  image: "image",
  imageGeneration: "image",
  imageTransform: "image",
  frameGenerate: "image",
  frameExtract: "image",
  assetImageGenerate: "image",
  video: "video",
  videoGeneration: "video",
  videoCompose: "video",
  audio: "audio",
  musicGeneration: "audio",
  sfxGeneration: "audio",
  audioMix: "audio",
  project: "business",
  script: "business",
  scriptEpisode: "business",
  scriptScene: "business",
  storyboard: "business",
  storyboardEpisode: "business",
  storyboardScene: "business",
  shot: "business",
  asset: "business",
  assetItem: "business",
  document: "business",
  scriptParse: "business",
  storyToScript: "business",
  episodeParse: "business",
  storyboardGenerate: "business",
  assetExtract: "business",
  selection: "flow",
  collection: "flow",
  foreach: "flow",
  writeback: "business",
  export: "flow",
  note: "note",
  group: "neutral",
};
export function canvasNodeColor(kind: CanvasNodeKind): string {
  return `var(--canvas-${tones[kind]})`;
}
