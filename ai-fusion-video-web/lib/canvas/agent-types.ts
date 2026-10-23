import type { CanvasNodeKind } from "./types";

export const canvasAgentKinds: readonly CanvasNodeKind[] = [
  "agent", "scriptParse", "storyToScript", "episodeParse", "storyboardGenerate", "assetExtract", "assetImageGenerate", "frameGenerate",
];
export const imageAgentTypes = new Set(["ai_media", "asset_image_gen", "asset_image_executor", "storyboard_frame_gen", "storyboard_frame_executor"]);
export const videoAgentTypes = new Set(["ai_media", "storyboard_video_gen", "storyboard_video_executor"]);
export const agentFormFields = new Set(["agentType", "imageModelId", "videoModelId", "maxImages", "maxVideoSeconds", "maxAgentRuns", "inputType", "resultType", "skillMode", "skillNames"]);
