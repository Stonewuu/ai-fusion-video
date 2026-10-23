import type { CanvasNode, CanvasResource } from "./types";

export interface CanvasMediaPreview {
  source: string;
  kind: "image" | "video" | "audio" | "textFile" | "document";
  title: string;
}
export function nodeMedia(
  node: CanvasNode,
  resource: CanvasResource | null,
): CanvasMediaPreview | null {
  if (
    ["image", "video", "audio", "textFile", "document"].includes(node.kind) &&
    typeof node.config.resourceId === "string" &&
    node.config.resourceId
  ) {
    return {
      source: `media://${node.config.resourceId}`,
      kind: node.kind as CanvasMediaPreview["kind"],
      title: node.title,
    };
  }
  const data = resource?.data;
  if (!data) return null;
  if (node.kind === "shot") {
    if (typeof data.videoUrl === "string" && data.videoUrl)
      return { source: data.videoUrl, kind: "video", title: node.title };
    if (typeof data.generatedVideoUrl === "string" && data.generatedVideoUrl)
      return {
        source: data.generatedVideoUrl,
        kind: "video",
        title: node.title,
      };
    if (typeof data.firstFrameImageUrl === "string" && data.firstFrameImageUrl)
      return {
        source: data.firstFrameImageUrl,
        kind: "image",
        title: node.title,
      };
  }
  if (
    node.kind === "storyboardEpisode" &&
    typeof data.composedVideoUrl === "string" &&
    data.composedVideoUrl
  )
    return { source: data.composedVideoUrl, kind: "video", title: node.title };
  const imageField = node.kind === "assetItem" ? "imageUrl" : "coverUrl";
  if (typeof data[imageField] === "string" && data[imageField])
    return { source: data[imageField], kind: "image", title: node.title };
  return null;
}

export function nodeSummary(
  node: CanvasNode,
  resource: CanvasResource | null,
): string {
  if (node.kind === "text" || node.kind === "story" || node.kind === "note")
    return String(node.config.content ?? "");
  if (node.kind === "mcpTool") {
    if (!node.config.contractId) return "选择 MCP 操作并填写参数";
    const input = node.config.inputMode === "TEXT_FIELD" ? "接收文本"
      : node.config.inputMode === "JSON_OBJECT" ? "接收 JSON" : "使用固定参数";
    return `操作版本 ${Number(node.config.contractVersion) + 1} · ${input}`;
  }
  if (!resource)
    return String(node.config.prompt ?? node.config.instruction ?? "");
  const field = {
    project: "description",
    script: "summary",
    scriptEpisode: "synopsis",
    scriptScene: "sceneDescription",
    storyboard: "description",
    storyboardEpisode: "synopsis",
    storyboardScene: "sceneHeading",
    shot: "content",
    asset: "description",
    assetItem: "description",
  }[node.kind as string];
  return field && typeof resource.data[field] === "string"
    ? (resource.data[field] as string)
    : "";
}
