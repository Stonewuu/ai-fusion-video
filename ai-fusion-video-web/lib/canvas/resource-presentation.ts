import type { CanvasResource } from "./types";
import { canvasTypeNames } from "./labels";

export function resourceTitle(resource: CanvasResource): string {
  const type = resource.header.resourceType;
  const data = resource.data;
  const key = {
    project: "name",
    script: "title",
    scriptEpisode: "title",
    scriptScene: "sceneHeading",
    asset: "name",
    assetItem: "name",
    storyboard: "title",
    storyboardEpisode: "title",
    storyboardScene: "sceneHeading",
    shot: "shotNumber",
  }[type];
  const label =
    typeof data[key] === "string" && data[key]
      ? String(data[key])
      : canvasTypeNames[type];
  return type === "shot"
    ? `${label} · ${String(data.content ?? "未填写画面").slice(0, 60)}`
    : label;
}

export function resourceKey(resource: CanvasResource) {
  return `${resource.header.resourceType}:${resource.header.resourceId}`;
}
